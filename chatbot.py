import argparse
import json
import os
import numpy as np

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Generic fallback responses for each intent when similarity is too low
GENERIC_INTENT_RESPONSES = {
    "fund_transfer": "To transfer money, please provide more details about which method you'd like to use (ABA, Wing, Bakong, or KHQR). You can also check your transfer limits and account balance in the app settings.",
    "transaction_status": "To check your transaction status, go to your transaction history in the app. You can filter by date range or search for specific transactions. If you need more details, please provide the transaction date or amount.",
    "card_issue": "For card-related issues, you can freeze or block your card from the app settings immediately. For other issues like lost cards or billing problems, please contact customer support for immediate assistance.",
    "cash_withdrawal": "For ATM cash withdrawal issues, check that you have sufficient balance and are within daily withdrawal limits. If the issue persists, try a different ATM or contact your bank's customer support.",
    "wallet_topup": "To top up your wallet, select 'Add Money' in the app and choose your preferred payment method. Ensure you have sufficient balance in your linked account and check the applicable limits.",
    "fraud_report": "We take fraud seriously. If you believe your account or card has been compromised, please report it immediately through the app or contact customer support. Your account can be secured right away.",
    "kyc_verification": "For identity verification, ensure your documents are clear, current, and match the information in your profile. If verification fails, try uploading again or contact support for assistance.",
    "fee_inquiry": "Transfer fees vary depending on the method used (ABA, Wing, Bakong) and your account type. Check the app's fee schedule or transaction history to see the exact fee for your transfer.",
}

# Confidence thresholds
CONFIDENCE_THRESHOLD = 0.40  # Below this: clarification fallback
SIMILARITY_THRESHOLD = 0.60  # Below this: generic intent response

CLARIFICATION_FALLBACK = "I'm not fully sure I understood. Could you rephrase or provide more details about what you need?"


class ClassicalChatbot:
    def __init__(self, model_path: str, responses_path: str):
        self.pipeline = joblib.load(model_path)
        with open(responses_path, "r", encoding="utf-8") as f:
            self.responses_by_intent = json.load(f)
        self._build_tfidf()

    def _build_tfidf(self) -> None:
        """Build TF-IDF vectorizer and cache for all training texts per intent."""
        self.tfidf_cache = {}
        for intent, examples in self.responses_by_intent.items():
            if examples:
                training_texts = [ex["text"] for ex in examples]
                vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3))
                vectorizer.fit(training_texts)
                self.tfidf_cache[intent] = {
                    "vectorizer": vectorizer,
                    "vectors": vectorizer.transform(training_texts),
                    "examples": examples,
                }

    def _get_confidence(self, user_text: str) -> float:
        """Extract confidence from model prediction."""
        try:
            probabilities = self.pipeline.predict_proba([user_text])[0]
            return float(max(probabilities))
        except AttributeError:
            # SVM doesn't have predict_proba, use decision_function
            decisions = self.pipeline.decision_function([user_text])[0]
            predicted_class_idx = np.argmax(decisions)
            confidence = 1 / (1 + np.exp(-decisions[predicted_class_idx]))
            return float(confidence)

    def _get_best_response(self, user_text: str, intent: str, confidence: float) -> tuple:
        """
        Find most similar training example and return response with fallback logic.
        
        Returns:
            (response: str, fallback_type: str) - fallback_type is 'none', 'confidence', 'similarity', or 'generic_intent'
        """
        # Fallback 1: Confidence too low
        if confidence < CONFIDENCE_THRESHOLD:
            return CLARIFICATION_FALLBACK, "confidence"
        
        # Fallback 2: Intent has no training data
        if intent not in self.tfidf_cache:
            generic = GENERIC_INTENT_RESPONSES.get(intent, f"I don't have specific information about that. Please try rephrasing your question.")
            return generic, "generic_intent"
        
        cache = self.tfidf_cache[intent]
        user_vector = cache["vectorizer"].transform([user_text])
        similarities = cosine_similarity(user_vector, cache["vectors"])[0]
        best_idx = similarities.argmax()
        best_similarity = similarities[best_idx]
        
        # Fallback 3: Similarity too low - use generic intent response
        if best_similarity < SIMILARITY_THRESHOLD:
            generic = GENERIC_INTENT_RESPONSES.get(intent, cache["examples"][0]["response"])
            return generic, "similarity"
        
        # Success: Use matched response
        return cache["examples"][best_idx]["response"], "none"

    def reply(self, user_text: str) -> str:
        intent = self.pipeline.predict([user_text])[0]
        confidence = self._get_confidence(user_text)
        response, _ = self._get_best_response(user_text, intent, confidence)
        return response


class BertIntentChatbot:
    def __init__(self, model_dir: str, responses_path: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        
        # Normalize path to OS-native format
        model_dir = os.path.normpath(model_dir.strip())
        best_model_path = os.path.normpath(os.path.join(model_dir, "best_model"))
        
        # Use best_model if it exists, otherwise use model_dir as-is
        if os.path.isdir(best_model_path):
            load_path = best_model_path
            print(f"Loading BERT from: {load_path}")
        else:
            load_path = model_dir
            print(f"Loading BERT from: {load_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(load_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(load_path)
        except Exception as e:
            raise ValueError(
                f"Failed to load BERT model from {load_path}. "
                f"Make sure the model is trained and exists at that location. "
                f"Error: {str(e)}"
            )
        
        # Load label classes - check best_model first, then parent directory
        label_classes_path = os.path.join(load_path, "label_classes.json")
        
        # If not found in load_path, check parent directory
        if not os.path.exists(label_classes_path):
            parent_path = os.path.dirname(load_path)
            label_classes_path = os.path.join(parent_path, "label_classes.json")
        
        if not os.path.exists(label_classes_path):
            raise FileNotFoundError(
                f"label_classes.json not found. "
                f"Checked: {os.path.join(load_path, 'label_classes.json')} "
                f"and {os.path.join(os.path.dirname(load_path), 'label_classes.json')}"
            )
        
        with open(label_classes_path, "r", encoding="utf-8") as f:
            self.label_classes = json.load(f)
        
        with open(responses_path, "r", encoding="utf-8") as f:
            self.responses_by_intent = json.load(f)
        self._build_tfidf()

    def _build_tfidf(self) -> None:
        """Build TF-IDF vectorizer and cache for all training texts per intent."""
        self.tfidf_cache = {}
        for intent, examples in self.responses_by_intent.items():
            if examples:
                training_texts = [ex["text"] for ex in examples]
                vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3))
                vectorizer.fit(training_texts)
                self.tfidf_cache[intent] = {
                    "vectorizer": vectorizer,
                    "vectors": vectorizer.transform(training_texts),
                    "examples": examples,
                }

    def _get_best_response(self, user_text: str, intent: str, confidence: float) -> tuple:
        """
        Find most similar training example and return response with fallback logic.
        
        Returns:
            (response: str, fallback_type: str) - fallback_type is 'none', 'confidence', 'similarity', or 'generic_intent'
        """
        # Fallback 1: Confidence too low
        if confidence < CONFIDENCE_THRESHOLD:
            return CLARIFICATION_FALLBACK, "confidence"
        
        # Fallback 2: Intent has no training data
        if intent not in self.tfidf_cache:
            generic = GENERIC_INTENT_RESPONSES.get(intent, f"I don't have specific information about that. Please try rephrasing your question.")
            return generic, "generic_intent"
        
        cache = self.tfidf_cache[intent]
        user_vector = cache["vectorizer"].transform([user_text])
        similarities = cosine_similarity(user_vector, cache["vectors"])[0]
        best_idx = similarities.argmax()
        best_similarity = similarities[best_idx]
        
        # Fallback 3: Similarity too low - use generic intent response
        if best_similarity < SIMILARITY_THRESHOLD:
            generic = GENERIC_INTENT_RESPONSES.get(intent, cache["examples"][0]["response"])
            return generic, "similarity"
        
        # Success: Use matched response
        return cache["examples"][best_idx]["response"], "none"

    def reply(self, user_text: str) -> str:
        inputs = self.tokenizer(user_text, return_tensors="pt", truncation=True, padding=True)
        with self._torch.no_grad():
            logits = self.model(**inputs).logits
            probabilities = self._torch.softmax(logits, dim=-1)
        
        pred_id = int(self._torch.argmax(logits, dim=-1).item())
        confidence = float(probabilities[0][pred_id].item())
        intent = self.label_classes[pred_id]

        response, _ = self._get_best_response(user_text, intent, confidence)
        return response


class GPTChatbot:
    def __init__(self, model_dir: str, max_new_tokens: int = 80):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir)
        self.max_new_tokens = max_new_tokens

    def reply(self, user_text: str) -> str:
        prompt = f"User: {user_text}\nAssistant:"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if "Assistant:" in text:
            return text.split("Assistant:", 1)[1].strip()
        return text


def run_chat(chatbot) -> None:
    print("Type 'exit' to stop chat.")
    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in {"exit", "quit"}:
            print("Bye.")
            break
        if not user_text:
            continue
        response = chatbot.reply(user_text)
        print(f"Bot: {response}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fintech chatbot inference")
    parser.add_argument("--backend", choices=["classical", "bert", "gpt"], required=True)
    parser.add_argument("--model_path", type=str, default="models/classical/naive_bayes_pipeline.joblib")
    parser.add_argument("--model_dir", type=str, default="models/bert_intent")
    parser.add_argument("--responses_path", type=str, default="models/classical/responses_by_intent.json")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    args = parser.parse_args()

    if args.backend == "classical":
        bot = ClassicalChatbot(model_path=args.model_path, responses_path=args.responses_path)
    elif args.backend == "bert":
        bot = BertIntentChatbot(model_dir=args.model_dir, responses_path=args.responses_path)
    else:
        bot = GPTChatbot(model_dir=args.model_dir, max_new_tokens=args.max_new_tokens)

    run_chat(bot)


if __name__ == "__main__":
    main()
