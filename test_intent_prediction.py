import argparse
import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Fallback thresholds (must match chatbot.py)
CONFIDENCE_THRESHOLD = 0.40  # Below this: clarification fallback
SIMILARITY_THRESHOLD = 0.60  # Below this: generic intent response

CLARIFICATION_FALLBACK = "I'm not fully sure I understood. Could you rephrase or provide more details about what you need?"

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


def load_classical_model(model_path: str):
    """Load a trained classical model pipeline."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    print(f"  Loading model from: {os.path.abspath(model_path)}")
    return joblib.load(model_path)


def load_responses(responses_path: str) -> dict:
    """Load intent responses mapping."""
    if not os.path.exists(responses_path):
        raise FileNotFoundError(f"Responses file not found: {responses_path}")
    print(f"  Loading responses from: {os.path.abspath(responses_path)}")
    with open(responses_path, "r") as f:
        return json.load(f)


def load_bert_model(model_dir: str):
    """Load a trained BERT model."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    
    # Load label mapping
    label_classes_path = os.path.join(model_dir, "label_classes.json")
    if os.path.exists(label_classes_path):
        with open(label_classes_path, "r") as f:
            label_classes = json.load(f)
    else:
        label_classes = {}
    
    return tokenizer, model, label_classes


def predict_classical(model, text: str) -> Tuple[str, float]:
    """Make prediction using classical model."""
    prediction = model.predict([text])[0]
    
    # Try to get probabilities (works for Naive Bayes, Logistic Regression)
    # Fall back to decision function for SVM
    try:
        probabilities = model.predict_proba([text])[0]
        confidence = max(probabilities)
    except AttributeError:
        # SVM doesn't have predict_proba, use decision_function instead
        decisions = model.decision_function([text])[0]
        # Get the decision value for the predicted class (highest value)
        predicted_class_idx = np.argmax(decisions)
        # Normalize to [0, 1] range using sigmoid
        confidence = 1 / (1 + np.exp(-decisions[predicted_class_idx]))
    
    return prediction, confidence


def predict_bert(tokenizer, model, text: str, device: str = "cpu") -> Tuple[str, float]:
    """Make prediction using BERT model."""
    import torch
    
    model.eval()
    model.to(device)
    
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        pred_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][pred_idx].item()
    
    return pred_idx, confidence


def build_tfidf_cache(responses: dict):
    """Build TF-IDF vectorizer cache for all intents - SAME AS CHATBOT."""
    tfidf_cache = {}
    for intent, examples in responses.items():
        if examples:
            training_texts = [ex["text"] for ex in examples]
            vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3))
            vectorizer.fit(training_texts)
            tfidf_cache[intent] = {
                "vectorizer": vectorizer,
                "vectors": vectorizer.transform(training_texts),
                "examples": examples,
            }
    return tfidf_cache


def get_most_similar_response(user_text: str, intent: str, tfidf_cache: dict):
    """Find most similar training example and return response - SAME AS CHATBOT."""
    if intent not in tfidf_cache:
        return None, None, 0.0
    
    cache = tfidf_cache[intent]
    user_vector = cache["vectorizer"].transform([user_text])
    similarities = cosine_similarity(user_vector, cache["vectors"])[0]
    best_idx = similarities.argmax()
    best_similarity = similarities[best_idx]
    
    if best_similarity < 0.1:
        # Fallback to first example if similarity is too low
        return cache["examples"][0]["response"], cache["examples"][0]["text"], best_similarity
    
    return cache["examples"][best_idx]["response"], cache["examples"][best_idx]["text"], best_similarity


def process_single_question(user_input: str, args, model, tokenizer=None, id2label=None, tfidf_cache=None):
    """Process a single question and return result with fallback logic."""
    if args.model_type == "classical":
        predicted_intent, confidence = predict_classical(model, user_input)
    else:
        pred_idx, confidence = predict_bert(tokenizer, model, user_input, device=args.device)
        predicted_intent = id2label.get(int(pred_idx), str(pred_idx)) if id2label else str(pred_idx)
    
    # Apply fallback logic
    fallback_type = "none"
    
    # Fallback 1: Confidence too low
    if confidence < CONFIDENCE_THRESHOLD:
        response = CLARIFICATION_FALLBACK
        similar_question = "N/A (confidence too low)"
        similarity = 0.0
        fallback_type = "confidence"
    else:
        # Find most similar training example
        response, similar_question, similarity = get_most_similar_response(user_input, predicted_intent, tfidf_cache)
        
        # Fallback 2: Similarity too low - use generic intent response
        if similarity < SIMILARITY_THRESHOLD:
            generic = GENERIC_INTENT_RESPONSES.get(predicted_intent, response)
            if generic != response:
                response = generic
                fallback_type = "similarity"
    
    return {
        "question": user_input,
        "predicted_intent": predicted_intent,
        "intent_confidence": f"{confidence:.4f}",
        "similar_question": similar_question,
        "similarity_score": f"{similarity:.4f}",
        "response": response,
        "fallback_type": fallback_type
    }


def main():
    parser = argparse.ArgumentParser(description="Check what intent a user question maps to")
    parser.add_argument("--model_type", type=str, choices=["classical", "bert"], default="classical")
    parser.add_argument("--model_path", type=str, default="models/classical/svm_pipeline.joblib")
    parser.add_argument("--responses_path", type=str, default="models/classical/responses_by_intent.json")
    parser.add_argument("--question", type=str, default="", help="Single question to classify")
    parser.add_argument("--batch_file", type=str, default="", help="File with multiple questions (one per line)")
    parser.add_argument("--output_file", type=str, default="", help="Save results to file (optional)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    
    args = parser.parse_args()
    
    # Load model and responses
    print(f"Loading {args.model_type.upper()} model...")
    
    if args.model_type == "classical":
        model = load_classical_model(args.model_path)
        responses = load_responses(args.responses_path)
        id2label = None
    else:  # bert
        tokenizer, model, label_classes = load_bert_model(args.model_path)
        id2label = {v: k for k, v in label_classes.items()}
        responses = load_responses(args.responses_path) if os.path.exists(args.responses_path) else {}
    
    # Build TF-IDF cache (like the chatbot does)
    print("Building TF-IDF similarity cache...")
    tfidf_cache = build_tfidf_cache(responses)
    
    print(f"Model loaded successfully!\n")
    
    # Batch file mode
    if args.batch_file:
        if not os.path.exists(args.batch_file):
            print(f"Error: Batch file not found: {args.batch_file}")
            return
        
        print(f"Processing batch file: {args.batch_file}\n")
        with open(args.batch_file, "r", encoding="utf-8") as f:
            questions = [line.strip() for line in f if line.strip()]
        
        results = []
        for i, question in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}] Processing: {question}")
            
            result = process_single_question(question, args, model if args.model_type == "classical" else None, 
                                            tokenizer if args.model_type == "bert" else None, 
                                            id2label, tfidf_cache)
            results.append(result)
            
            # Show result with fallback indicator
            fallback_indicator = ""
            if result['fallback_type'] == 'confidence':
                fallback_indicator = " [FALLBACK: Low Confidence]"
            elif result['fallback_type'] == 'similarity':
                fallback_indicator = " [FALLBACK: Low Similarity]"
            
            print(f"  → Intent: {result['predicted_intent']} ({float(result['intent_confidence'])*100:.2f}%){fallback_indicator}")
            print(f"  → Similar to: {result['similar_question']} ({float(result['similarity_score'])*100:.2f}% match)")
            print(f"  → Response: {result['response'][:100]}..." if len(result['response']) > 100 else f"  → Response: {result['response']}")
            print()
        
        # Save results if output file specified
        if args.output_file:
            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"✓ Results saved to: {args.output_file}")
        
        return
    
    # Single question mode
    if args.question:
        result = process_single_question(args.question, args, model if args.model_type == "classical" else None,
                                        tokenizer if args.model_type == "bert" else None,
                                        id2label, tfidf_cache)
        
        print(f"\nQuestion: {result['question']}")
        print(f"Predicted Intent: {result['predicted_intent']}")
        print(f"Intent Confidence: {result['intent_confidence']} ({float(result['intent_confidence'])*100:.2f}%)")
        
        print(f"\n🔍 Most Similar Training Question:")
        print(f"   {result['similar_question']}")
        print(f"   Similarity Score: {result['similarity_score']} ({float(result['similarity_score'])*100:.2f}%)")
        print(f"\n📝 Response:")
        print(f"{result['response']}")
        
        return
    
    # Interactive mode
    print("Interactive Mode - Type 'quit' or 'exit' to stop\n")
    
    while True:
        user_input = input("Enter user question: ").strip()
        
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        
        if not user_input:
            print("Please enter a valid question.\n")
            continue
        
        # Make prediction
        result = process_single_question(user_input, args, model if args.model_type == "classical" else None,
                                        tokenizer if args.model_type == "bert" else None,
                                        id2label, tfidf_cache)
        
        print(f"\n✓ Predicted Intent: {result['predicted_intent']}")
        print(f"  Intent Confidence: {result['intent_confidence']} ({float(result['intent_confidence'])*100:.2f}%)")
        
        # Find most similar training example (same as chatbot does)
        print(f"\n  🔍 Most Similar Training Question:")
        print(f"     {result['similar_question']}")
        print(f"     Similarity Score: {result['similarity_score']} ({float(result['similarity_score'])*100:.2f}%)")
        print(f"\n  📝 Response:")
        print(f"     {result['response']}")
        
        print()


if __name__ == "__main__":
    main()
