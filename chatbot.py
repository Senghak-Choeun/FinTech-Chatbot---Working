import argparse
import json
import random

import joblib
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer


class ClassicalChatbot:
    def __init__(self, model_path: str, responses_path: str):
        self.pipeline = joblib.load(model_path)
        with open(responses_path, "r", encoding="utf-8") as f:
            self.responses_by_intent = json.load(f)

    def reply(self, user_text: str) -> str:
        intent = self.pipeline.predict([user_text])[0]
        responses = self.responses_by_intent.get(intent, [])
        if responses:
            return random.choice(responses)
        return f"Predicted intent: {intent}. Please add responses for this intent in responses_by_intent.json"


class BertIntentChatbot:
    def __init__(self, model_dir: str, responses_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        with open(f"{model_dir}/label_classes.json", "r", encoding="utf-8") as f:
            self.label_classes = json.load(f)
        with open(responses_path, "r", encoding="utf-8") as f:
            self.responses_by_intent = json.load(f)

    def reply(self, user_text: str) -> str:
        inputs = self.tokenizer(user_text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        pred_id = int(torch.argmax(logits, dim=-1).item())
        intent = self.label_classes[pred_id]

        responses = self.responses_by_intent.get(intent, [])
        if responses:
            return random.choice(responses)
        return f"Predicted intent: {intent}. Please add responses for this intent in responses_by_intent.json"


class GPTChatbot:
    def __init__(self, model_dir: str, max_new_tokens: int = 80):
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
    parser.add_argument("--model_path", type=str, default="models/classical/logreg_pipeline.joblib")
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
