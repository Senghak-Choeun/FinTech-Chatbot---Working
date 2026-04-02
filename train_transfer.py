import argparse
import json

from processing.trainers import TransferTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train transfer-learning models for fintech chatbot")
    subparsers = parser.add_subparsers(dest="task", required=True)

    bert_parser = subparsers.add_parser("intent", help="Fine-tune BERT-like encoder for intent classification")
    bert_parser.add_argument("--data", type=str, default="dataset/processed/fintech_intents_train.csv")
    bert_parser.add_argument("--text_col", type=str, default="text")
    bert_parser.add_argument("--label_col", type=str, default="intent")
    bert_parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    bert_parser.add_argument("--epochs", type=int, default=3)
    bert_parser.add_argument("--batch_size", type=int, default=8)
    bert_parser.add_argument("--learning_rate", type=float, default=2e-5)
    bert_parser.add_argument("--max_length", type=int, default=128)
    bert_parser.add_argument("--test_size", type=float, default=0.2)
    bert_parser.add_argument("--random_state", type=int, default=42)
    bert_parser.add_argument("--output_dir", type=str, default="models/bert_intent")

    gpt_parser = subparsers.add_parser("gpt", help="Fine-tune GPT-like causal LM on prompt-response pairs")
    gpt_parser.add_argument("--data", type=str, default="dataset/processed/fintech_gpt_train.jsonl")
    gpt_parser.add_argument("--prompt_col", type=str, default="prompt")
    gpt_parser.add_argument("--response_col", type=str, default="response")
    gpt_parser.add_argument("--model_name", type=str, default="distilgpt2")
    gpt_parser.add_argument("--epochs", type=int, default=2)
    gpt_parser.add_argument("--batch_size", type=int, default=4)
    gpt_parser.add_argument("--learning_rate", type=float, default=5e-5)
    gpt_parser.add_argument("--max_length", type=int, default=192)
    gpt_parser.add_argument("--test_size", type=float, default=0.1)
    gpt_parser.add_argument("--random_state", type=int, default=42)
    gpt_parser.add_argument("--output_dir", type=str, default="models/gpt_finetuned")

    args = parser.parse_args()
    trainer = TransferTrainer()

    if args.task == "intent":
        metrics = trainer.train_bert_intent(
            data=args.data,
            text_col=args.text_col,
            label_col=args.label_col,
            model_name=args.model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            test_size=args.test_size,
            random_state=args.random_state,
            output_dir=args.output_dir,
        )
        print("BERT intent classifier training completed.")
    else:
        metrics = trainer.train_gpt(
            data=args.data,
            prompt_col=args.prompt_col,
            response_col=args.response_col,
            model_name=args.model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            test_size=args.test_size,
            random_state=args.random_state,
            output_dir=args.output_dir,
        )
        print("GPT-style model training completed.")

    print(json.dumps(metrics, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
