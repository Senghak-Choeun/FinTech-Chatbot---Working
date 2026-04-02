import argparse
import json

from processing.trainers import ClassicalTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train classical ML intent classifiers for fintech chatbot")
    parser.add_argument("--data", type=str, default="dataset/processed/fintech_intents_train.csv")
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="intent")
    parser.add_argument("--response_col", type=str, default="response")
    parser.add_argument("--model", type=str, default="logreg", choices=["logreg", "naive_bayes", "svm"])
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="models/classical")
    args = parser.parse_args()

    stats = ClassicalTrainer().train(
        data=args.data,
        model=args.model,
        text_col=args.text_col,
        label_col=args.label_col,
        response_col=args.response_col,
        test_size=args.test_size,
        random_state=args.random_state,
        output_dir=args.output_dir,
    )

    print(f"Model: {args.model}")
    print(f"Accuracy: {stats['accuracy']:.4f}")
    print(stats["classification_report"])
    print(json.dumps({k: v for k, v in stats.items() if k != "classification_report"}, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
