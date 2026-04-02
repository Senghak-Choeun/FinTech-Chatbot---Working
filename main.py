import argparse
import json

from processing.downloader import Banking77Downloader
from processing.preprocessor import FintechDatasetProcessor
from processing.trainers import ClassicalTrainer, TransferTrainer


def cmd_download(args) -> None:
    stats = Banking77Downloader(output_dir=args.output_dir).download()
    print(json.dumps(stats, indent=2, ensure_ascii=True))


def cmd_process(args) -> None:
    processor = FintechDatasetProcessor(random_seed=args.random_seed)
    stats = processor.process(
        output_dir=args.output_dir,
        regional_aug=args.regional_aug,
        bitext_path=args.bitext_path,
        kaggle_path=args.kaggle_path,
        max_banking77_rows=args.max_banking77_rows,
        banking77_raw_dir=args.banking77_raw_dir,
        min_samples_per_intent=args.min_samples_per_intent,
        no_synthetic_fill=args.no_synthetic_fill,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=True))


def cmd_train_classical(args) -> None:
    trainer = ClassicalTrainer()
    stats = trainer.train(
        data=args.data,
        model=args.model,
        text_col=args.text_col,
        label_col=args.label_col,
        response_col=args.response_col,
        test_size=args.test_size,
        random_state=args.random_state,
        output_dir=args.output_dir,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=True))


def cmd_train_intent(args) -> None:
    trainer = TransferTrainer()
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
    print(json.dumps(metrics, indent=2, ensure_ascii=True))


def cmd_train_gpt(args) -> None:
    trainer = TransferTrainer()
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
    print(json.dumps(metrics, indent=2, ensure_ascii=True))


def cmd_all(args) -> None:
    download_stats = Banking77Downloader(output_dir=args.raw_output_dir).download()
    print("Download complete:")
    print(json.dumps(download_stats, indent=2, ensure_ascii=True))

    processor = FintechDatasetProcessor(random_seed=args.random_seed)
    process_stats = processor.process(
        output_dir=args.processed_output_dir,
        regional_aug=args.regional_aug,
        bitext_path=args.bitext_path,
        kaggle_path=args.kaggle_path,
        banking77_raw_dir=args.raw_output_dir,
        max_banking77_rows=args.max_banking77_rows,
        min_samples_per_intent=args.min_samples_per_intent,
        no_synthetic_fill=args.no_synthetic_fill,
    )
    print("Processing complete:")
    print(json.dumps(process_stats, indent=2, ensure_ascii=True))

    trainer = ClassicalTrainer()
    classical_stats = trainer.train(
        data=process_stats["files"]["train"],
        model=args.classical_model,
        output_dir=args.classical_output_dir,
    )
    print("Classical training complete:")
    print(json.dumps(classical_stats, indent=2, ensure_ascii=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FinTech chatbot pipeline runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download BANKING77 to local raw folder")
    download.add_argument("--output_dir", type=str, default="dataset/raw/banking77")
    download.set_defaults(func=cmd_download)

    process = subparsers.add_parser("process", help="Preprocess datasets into template-ready outputs")
    process.add_argument("--output_dir", type=str, default="dataset/processed")
    process.add_argument("--regional_aug", type=str, default="dataset/regional_augmentation_template.csv")
    process.add_argument("--bitext_path", type=str, default="")
    process.add_argument("--kaggle_path", type=str, default="")
    process.add_argument("--banking77_raw_dir", type=str, default="dataset/raw/banking77")
    process.add_argument("--max_banking77_rows", type=int, default=0)
    process.add_argument("--min_samples_per_intent", type=int, default=40)
    process.add_argument("--no_synthetic_fill", action="store_true")
    process.add_argument("--random_seed", type=int, default=42)
    process.set_defaults(func=cmd_process)

    train_classical = subparsers.add_parser("train-classical", help="Train Naive Bayes/LogReg/SVM")
    train_classical.add_argument("--data", type=str, default="dataset/processed/fintech_intents_train.csv")
    train_classical.add_argument("--model", type=str, default="logreg", choices=["logreg", "naive_bayes", "svm"])
    train_classical.add_argument("--text_col", type=str, default="text")
    train_classical.add_argument("--label_col", type=str, default="intent")
    train_classical.add_argument("--response_col", type=str, default="response")
    train_classical.add_argument("--test_size", type=float, default=0.2)
    train_classical.add_argument("--random_state", type=int, default=42)
    train_classical.add_argument("--output_dir", type=str, default="models/classical")
    train_classical.set_defaults(func=cmd_train_classical)

    train_intent = subparsers.add_parser("train-intent", help="Train BERT intent classifier")
    train_intent.add_argument("--data", type=str, default="dataset/processed/fintech_intents_train.csv")
    train_intent.add_argument("--text_col", type=str, default="text")
    train_intent.add_argument("--label_col", type=str, default="intent")
    train_intent.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    train_intent.add_argument("--epochs", type=int, default=3)
    train_intent.add_argument("--batch_size", type=int, default=8)
    train_intent.add_argument("--learning_rate", type=float, default=2e-5)
    train_intent.add_argument("--max_length", type=int, default=128)
    train_intent.add_argument("--test_size", type=float, default=0.2)
    train_intent.add_argument("--random_state", type=int, default=42)
    train_intent.add_argument("--output_dir", type=str, default="models/bert_intent")
    train_intent.set_defaults(func=cmd_train_intent)

    train_gpt = subparsers.add_parser("train-gpt", help="Train GPT-style model")
    train_gpt.add_argument("--data", type=str, default="dataset/processed/fintech_gpt_train.jsonl")
    train_gpt.add_argument("--prompt_col", type=str, default="prompt")
    train_gpt.add_argument("--response_col", type=str, default="response")
    train_gpt.add_argument("--model_name", type=str, default="distilgpt2")
    train_gpt.add_argument("--epochs", type=int, default=2)
    train_gpt.add_argument("--batch_size", type=int, default=4)
    train_gpt.add_argument("--learning_rate", type=float, default=5e-5)
    train_gpt.add_argument("--max_length", type=int, default=192)
    train_gpt.add_argument("--test_size", type=float, default=0.1)
    train_gpt.add_argument("--random_state", type=int, default=42)
    train_gpt.add_argument("--output_dir", type=str, default="models/gpt_finetuned")
    train_gpt.set_defaults(func=cmd_train_gpt)

    all_cmd = subparsers.add_parser("all", help="Run download + process + classical training")
    all_cmd.add_argument("--raw_output_dir", type=str, default="dataset/raw/banking77")
    all_cmd.add_argument("--processed_output_dir", type=str, default="dataset/processed")
    all_cmd.add_argument("--regional_aug", type=str, default="dataset/regional_augmentation_template.csv")
    all_cmd.add_argument("--bitext_path", type=str, default="")
    all_cmd.add_argument("--kaggle_path", type=str, default="")
    all_cmd.add_argument("--max_banking77_rows", type=int, default=0)
    all_cmd.add_argument("--min_samples_per_intent", type=int, default=40)
    all_cmd.add_argument("--no_synthetic_fill", action="store_true")
    all_cmd.add_argument("--classical_model", type=str, default="logreg", choices=["logreg", "naive_bayes", "svm"])
    all_cmd.add_argument("--classical_output_dir", type=str, default="models/classical")
    all_cmd.add_argument("--random_seed", type=int, default=42)
    all_cmd.set_defaults(func=cmd_all)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
