import argparse
import json
import os

from processing.downloader import Banking77Downloader
from processing.preprocessor import FintechDatasetProcessor
from processing.trainers import ClassicalTrainer, TransferTrainer


def _print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def _print_raw_json_if_requested(args, payload: dict) -> None:
    if getattr(args, "json_output", False):
        _print_section("Raw JSON")
        print(json.dumps(payload, indent=2, ensure_ascii=True))


def _print_download_summary(stats: dict) -> None:
    _print_section("Download Summary")
    for key in ["split_train_rows", "split_test_rows", "saved_train", "saved_test", "saved_label_names"]:
        if key in stats:
            print(f"- {key}: {stats[key]}")


def _print_process_summary(stats: dict) -> None:
    _print_section("Dataset Summary")
    print(f"- total_samples: {stats.get('total_samples')}")
    print(f"- train/val/test: {stats.get('train_samples')}/{stats.get('val_samples')}/{stats.get('test_samples')}")
    print(f"- intents: {len(stats.get('intents', []))}")

    top_intents = list((stats.get("intent_distribution") or {}).items())[:5]
    if top_intents:
        print("- top_intents:")
        for intent, count in top_intents:
            print(f"  * {intent}: {count}")

    _print_section("Generated Files")
    for name, path in (stats.get("files") or {}).items():
        print(f"- {name}: {path}")


def _print_classical_summary(stats: dict) -> None:
    _print_section("Classical Training Summary")
    print(f"- model: {stats.get('model_type')}")
    print(f"- accuracy: {float(stats.get('accuracy', 0.0)):.4f}")
    print(f"- num_samples: {stats.get('num_samples')}")
    print(f"- num_intents: {stats.get('num_intents')}")

    _print_section("Saved Artifacts")
    print(f"- model_path: {stats.get('model_path')}")
    print(f"- responses_path: {stats.get('responses_path')}")


def _print_transfer_summary(model_name: str, metrics: dict, output_dir: str) -> None:
    _print_section(f"{model_name} Training Summary")
    for key in ["eval_loss", "eval_accuracy", "eval_f1_macro", "eval_runtime", "epoch"]:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, float):
                print(f"- {key}: {value:.4f}")
            else:
                print(f"- {key}: {value}")

    _print_section("Saved Artifacts")
    print(f"- output_dir: {output_dir}")


def cmd_download(args) -> None:
    stats = Banking77Downloader(output_dir=args.output_dir).download()
    _print_download_summary(stats)
    _print_raw_json_if_requested(args, stats)


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
    _print_process_summary(stats)
    _print_raw_json_if_requested(args, stats)


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
    _print_classical_summary(stats)
    _print_raw_json_if_requested(args, stats)


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
    _print_transfer_summary("BERT", metrics, args.output_dir)
    _print_raw_json_if_requested(args, metrics)


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
    _print_transfer_summary("GPT", metrics, args.output_dir)
    _print_raw_json_if_requested(args, metrics)


def cmd_all(args) -> None:
    download_stats = Banking77Downloader(output_dir=args.raw_output_dir).download()
    _print_download_summary(download_stats)

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
    _print_process_summary(process_stats)

    trainer = ClassicalTrainer()
    classical_stats = trainer.train(
        data=process_stats["files"]["train"],
        model=args.classical_model,
        output_dir=args.classical_output_dir,
    )
    _print_classical_summary(classical_stats)
    _print_raw_json_if_requested(args, {"download": download_stats, "process": process_stats, "classical": classical_stats})


def cmd_quickstart(args) -> None:
    if args.download_first:
        download_stats = Banking77Downloader(output_dir=args.raw_output_dir).download()
        _print_download_summary(download_stats)

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
    _print_process_summary(process_stats)

    trainer = TransferTrainer()

    if args.mode in {"best_classical", "classical"}:
        selected_model = "logreg" if args.mode == "best_classical" else args.classical_model
        classical_stats = ClassicalTrainer().train(
            data=process_stats["files"]["train"],
            model=selected_model,
            output_dir=args.classical_output_dir,
        )
        _print_classical_summary(classical_stats)
        _print_raw_json_if_requested(args, classical_stats)
        return

    if args.mode == "bert":
        bert_stats = trainer.train_bert_intent(
            data=process_stats["files"]["train"],
            model_name=args.bert_model_name,
            epochs=args.bert_epochs,
            batch_size=args.bert_batch_size,
            learning_rate=args.bert_learning_rate,
            max_length=args.bert_max_length,
            output_dir=args.bert_output_dir,
        )
        _print_transfer_summary("BERT", bert_stats, args.bert_output_dir)
        _print_raw_json_if_requested(args, bert_stats)
        return

    gpt_stats = trainer.train_gpt(
        data=process_stats["files"]["gpt"],
        model_name=args.gpt_model_name,
        epochs=args.gpt_epochs,
        batch_size=args.gpt_batch_size,
        learning_rate=args.gpt_learning_rate,
        max_length=args.gpt_max_length,
        output_dir=args.gpt_output_dir,
    )
    _print_transfer_summary("GPT", gpt_stats, args.gpt_output_dir)
    _print_raw_json_if_requested(args, gpt_stats)


def _prompt_choice() -> str:
    print("Choose chatbot mode:")
    print("1. Best Classical Model")
    print("2. GPT")
    print("3. BERT")
    print("4. Exit")
    return input("Enter choice (1/2/3/4): ").strip()


def _run_interactive_menu() -> None:
    from chatbot import BertIntentChatbot, ClassicalChatbot, GPTChatbot, run_chat

    choice = _prompt_choice()
    if choice == "4":
        print("Bye.")
        return

    mode_map = {
        "1": "best_classical",
        "2": "gpt",
        "3": "bert",
    }
    if choice not in mode_map:
        print("Invalid choice. Please run again and select 1, 2, 3, or 4.")
        return

    mode = mode_map[choice]
    run_train = input("Run training before chat? (y/n, default n): ").strip().lower() == "y"

    if run_train:
        quickstart_args = argparse.Namespace(
            mode=mode,
            download_first=False,
            raw_output_dir="dataset/raw/banking77",
            processed_output_dir="dataset/processed",
            regional_aug="dataset/regional_augmentation_template.csv",
            bitext_path="",
            kaggle_path="",
            max_banking77_rows=0,
            min_samples_per_intent=40,
            no_synthetic_fill=False,
            random_seed=42,
            json_output=False,
            classical_model="logreg",
            classical_output_dir="models/classical",
            bert_model_name="distilbert-base-uncased",
            bert_epochs=3,
            bert_batch_size=8,
            bert_learning_rate=2e-5,
            bert_max_length=128,
            bert_output_dir="models/bert_intent",
            gpt_model_name="distilgpt2",
            gpt_epochs=2,
            gpt_batch_size=4,
            gpt_learning_rate=5e-5,
            gpt_max_length=192,
            gpt_output_dir="models/gpt_finetuned",
        )
        cmd_quickstart(quickstart_args)

    if mode == "best_classical":
        model_path = "models/classical/logreg_pipeline.joblib"
        responses_path = "models/classical/responses_by_intent.json"
        if not os.path.exists(model_path) or not os.path.exists(responses_path):
            print("Model files not found. Run training first with quickstart mode best_classical.")
            return
        bot = ClassicalChatbot(model_path=model_path, responses_path=responses_path)
        run_chat(bot)
        return

    if mode == "bert":
        model_dir = "models/bert_intent"
        responses_path = "models/classical/responses_by_intent.json"
        if not os.path.exists(model_dir):
            print("BERT model not found. Run training first with quickstart mode bert.")
            return
        if not os.path.exists(responses_path):
            print("Responses file missing at models/classical/responses_by_intent.json.")
            print("Train best_classical once to generate intent responses, then try BERT chat again.")
            return
        bot = BertIntentChatbot(model_dir=model_dir, responses_path=responses_path)
        run_chat(bot)
        return

    model_dir = "models/gpt_finetuned"
    if not os.path.exists(model_dir):
        print("GPT model not found. Run training first with quickstart mode gpt.")
        return
    bot = GPTChatbot(model_dir=model_dir, max_new_tokens=80)
    run_chat(bot)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FinTech chatbot pipeline runner")
    subparsers = parser.add_subparsers(dest="command")

    download = subparsers.add_parser("download", help="Download BANKING77 to local raw folder")
    download.add_argument("--output_dir", type=str, default="dataset/raw/banking77")
    download.add_argument("--json_output", action="store_true")
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
    process.add_argument("--json_output", action="store_true")
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
    train_classical.add_argument("--json_output", action="store_true")
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
    train_intent.add_argument("--json_output", action="store_true")
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
    train_gpt.add_argument("--json_output", action="store_true")
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
    all_cmd.add_argument("--json_output", action="store_true")
    all_cmd.set_defaults(func=cmd_all)

    quickstart = subparsers.add_parser("quickstart", help="One-command setup + train with selected mode")
    quickstart.add_argument(
        "--mode",
        type=str,
        default="best_classical",
        choices=["best_classical", "classical", "bert", "gpt"],
    )
    quickstart.add_argument("--download_first", action="store_true")
    quickstart.add_argument("--raw_output_dir", type=str, default="dataset/raw/banking77")
    quickstart.add_argument("--processed_output_dir", type=str, default="dataset/processed")
    quickstart.add_argument("--regional_aug", type=str, default="dataset/regional_augmentation_template.csv")
    quickstart.add_argument("--bitext_path", type=str, default="")
    quickstart.add_argument("--kaggle_path", type=str, default="")
    quickstart.add_argument("--max_banking77_rows", type=int, default=0)
    quickstart.add_argument("--min_samples_per_intent", type=int, default=40)
    quickstart.add_argument("--no_synthetic_fill", action="store_true")
    quickstart.add_argument("--random_seed", type=int, default=42)
    quickstart.add_argument("--json_output", action="store_true")

    quickstart.add_argument("--classical_model", type=str, default="logreg", choices=["logreg", "naive_bayes", "svm"])
    quickstart.add_argument("--classical_output_dir", type=str, default="models/classical")

    quickstart.add_argument("--bert_model_name", type=str, default="distilbert-base-uncased")
    quickstart.add_argument("--bert_epochs", type=int, default=3)
    quickstart.add_argument("--bert_batch_size", type=int, default=8)
    quickstart.add_argument("--bert_learning_rate", type=float, default=2e-5)
    quickstart.add_argument("--bert_max_length", type=int, default=128)
    quickstart.add_argument("--bert_output_dir", type=str, default="models/bert_intent")

    quickstart.add_argument("--gpt_model_name", type=str, default="distilgpt2")
    quickstart.add_argument("--gpt_epochs", type=int, default=2)
    quickstart.add_argument("--gpt_batch_size", type=int, default=4)
    quickstart.add_argument("--gpt_learning_rate", type=float, default=5e-5)
    quickstart.add_argument("--gpt_max_length", type=int, default=192)
    quickstart.add_argument("--gpt_output_dir", type=str, default="models/gpt_finetuned")
    quickstart.set_defaults(func=cmd_quickstart)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not getattr(args, "command", None):
        _run_interactive_menu()
        return
    args.func(args)


if __name__ == "__main__":
    main()
