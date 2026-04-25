import argparse
import json
import os
from typing import List

from processing.downloader import Banking77Downloader
from processing.preprocessor import FintechDatasetProcessor
from processing.trainers import ClassicalTrainer, TransferTrainer


def _prompt_choice(title: str, options):
    while True:
        print(f"\n{title}")
        for i, option in enumerate(options, start=1):
            print(f"  {i}. {option}")
        raw = input("Select an option number: ").strip()
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(options):
                return idx
        print("Invalid selection. Please enter a valid number.")


def _prompt_with_default(label: str, default: str) -> str:
    raw = input(f"{label} [{default}]: ").strip()
    return raw if raw else default


def _prompt_int_with_default(label: str, default: int) -> int:
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if raw == "":
            return default
        if raw.isdigit():
            return int(raw)
        print("Invalid number. Please enter a valid integer.")


def _find_classical_defaults() -> tuple[str, str]:
    candidate_dirs = [
        "models/classical",
        "models/classical_main_check",
        "models/classical_main_check_balanced",
        "models/classical_nb_check",
    ]

    for folder in candidate_dirs:
        metadata_path = os.path.join(folder, "metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "best_model" in data and "responses_path" in data:
                    return data["best_model"]["model_path"], data["responses_path"]
                if "model_path" in data and "responses_path" in data:
                    return data["model_path"], data["responses_path"]
            except Exception:
                continue

        nb_model = os.path.join(folder, "naive_bayes_pipeline.joblib")
        responses = os.path.join(folder, "responses_by_intent.json")
        if os.path.exists(nb_model) and os.path.exists(responses):
            return nb_model, responses

    return "models/classical/naive_bayes_pipeline.joblib", "models/classical/responses_by_intent.json"


def _has_any_model_weights(folder: str) -> bool:
    return (
        os.path.exists(os.path.join(folder, "model.safetensors"))
        or os.path.exists(os.path.join(folder, "pytorch_model.bin"))
    )


def _discover_model_dirs(model_kind: str) -> List[str]:
    candidates: List[str] = []

    if model_kind == "bert":
        preferred = [
            "models/bert_intent/best_model",
            "models/bert_intent",
        ]
    else:
        preferred = [
            "models/gpt_finetuned/best_model",
            "models/gpt_finetuned",
        ]

    for folder in preferred:
        if os.path.isdir(folder):
            candidates.append(folder)

    for root, dirs, _ in os.walk("models"):
        for d in dirs:
            folder = os.path.join(root, d)
            if model_kind == "bert":
                is_match = (
                    os.path.exists(os.path.join(folder, "label_classes.json"))
                    and os.path.exists(os.path.join(folder, "config.json"))
                    and _has_any_model_weights(folder)
                )
            else:
                is_match = (
                    not os.path.exists(os.path.join(folder, "label_classes.json"))
                    and os.path.exists(os.path.join(folder, "config.json"))
                    and os.path.exists(os.path.join(folder, "tokenizer_config.json"))
                    and _has_any_model_weights(folder)
                )
            if is_match:
                candidates.append(folder)

    # Deduplicate while preserving order.
    seen = set()
    ordered_unique = []
    for path in candidates:
        normalized = os.path.normpath(path)
        if normalized not in seen:
            seen.add(normalized)
            ordered_unique.append(normalized)

    # Prefer the most recently modified candidate.
    ordered_unique.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return ordered_unique


def _find_transfer_default(model_kind: str) -> str:
    discovered = _discover_model_dirs(model_kind=model_kind)
    if discovered:
        return discovered[0]
    if model_kind == "bert":
        return "models/bert_intent/best_model"
    return "models/gpt_finetuned/best_model"


def _run_chat_menu() -> None:
    from chatbot import BertIntentChatbot, ClassicalChatbot, GPTChatbot, run_chat

    backend_idx = _prompt_choice("Choose chatbot backend:", ["Classical", "BERT", "GPT"])

    if backend_idx == 1:
        default_model_path, default_responses_path = _find_classical_defaults()
        model_path = _prompt_with_default("Classical model path", default_model_path)
        responses_path = _prompt_with_default("Responses JSON path", default_responses_path)
        bot = ClassicalChatbot(model_path=model_path, responses_path=responses_path)
        run_chat(bot)
        return

    if backend_idx == 2:
        default_model_dir = _find_transfer_default(model_kind="bert")
        _, default_responses_path = _find_classical_defaults()
        model_dir = _prompt_with_default("BERT model directory", default_model_dir)
        responses_path = _prompt_with_default("Responses JSON path", default_responses_path)
        bot = BertIntentChatbot(model_dir=model_dir, responses_path=responses_path)
        run_chat(bot)
        return

    default_model_dir = _find_transfer_default(model_kind="gpt")
    model_dir = _prompt_with_default("GPT model directory", default_model_dir)
    max_new_tokens = _prompt_int_with_default("Max new tokens", 80)
    bot = GPTChatbot(model_dir=model_dir, max_new_tokens=max_new_tokens)
    run_chat(bot)


def _train_interactive() -> None:
    mode_idx = _prompt_choice("Choose model to train:", ["Classical", "BERT Intent", "GPT"])

    if mode_idx == 1:
        model_idx = _prompt_choice(
            "Choose classical model:",
            ["All (logreg + naive_bayes + svm)", "logreg", "naive_bayes", "svm"],
        )
        model_map = {1: "all", 2: "logreg", 3: "naive_bayes", 4: "svm"}

        args = argparse.Namespace(
            data=_prompt_with_default("Dataset CSV path", "dataset/processed/fintech_intents_train.csv"),
            model=model_map[model_idx],
            text_col="text",
            label_col="intent",
            response_col="response",
            test_size=0.2,
            random_state=42,
            output_dir=_prompt_with_default("Output directory", "models/classical"),
        )
        cmd_train_classical(args)
    elif mode_idx == 2:
        strategy_idx = _prompt_choice(
            "BERT training mode:",
            ["Train from base model", "Load best weights and continue"],
        )
        output_dir = _prompt_with_default("Output directory", "models/bert_intent")
        init_model_path = ""
        if strategy_idx == 2:
            init_model_path = _prompt_with_default("Best model path", _find_transfer_default(model_kind="bert"))

        args = argparse.Namespace(
            data=_prompt_with_default("Dataset CSV path", "dataset/processed/fintech_intents_train.csv"),
            text_col="text",
            label_col="intent",
            model_name=_prompt_with_default("Base model name", "distilbert-base-uncased"),
            init_model_path=init_model_path,
            resume_from_checkpoint="",
            epochs=_prompt_int_with_default("Epochs", 3),
            batch_size=_prompt_int_with_default("Batch size", 8),
            learning_rate=2e-5,
            max_length=128,
            test_size=0.2,
            random_state=42,
            output_dir=output_dir,
        )
        cmd_train_intent(args)
    else:
        strategy_idx = _prompt_choice(
            "GPT training mode:",
            ["Train from base model", "Load best weights and continue"],
        )
        output_dir = _prompt_with_default("Output directory", "models/gpt_finetuned")
        init_model_path = ""
        if strategy_idx == 2:
            init_model_path = _prompt_with_default("Best model path", _find_transfer_default(model_kind="gpt"))

        args = argparse.Namespace(
            data=_prompt_with_default("Dataset JSONL path", "dataset/processed/fintech_gpt_train.jsonl"),
            prompt_col="prompt",
            response_col="response",
            model_name=_prompt_with_default("Base model name", "distilgpt2"),
            init_model_path=init_model_path,
            resume_from_checkpoint="",
            epochs=_prompt_int_with_default("Epochs", 2),
            batch_size=_prompt_int_with_default("Batch size", 4),
            learning_rate=5e-5,
            max_length=192,
            test_size=0.1,
            random_state=42,
            output_dir=output_dir,
        )
        cmd_train_gpt(args)

    open_chat = _prompt_choice("Start chat now?", ["Yes", "No"])
    if open_chat == 1:
        _run_chat_menu()


def _interactive_run() -> None:
    print("FinTech Chatbot - simple mode")
    while True:
        choice = _prompt_choice(
            "Choose action:",
            ["Train model", "Load model and chat", "Prepare dataset", "Exit"],
        )

        if choice == 1:
            _train_interactive()
        elif choice == 2:
            _run_chat_menu()
        elif choice == 3:
            args = argparse.Namespace(
                output_dir=_prompt_with_default("Processed output folder", "dataset/processed"),
                regional_aug=_prompt_with_default(
                    "Regional augmentation CSV",
                    "dataset/cambodian_asia_banking_qa_2016.csv",
                ),
                asia_custom_aug=_prompt_with_default(
                    "Custom augmentation CSV",
                    "dataset/cambodian_asia_banking_custom.csv",
                ),
                bitext_path="",
                kaggle_path="",
                max_banking77_rows=_prompt_int_with_default("Max BANKING77 rows (0 = all)", 0),
                banking77_raw_dir=_prompt_with_default("BANKING77 raw folder", "dataset/raw/banking77"),
                min_samples_per_intent=_prompt_int_with_default("Min samples per intent", 40),
                no_synthetic_fill=False,
                cambodia_only=False,
                random_seed=42,
            )
            cmd_process(args)
        else:
            print("Bye.")
            break


def cmd_download(args) -> None:
    stats = Banking77Downloader(output_dir=args.output_dir).download()
    print(json.dumps(stats, indent=2, ensure_ascii=True))


def cmd_process(args) -> None:
    processor = FintechDatasetProcessor(random_seed=args.random_seed)
    stats = processor.process(
        output_dir=args.output_dir,
        regional_aug=args.regional_aug,
        asia_custom_aug=args.asia_custom_aug,
        bitext_path=args.bitext_path,
        kaggle_path=args.kaggle_path,
        max_banking77_rows=args.max_banking77_rows,
        banking77_raw_dir=args.banking77_raw_dir,
        min_samples_per_intent=args.min_samples_per_intent,
        no_synthetic_fill=args.no_synthetic_fill,
        cambodia_only=args.cambodia_only,
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
    if args.model == "all":
        print("Classical model evaluation summary:")
        for model_name, model_stats in stats["models"].items():
            print(f"\n[{model_name}] accuracy={model_stats['accuracy']:.4f}")
            print(model_stats["classification_report"])
        print(
            "Best model: "
            f"{stats['best_model']['name']} (accuracy={stats['best_model']['accuracy']:.4f})"
        )
    print(json.dumps(stats, indent=2, ensure_ascii=True))
    if args.model == "all":
        print(f"Saved evaluation files under {args.output_dir}: evaluation_logreg.* evaluation_naive_bayes.* evaluation_svm.*")
    else:
        print(f"Saved evaluation files under {args.output_dir}: evaluation_{args.model}.json and evaluation_{args.model}.csv")
    return stats


def cmd_train_intent(args) -> None:
    trainer = TransferTrainer()
    metrics = trainer.train_bert_intent(
        data=args.data,
        text_col=args.text_col,
        label_col=args.label_col,
        model_name=args.model_name,
        init_model_path=args.init_model_path,
        resume_from_checkpoint=args.resume_from_checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        test_size=args.test_size,
        random_state=args.random_state,
        output_dir=args.output_dir,
    )
    print("BERT evaluation report:")
    print(metrics.get("classification_report", "N/A"))
    if metrics.get("best_model_checkpoint"):
        print(f"Best checkpoint: {metrics['best_model_checkpoint']}")
    print(f"Best model directory: {metrics.get('best_model_dir', args.output_dir)}")
    print(json.dumps(metrics, indent=2, ensure_ascii=True))
    print(f"Saved evaluation files: {os.path.join(args.output_dir, 'evaluation_bert.json')} and evaluation_bert.csv")
    return metrics


def cmd_train_gpt(args) -> None:
    trainer = TransferTrainer()
    metrics = trainer.train_gpt(
        data=args.data,
        prompt_col=args.prompt_col,
        response_col=args.response_col,
        model_name=args.model_name,
        init_model_path=args.init_model_path,
        resume_from_checkpoint=args.resume_from_checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        test_size=args.test_size,
        random_state=args.random_state,
        output_dir=args.output_dir,
    )
    if "eval_loss" in metrics:
        print(f"GPT eval_loss: {metrics['eval_loss']:.6f}")
    if "eval_perplexity" in metrics:
        print(f"GPT eval_perplexity: {metrics['eval_perplexity']:.4f}")
    if metrics.get("best_model_checkpoint"):
        print(f"Best checkpoint: {metrics['best_model_checkpoint']}")
    print(f"Best model directory: {metrics.get('best_model_dir', args.output_dir)}")
    print(json.dumps(metrics, indent=2, ensure_ascii=True))
    print(f"Saved evaluation files: {os.path.join(args.output_dir, 'evaluation_gpt.json')} and evaluation_gpt.csv")
    return metrics


def cmd_all(args) -> None:
    download_stats = Banking77Downloader(output_dir=args.raw_output_dir).download()
    print("Download complete:")
    print(json.dumps(download_stats, indent=2, ensure_ascii=True))

    processor = FintechDatasetProcessor(random_seed=args.random_seed)
    process_stats = processor.process(
        output_dir=args.processed_output_dir,
        regional_aug=args.regional_aug,
        asia_custom_aug=args.asia_custom_aug,
        bitext_path=args.bitext_path,
        kaggle_path=args.kaggle_path,
        banking77_raw_dir=args.raw_output_dir,
        max_banking77_rows=args.max_banking77_rows,
        min_samples_per_intent=args.min_samples_per_intent,
        no_synthetic_fill=args.no_synthetic_fill,
        cambodia_only=args.cambodia_only,
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
    if args.classical_model == "all":
        for model_name, model_stats in classical_stats["models"].items():
            print(f"\n[{model_name}] accuracy={model_stats['accuracy']:.4f}")
            print(model_stats["classification_report"])
        print(
            "Best classical model: "
            f"{classical_stats['best_model']['name']} "
            f"(accuracy={classical_stats['best_model']['accuracy']:.4f})"
        )
    else:
        print(classical_stats["classification_report"])
    print(json.dumps(classical_stats, indent=2, ensure_ascii=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FinTech chatbot pipeline runner")
    subparsers = parser.add_subparsers(dest="command", required=False)

    download = subparsers.add_parser("download", help="Download BANKING77 to local raw folder")
    download.add_argument("--output_dir", type=str, default="dataset/raw/banking77")
    download.set_defaults(func=cmd_download)

    process = subparsers.add_parser("process", help="Preprocess datasets into template-ready outputs")
    process.add_argument("--output_dir", type=str, default="dataset/processed")
    process.add_argument("--regional_aug", type=str, default="dataset/cambodian_asia_banking_qa_2016.csv")
    process.add_argument("--asia_custom_aug", type=str, default="dataset/cambodian_asia_banking_custom.csv")
    process.add_argument("--bitext_path", type=str, default="")
    process.add_argument("--kaggle_path", type=str, default="")
    process.add_argument("--banking77_raw_dir", type=str, default="dataset/raw/banking77")
    process.add_argument("--max_banking77_rows", type=int, default=0)
    process.add_argument("--min_samples_per_intent", type=int, default=40)
    process.add_argument("--no_synthetic_fill", action="store_true")
    process.add_argument("--cambodia_only", action="store_true")
    process.add_argument("--random_seed", type=int, default=42)
    process.set_defaults(func=cmd_process)

    train_classical = subparsers.add_parser("train-classical", help="Train Naive Bayes/LogReg/SVM")
    train_classical.add_argument("--data", type=str, default="dataset/processed/fintech_intents_train.csv")
    train_classical.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["logreg", "naive_bayes", "svm", "all"],
    )
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
    train_intent.add_argument("--init_model_path", type=str, default="")
    train_intent.add_argument("--resume_from_checkpoint", type=str, default="")
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
    train_gpt.add_argument("--init_model_path", type=str, default="")
    train_gpt.add_argument("--resume_from_checkpoint", type=str, default="")
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
    all_cmd.add_argument("--regional_aug", type=str, default="dataset/cambodian_asia_banking_qa_2016.csv")
    all_cmd.add_argument("--asia_custom_aug", type=str, default="dataset/cambodian_asia_banking_custom.csv")
    all_cmd.add_argument("--bitext_path", type=str, default="")
    all_cmd.add_argument("--kaggle_path", type=str, default="")
    all_cmd.add_argument("--max_banking77_rows", type=int, default=0)
    all_cmd.add_argument("--min_samples_per_intent", type=int, default=40)
    all_cmd.add_argument("--no_synthetic_fill", action="store_true")
    all_cmd.add_argument("--cambodia_only", action="store_true")
    all_cmd.add_argument(
        "--classical_model",
        type=str,
        default="all",
        choices=["logreg", "naive_bayes", "svm", "all"],
    )
    all_cmd.add_argument("--classical_output_dir", type=str, default="models/classical")
    all_cmd.add_argument("--random_seed", type=int, default=42)
    all_cmd.set_defaults(func=cmd_all)

    chat_cmd = subparsers.add_parser("chat", help="Start chatbot with selected backend")
    chat_cmd.add_argument("--backend", choices=["classical", "bert", "gpt"], required=True)
    chat_cmd.add_argument("--model_path", type=str, default="models/classical/naive_bayes_pipeline.joblib")
    chat_cmd.add_argument("--model_dir", type=str, default="models/bert_intent/best_model")
    chat_cmd.add_argument("--responses_path", type=str, default="models/classical/responses_by_intent.json")
    chat_cmd.add_argument("--max_new_tokens", type=int, default=80)
    chat_cmd.set_defaults(func=cmd_chat)

    return parser


def cmd_chat(args) -> None:
    from chatbot import BertIntentChatbot, ClassicalChatbot, GPTChatbot, run_chat

    if args.backend == "classical":
        bot = ClassicalChatbot(model_path=args.model_path, responses_path=args.responses_path)
    elif args.backend == "bert":
        bot = BertIntentChatbot(model_dir=args.model_dir, responses_path=args.responses_path)
    else:
        bot = GPTChatbot(model_dir=args.model_dir, max_new_tokens=args.max_new_tokens)
    run_chat(bot)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        _interactive_run()
        return
    args.func(args)


if __name__ == "__main__":
    main()
