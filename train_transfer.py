import argparse
import json

from processing.trainers import TransferTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train transfer-learning models for fintech chatbot")
    subparsers = parser.add_subparsers(dest="task", required=True)

    bert_parser = subparsers.add_parser("intent", help="Fine-tune BERT-like encoder for intent classification")
    bert_parser.add_argument("--data", type=str, default="dataset/processed/fintech_intents_final_train.csv")
    bert_parser.add_argument("--text_col", type=str, default="text")
    bert_parser.add_argument("--label_col", type=str, default="intent")
    bert_parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    bert_parser.add_argument("--init_model_path", type=str, default="")
    bert_parser.add_argument("--resume_from_checkpoint", type=str, default="")
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
    gpt_parser.add_argument("--init_model_path", type=str, default="")
    gpt_parser.add_argument("--resume_from_checkpoint", type=str, default="")
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
        print("BERT intent classifier training completed.")
        print("BERT evaluation report:")
        print(metrics.get("classification_report", "N/A"))
        if metrics.get("best_model_checkpoint"):
            print(f"Best checkpoint: {metrics['best_model_checkpoint']}")
        print(f"Best model directory: {metrics.get('best_model_dir', args.output_dir)}")
        if metrics.get("run_dir"):
            print(f"Run directory: {metrics['run_dir']}")
        if metrics.get("evaluation_dir"):
            print(f"Evaluation directory: {metrics['evaluation_dir']}")
        if metrics.get("latest_best_model_dir"):
            print(f"Latest best model directory: {metrics['latest_best_model_dir']}")
        if metrics.get("run_archive_path"):
            print(f"Run archive (.zip): {metrics['run_archive_path']}")
    else:
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
        print("GPT-style model training completed.")
        if "eval_loss" in metrics:
            print(f"GPT eval_loss: {metrics['eval_loss']:.6f}")
        if "eval_perplexity" in metrics:
            print(f"GPT eval_perplexity: {metrics['eval_perplexity']:.4f}")
        if metrics.get("best_model_checkpoint"):
            print(f"Best checkpoint: {metrics['best_model_checkpoint']}")
        print(f"Best model directory: {metrics.get('best_model_dir', args.output_dir)}")
        if metrics.get("run_dir"):
            print(f"Run directory: {metrics['run_dir']}")
        if metrics.get("evaluation_dir"):
            print(f"Evaluation directory: {metrics['evaluation_dir']}")
        if metrics.get("latest_best_model_dir"):
            print(f"Latest best model directory: {metrics['latest_best_model_dir']}")
        if metrics.get("run_archive_path"):
            print(f"Run archive (.zip): {metrics['run_archive_path']}")

    print(json.dumps(metrics, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
