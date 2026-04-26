import json
import math
import os
import inspect
import shutil
from datetime import datetime
from typing import Dict, List

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


class ClassicalTrainer:
    @staticmethod
    def _is_colab() -> bool:
        return bool(os.environ.get("COLAB_RELEASE_TAG")) or os.path.isdir("/content")

    @staticmethod
    def _timestamp() -> str:
        return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    def _resolve_output_root(self, output_dir: str, model_family: str) -> str:
        normalized = output_dir.strip()
        if normalized:
            return normalized
        if self._is_colab():
            return f"/content/model_outputs/{model_family}"
        return f"models/{model_family}"

    def _prepare_run_dirs(self, output_dir: str, model_family: str) -> dict:
        output_root = self._resolve_output_root(output_dir=output_dir, model_family=model_family)
        run_dir = os.path.join(output_root, "runs", f"{model_family}_{self._timestamp()}")
        evaluation_dir = os.path.join(run_dir, "evaluation")
        model_dir = os.path.join(run_dir, "model")

        os.makedirs(output_root, exist_ok=True)
        os.makedirs(evaluation_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        return {
            "output_root": output_root,
            "run_dir": run_dir,
            "evaluation_dir": evaluation_dir,
            "model_dir": model_dir,
        }

    @staticmethod
    def _archive_run(run_dir: str, output_root: str) -> str:
        exports_dir = os.path.join(output_root, "exports")
        os.makedirs(exports_dir, exist_ok=True)
        archive_base = os.path.join(exports_dir, os.path.basename(run_dir))
        archive_path = shutil.make_archive(archive_base, "zip", root_dir=run_dir)
        return archive_path

    def build_model(self, model_name: str) -> Pipeline:
        if model_name == "logreg":
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        elif model_name == "naive_bayes":
            clf = MultinomialNB(alpha=0.5)
        elif model_name == "svm":
            clf = LinearSVC()
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=(1, 2),
                        min_df=1,
                        strip_accents="unicode",
                    ),
                ),
                ("clf", clf),
            ]
        )

    def build_responses_map(self, df: pd.DataFrame, label_col: str, response_col: str) -> Dict[str, List[str]]:
        response_map: Dict[str, List[str]] = {}
        if response_col not in df.columns:
            return response_map

        for label, group in df.groupby(label_col):
            values = [str(v).strip() for v in group[response_col].dropna().tolist() if str(v).strip()]
            if values:
                response_map[str(label)] = sorted(list(set(values)))
        return response_map

    def train(
        self,
        data: str,
        model: str = "logreg",
        text_col: str = "text",
        label_col: str = "intent",
        response_col: str = "response",
        test_size: float = 0.2,
        random_state: int = 42,
        output_dir: str = "models/classical",
    ) -> dict:
        paths = self._prepare_run_dirs(output_dir=output_dir, model_family="classical")
        output_root = paths["output_root"]
        run_dir = paths["run_dir"]
        evaluation_dir = paths["evaluation_dir"]
        run_model_dir = paths["model_dir"]

        df = pd.read_csv(data)
        if text_col not in df.columns or label_col not in df.columns:
            raise ValueError(f"Dataset must include '{text_col}' and '{label_col}'. Found: {list(df.columns)}")

        used_cols = [text_col, label_col] + ([response_col] if response_col in df.columns else [])
        df = df[used_cols].dropna(subset=[text_col, label_col])

        X = df[text_col].astype(str).tolist()
        y = df[label_col].astype(str).tolist()

        # Fall back to non-stratified split when class counts are too small.
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=y,
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=None,
            )

        models_to_run = [model]
        if model == "all":
            models_to_run = ["logreg", "naive_bayes", "svm"]

        responses_map = self.build_responses_map(df, label_col, response_col)
        responses_path = os.path.join(output_root, "responses_by_intent.json")
        responses_run_path = os.path.join(run_model_dir, "responses_by_intent.json")
        with open(responses_path, "w", encoding="utf-8") as f:
            json.dump(responses_map, f, indent=2, ensure_ascii=True)
        with open(responses_run_path, "w", encoding="utf-8") as f:
            json.dump(responses_map, f, indent=2, ensure_ascii=True)

        per_model = {}
        for model_name in models_to_run:
            pipeline = self.build_model(model_name)
            pipeline.fit(X_train, y_train)

            preds = pipeline.predict(X_test)
            acc = accuracy_score(y_test, preds)
            report = classification_report(y_test, preds, digits=4, zero_division=0)
            report_dict = classification_report(y_test, preds, digits=4, zero_division=0, output_dict=True)

            model_path = os.path.join(output_root, f"{model_name}_pipeline.joblib")
            model_run_path = os.path.join(run_model_dir, f"{model_name}_pipeline.joblib")
            joblib.dump(pipeline, model_path)
            joblib.dump(pipeline, model_run_path)

            per_model[model_name] = {
                "accuracy": float(acc),
                "classification_report": report,
                "model_path": model_path,
                "run_model_path": model_run_path,
            }

            with open(os.path.join(output_root, f"metadata_{model_name}.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "model_type": model_name,
                        "text_col": text_col,
                        "label_col": label_col,
                        "response_col": response_col,
                        "num_samples": len(df),
                        "num_intents": int(df[label_col].nunique()),
                        "intents": sorted(df[label_col].astype(str).unique().tolist()),
                        "accuracy": float(acc),
                        "model_path": model_path,
                        "run_model_path": model_run_path,
                        "responses_path": responses_path,
                        "responses_run_path": responses_run_path,
                        "classification_report": report,
                    },
                    f,
                    indent=2,
                    ensure_ascii=True,
                )

            with open(
                os.path.join(output_root, f"evaluation_{model_name}.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    {
                        "model": model_name,
                        "accuracy": float(acc),
                        "classification_report": report_dict,
                    },
                    f,
                    indent=2,
                    ensure_ascii=True,
                )
            with open(
                os.path.join(evaluation_dir, f"evaluation_{model_name}.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    {
                        "model": model_name,
                        "accuracy": float(acc),
                        "classification_report": report_dict,
                    },
                    f,
                    indent=2,
                    ensure_ascii=True,
                )

            pd.DataFrame(report_dict).T.to_csv(
                os.path.join(output_root, f"evaluation_{model_name}.csv"),
                index=True,
            )
            pd.DataFrame(report_dict).T.to_csv(
                os.path.join(evaluation_dir, f"evaluation_{model_name}.csv"),
                index=True,
            )

        best_model_name = max(per_model.items(), key=lambda item: item[1]["accuracy"])[0]
        summary = {
            "model_type": model,
            "text_col": text_col,
            "label_col": label_col,
            "response_col": response_col,
            "num_samples": len(df),
            "num_intents": int(df[label_col].nunique()),
            "intents": sorted(df[label_col].astype(str).unique().tolist()),
            "responses_path": responses_path,
            "responses_run_path": responses_run_path,
            "run_dir": run_dir,
            "evaluation_dir": evaluation_dir,
            "model_run_dir": run_model_dir,
            "models": per_model,
            "best_model": {
                "name": best_model_name,
                "accuracy": float(per_model[best_model_name]["accuracy"]),
                "model_path": per_model[best_model_name]["model_path"],
                "run_model_path": per_model[best_model_name]["run_model_path"],
            },
        }

        archive_path = self._archive_run(run_dir=run_dir, output_root=output_root)
        summary["run_archive_path"] = archive_path

        with open(os.path.join(output_root, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=True)
        with open(os.path.join(run_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=True)

        # Preserve old return shape for single-model callers.
        if model != "all":
            return {
                "model_type": model,
                "text_col": text_col,
                "label_col": label_col,
                "response_col": response_col,
                "num_samples": len(df),
                "num_intents": int(df[label_col].nunique()),
                "intents": sorted(df[label_col].astype(str).unique().tolist()),
                "accuracy": float(per_model[model]["accuracy"]),
                "model_path": per_model[model]["model_path"],
                "run_model_path": per_model[model]["run_model_path"],
                "responses_path": responses_path,
                "responses_run_path": responses_run_path,
                "run_dir": run_dir,
                "evaluation_dir": evaluation_dir,
                "model_run_dir": run_model_dir,
                "run_archive_path": archive_path,
                "classification_report": per_model[model]["classification_report"],
            }

        return summary


class TransferTrainer:
    @staticmethod
    def _is_colab() -> bool:
        return bool(os.environ.get("COLAB_RELEASE_TAG")) or os.path.isdir("/content")

    @staticmethod
    def _timestamp() -> str:
        return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    def _resolve_output_root(self, output_dir: str, model_family: str) -> str:
        normalized = output_dir.strip()
        if normalized:
            return normalized
        if self._is_colab():
            return f"/content/model_outputs/{model_family}"
        return f"models/{model_family}"

    def _prepare_run_dirs(self, output_dir: str, model_family: str) -> dict:
        output_root = self._resolve_output_root(output_dir=output_dir, model_family=model_family)
        run_dir = os.path.join(output_root, "runs", f"{model_family}_{self._timestamp()}")
        evaluation_dir = os.path.join(run_dir, "evaluation")
        model_dir = os.path.join(run_dir, "model")
        checkpoints_dir = os.path.join(run_dir, "checkpoints")

        os.makedirs(output_root, exist_ok=True)
        os.makedirs(evaluation_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)

        return {
            "output_root": output_root,
            "run_dir": run_dir,
            "evaluation_dir": evaluation_dir,
            "model_dir": model_dir,
            "checkpoints_dir": checkpoints_dir,
        }

    @staticmethod
    def _archive_run(run_dir: str, output_root: str) -> str:
        exports_dir = os.path.join(output_root, "exports")
        os.makedirs(exports_dir, exist_ok=True)
        archive_base = os.path.join(exports_dir, os.path.basename(run_dir))
        archive_path = shutil.make_archive(archive_base, "zip", root_dir=run_dir)
        return archive_path

    @staticmethod
    def _sync_directory(src: str, dst: str) -> None:
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    @staticmethod
    def _build_training_arguments_kwargs(**kwargs) -> dict:
        """Support both evaluation_strategy and eval_strategy across transformers versions."""
        from transformers import TrainingArguments

        parameters = inspect.signature(TrainingArguments.__init__).parameters
        if "evaluation_strategy" in parameters:
            kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
        return kwargs

    def train_bert_intent(
        self,
        data: str,
        text_col: str = "text",
        label_col: str = "intent",
        model_name: str = "distilbert-base-uncased",
        init_model_path: str = "",
        resume_from_checkpoint: str = "",
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        max_length: int = 128,
        test_size: float = 0.2,
        random_state: int = 42,
        output_dir: str = "models/bert_intent",
    ) -> dict:
        import evaluate
        import numpy as np
        import torch
        from datasets import Dataset
        from sklearn.preprocessing import LabelEncoder
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

        paths = self._prepare_run_dirs(output_dir=output_dir, model_family="bert_intent")
        output_root = paths["output_root"]
        run_dir = paths["run_dir"]
        evaluation_dir = paths["evaluation_dir"]
        run_model_dir = paths["model_dir"]
        run_checkpoints_dir = paths["checkpoints_dir"]

        df = pd.read_csv(data)
        if text_col not in df.columns or label_col not in df.columns:
            raise ValueError(f"Dataset must include '{text_col}' and '{label_col}'. Found: {list(df.columns)}")

        df = df[[text_col, label_col]].dropna()
        df[text_col] = df[text_col].astype(str)
        df[label_col] = df[label_col].astype(str)

        try:
            train_df, eval_df = train_test_split(
                df,
                test_size=test_size,
                random_state=random_state,
                stratify=df[label_col],
            )
        except ValueError:
            train_df, eval_df = train_test_split(
                df,
                test_size=test_size,
                random_state=random_state,
                stratify=None,
            )

        label_encoder = LabelEncoder()
        train_df = train_df.copy()
        eval_df = eval_df.copy()
        train_df["label"] = label_encoder.fit_transform(train_df[label_col])
        eval_df["label"] = label_encoder.transform(eval_df[label_col])

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        train_ds = Dataset.from_pandas(train_df[[text_col, "label"]], preserve_index=False)
        eval_ds = Dataset.from_pandas(eval_df[[text_col, "label"]], preserve_index=False)

        def tokenize_fn(batch):
            return tokenizer(batch[text_col], truncation=True, padding="max_length", max_length=max_length)

        train_ds = train_ds.map(tokenize_fn, batched=True)
        eval_ds = eval_ds.map(tokenize_fn, batched=True)

        train_ds = train_ds.rename_column("label", "labels")
        eval_ds = eval_ds.rename_column("label", "labels")
        train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        eval_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        model_source = init_model_path.strip() if init_model_path.strip() else model_name
        model = AutoModelForSequenceClassification.from_pretrained(
            model_source,
            num_labels=len(label_encoder.classes_),
            ignore_mismatched_sizes=True,
        )

        accuracy_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            accuracy = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
            f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
            return {"accuracy": accuracy, "f1_macro": f1}

        training_args = TrainingArguments(
            **self._build_training_arguments_kwargs(
                output_dir=run_checkpoints_dir,
                eval_strategy="epoch",
                save_strategy="epoch",
                learning_rate=learning_rate,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=epochs,
                weight_decay=0.01,
                logging_steps=10,
                load_best_model_at_end=True,
                metric_for_best_model="f1_macro",
                greater_is_better=True,
                report_to="none",
                fp16=torch.cuda.is_available(),
            )
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            compute_metrics=compute_metrics,
        )

        trainer.train(resume_from_checkpoint=resume_from_checkpoint.strip() or None)
        metrics = trainer.evaluate()

        pred_output = trainer.predict(eval_ds)
        preds = np.argmax(pred_output.predictions, axis=-1)
        labels = pred_output.label_ids
        class_report = classification_report(
            labels,
            preds,
            target_names=label_encoder.classes_.tolist(),
            digits=4,
            zero_division=0,
        )
        class_report_dict = classification_report(
            labels,
            preds,
            target_names=label_encoder.classes_.tolist(),
            digits=4,
            zero_division=0,
            output_dict=True,
        )

        trainer.save_model(run_model_dir)
        tokenizer.save_pretrained(run_model_dir)

        best_model_dir = os.path.join(run_model_dir, "best_model")
        os.makedirs(best_model_dir, exist_ok=True)
        trainer.save_model(best_model_dir)
        tokenizer.save_pretrained(best_model_dir)

        latest_model_dir = os.path.join(output_root, "latest_model")
        latest_best_model_dir = os.path.join(output_root, "best_model")
        self._sync_directory(run_model_dir, latest_model_dir)
        self._sync_directory(best_model_dir, latest_best_model_dir)

        clean_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                clean_metrics[key] = float(value)
            else:
                clean_metrics[key] = value

        with open(os.path.join(run_model_dir, "label_classes.json"), "w", encoding="utf-8") as f:
            json.dump(label_encoder.classes_.tolist(), f, indent=2, ensure_ascii=True)
        with open(os.path.join(output_root, "label_classes.json"), "w", encoding="utf-8") as f:
            json.dump(label_encoder.classes_.tolist(), f, indent=2, ensure_ascii=True)

        with open(os.path.join(best_model_dir, "label_classes.json"), "w", encoding="utf-8") as f:
            json.dump(label_encoder.classes_.tolist(), f, indent=2, ensure_ascii=True)

        result = {
            **clean_metrics,
            "classification_report": class_report,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "best_model_dir": best_model_dir,
            "latest_model_dir": latest_model_dir,
            "latest_best_model_dir": latest_best_model_dir,
            "run_dir": run_dir,
            "evaluation_dir": evaluation_dir,
            "init_model_source": model_source,
            "resume_from_checkpoint": resume_from_checkpoint.strip() or None,
        }

        result["run_archive_path"] = self._archive_run(run_dir=run_dir, output_root=output_root)

        with open(os.path.join(output_root, "training_summary.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=True)
        with open(os.path.join(run_dir, "training_summary.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=True)

        with open(os.path.join(evaluation_dir, "training_log.json"), "w", encoding="utf-8") as f:
            json.dump(trainer.state.log_history, f, indent=2, ensure_ascii=True)

        with open(os.path.join(output_root, "evaluation_bert.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metrics": clean_metrics,
                    "classification_report": class_report_dict,
                },
                f,
                indent=2,
                ensure_ascii=True,
            )
        with open(os.path.join(evaluation_dir, "evaluation_bert.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metrics": clean_metrics,
                    "classification_report": class_report_dict,
                },
                f,
                indent=2,
                ensure_ascii=True,
            )

        pd.DataFrame(class_report_dict).T.to_csv(
            os.path.join(output_root, "evaluation_bert.csv"),
            index=True,
        )
        pd.DataFrame(class_report_dict).T.to_csv(
            os.path.join(evaluation_dir, "evaluation_bert.csv"),
            index=True,
        )

        return result

    def train_gpt(
        self,
        data: str,
        prompt_col: str = "prompt",
        response_col: str = "response",
        model_name: str = "distilgpt2",
        init_model_path: str = "",
        resume_from_checkpoint: str = "",
        epochs: int = 2,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        max_length: int = 192,
        test_size: float = 0.1,
        random_state: int = 42,
        output_dir: str = "models/gpt_finetuned",
    ) -> dict:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )

        paths = self._prepare_run_dirs(output_dir=output_dir, model_family="gpt_finetuned")
        output_root = paths["output_root"]
        run_dir = paths["run_dir"]
        evaluation_dir = paths["evaluation_dir"]
        run_model_dir = paths["model_dir"]
        run_checkpoints_dir = paths["checkpoints_dir"]

        df = pd.read_json(data, lines=True)
        for col in [prompt_col, response_col]:
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}'. Found: {list(df.columns)}")

        df = df[[prompt_col, response_col]].dropna().copy()
        df[prompt_col] = df[prompt_col].astype(str)
        df[response_col] = df[response_col].astype(str)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_source = init_model_path.strip() if init_model_path.strip() else model_name
        model = AutoModelForCausalLM.from_pretrained(model_source)

        df["text"] = "User: " + df[prompt_col] + "\nAssistant: " + df[response_col] + tokenizer.eos_token

        train_df, eval_df = train_test_split(df[["text"]], test_size=test_size, random_state=random_state)
        train_ds = Dataset.from_pandas(train_df, preserve_index=False)
        eval_ds = Dataset.from_pandas(eval_df, preserve_index=False)

        def tokenize_fn(batch):
            return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)

        train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
        eval_ds = eval_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        train_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
        eval_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

        training_args = TrainingArguments(
            **self._build_training_arguments_kwargs(
                output_dir=run_checkpoints_dir,
                eval_strategy="epoch",
                save_strategy="epoch",
                learning_rate=learning_rate,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=epochs,
                weight_decay=0.01,
                logging_steps=10,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                report_to="none",
                fp16=torch.cuda.is_available(),
            )
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=data_collator,
        )

        trainer.train(resume_from_checkpoint=resume_from_checkpoint.strip() or None)
        metrics = trainer.evaluate()

        trainer.save_model(run_model_dir)
        tokenizer.save_pretrained(run_model_dir)

        best_model_dir = os.path.join(run_model_dir, "best_model")
        os.makedirs(best_model_dir, exist_ok=True)
        trainer.save_model(best_model_dir)
        tokenizer.save_pretrained(best_model_dir)

        latest_model_dir = os.path.join(output_root, "latest_model")
        latest_best_model_dir = os.path.join(output_root, "best_model")
        self._sync_directory(run_model_dir, latest_model_dir)
        self._sync_directory(best_model_dir, latest_best_model_dir)

        clean_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                clean_metrics[key] = float(value)
            else:
                clean_metrics[key] = value

        eval_loss = clean_metrics.get("eval_loss")
        if isinstance(eval_loss, float):
            clean_metrics["eval_perplexity"] = float(math.exp(eval_loss)) if eval_loss < 50 else float("inf")

        result = {
            **clean_metrics,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "best_model_dir": best_model_dir,
            "latest_model_dir": latest_model_dir,
            "latest_best_model_dir": latest_best_model_dir,
            "run_dir": run_dir,
            "evaluation_dir": evaluation_dir,
            "init_model_source": model_source,
            "resume_from_checkpoint": resume_from_checkpoint.strip() or None,
        }

        result["run_archive_path"] = self._archive_run(run_dir=run_dir, output_root=output_root)

        with open(os.path.join(output_root, "training_summary.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=True)
        with open(os.path.join(run_dir, "training_summary.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=True)

        with open(os.path.join(evaluation_dir, "training_log.json"), "w", encoding="utf-8") as f:
            json.dump(trainer.state.log_history, f, indent=2, ensure_ascii=True)

        with open(os.path.join(output_root, "evaluation_gpt.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=True)
        with open(os.path.join(evaluation_dir, "evaluation_gpt.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=True)

        pd.DataFrame(
            [{"metric": k, "value": v} for k, v in clean_metrics.items() if isinstance(v, (int, float))]
        ).to_csv(os.path.join(output_root, "evaluation_gpt.csv"), index=False)
        pd.DataFrame(
            [{"metric": k, "value": v} for k, v in clean_metrics.items() if isinstance(v, (int, float))]
        ).to_csv(os.path.join(evaluation_dir, "evaluation_gpt.csv"), index=False)

        return result
