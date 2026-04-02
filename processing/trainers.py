import json
import os
import inspect
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
        os.makedirs(output_dir, exist_ok=True)

        df = pd.read_csv(data)
        if text_col not in df.columns or label_col not in df.columns:
            raise ValueError(f"Dataset must include '{text_col}' and '{label_col}'. Found: {list(df.columns)}")

        used_cols = [text_col, label_col] + ([response_col] if response_col in df.columns else [])
        df = df[used_cols].dropna(subset=[text_col, label_col])

        X = df[text_col].astype(str).tolist()
        y = df[label_col].astype(str).tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        pipeline = self.build_model(model)
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, digits=4)

        model_path = os.path.join(output_dir, f"{model}_pipeline.joblib")
        joblib.dump(pipeline, model_path)

        responses_map = self.build_responses_map(df, label_col, response_col)
        responses_path = os.path.join(output_dir, "responses_by_intent.json")
        with open(responses_path, "w", encoding="utf-8") as f:
            json.dump(responses_map, f, indent=2, ensure_ascii=True)

        metadata = {
            "model_type": model,
            "text_col": text_col,
            "label_col": label_col,
            "response_col": response_col,
            "num_samples": len(df),
            "num_intents": int(df[label_col].nunique()),
            "intents": sorted(df[label_col].astype(str).unique().tolist()),
            "accuracy": float(acc),
            "model_path": model_path,
            "responses_path": responses_path,
            "classification_report": report,
        }

        with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=True)

        return metadata


class TransferTrainer:
    @staticmethod
    def _make_training_args_kwargs(TrainingArguments, base_kwargs: dict) -> dict:
        params = inspect.signature(TrainingArguments.__init__).parameters
        kwargs = dict(base_kwargs)

        if "eval_strategy" in params:
            kwargs["eval_strategy"] = "epoch"
        else:
            kwargs["evaluation_strategy"] = "epoch"

        if "save_strategy" in params:
            kwargs["save_strategy"] = "epoch"

        return kwargs

    @staticmethod
    def _build_trainer(Trainer, model, training_args, train_dataset, eval_dataset, tokenizer, compute_metrics=None, data_collator=None):
        params = inspect.signature(Trainer.__init__).parameters
        kwargs = {
            "model": model,
            "args": training_args,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
        }

        if "tokenizer" in params:
            kwargs["tokenizer"] = tokenizer
        if compute_metrics is not None and "compute_metrics" in params:
            kwargs["compute_metrics"] = compute_metrics
        if data_collator is not None and "data_collator" in params:
            kwargs["data_collator"] = data_collator

        return Trainer(**kwargs)

    def train_bert_intent(
        self,
        data: str,
        text_col: str = "text",
        label_col: str = "intent",
        model_name: str = "distilbert-base-uncased",
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

        os.makedirs(output_dir, exist_ok=True)

        df = pd.read_csv(data)
        if text_col not in df.columns or label_col not in df.columns:
            raise ValueError(f"Dataset must include '{text_col}' and '{label_col}'. Found: {list(df.columns)}")

        df = df[[text_col, label_col]].dropna()
        df[text_col] = df[text_col].astype(str)
        df[label_col] = df[label_col].astype(str)

        train_df, eval_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df[label_col],
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

        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))

        accuracy_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            accuracy = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
            f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
            return {"accuracy": accuracy, "f1_macro": f1}

        training_kwargs = self._make_training_args_kwargs(
            TrainingArguments,
            {
                "output_dir": output_dir,
                "learning_rate": learning_rate,
                "per_device_train_batch_size": batch_size,
                "per_device_eval_batch_size": batch_size,
                "num_train_epochs": epochs,
                "weight_decay": 0.01,
                "logging_steps": 10,
                "load_best_model_at_end": True,
                "metric_for_best_model": "f1_macro",
                "greater_is_better": True,
                "report_to": "none",
                "fp16": torch.cuda.is_available(),
            },
        )
        training_args = TrainingArguments(**training_kwargs)

        trainer = self._build_trainer(
            Trainer=Trainer,
            model=model,
            training_args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        metrics = trainer.evaluate()

        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        with open(os.path.join(output_dir, "label_classes.json"), "w", encoding="utf-8") as f:
            json.dump(label_encoder.classes_.tolist(), f, indent=2, ensure_ascii=True)

        return metrics

    def train_gpt(
        self,
        data: str,
        prompt_col: str = "prompt",
        response_col: str = "response",
        model_name: str = "distilgpt2",
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

        os.makedirs(output_dir, exist_ok=True)

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

        model = AutoModelForCausalLM.from_pretrained(model_name)

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

        training_kwargs = self._make_training_args_kwargs(
            TrainingArguments,
            {
                "output_dir": output_dir,
                "learning_rate": learning_rate,
                "per_device_train_batch_size": batch_size,
                "per_device_eval_batch_size": batch_size,
                "num_train_epochs": epochs,
                "weight_decay": 0.01,
                "logging_steps": 10,
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
                "report_to": "none",
                "fp16": torch.cuda.is_available(),
            },
        )
        training_args = TrainingArguments(**training_kwargs)

        trainer = self._build_trainer(
            Trainer=Trainer,
            model=model,
            training_args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        trainer.train()
        metrics = trainer.evaluate()

        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        return metrics
