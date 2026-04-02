import json
import os
import random
import re
import hashlib
from typing import Dict, List, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from .constants import BANKING77_TO_CORE, INTENT_RESPONSE_TEMPLATE, SYNTHETIC_REGIONAL_PATTERNS

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None


class FintechDatasetProcessor:
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        random.seed(random_seed)

    def clean_text(self, text: str) -> str:
        text = str(text).lower().strip()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _detect_text_col(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        # Map normalized column names to original names for stable lookup.
        normalized_map = {str(col).strip().lower(): col for col in df.columns}
        for candidate in candidates:
            key = str(candidate).strip().lower()
            if key in normalized_map:
                return normalized_map[key]
        return None

    def _apply_banking77_mapping(self, df: pd.DataFrame, source_label_col: str, source_name: str) -> pd.DataFrame:
        reverse_map = {}
        for core_intent, src_labels in BANKING77_TO_CORE.items():
            for src in src_labels:
                reverse_map[src] = core_intent

        work_df = df.copy()
        work_df["intent"] = work_df[source_label_col].astype(str).map(reverse_map)
        work_df = work_df[work_df["intent"].notna()].copy()

        return pd.DataFrame(
            {
                "text": work_df["text"].astype(str),
                "intent": work_df["intent"].astype(str),
                "response": work_df["intent"].map(INTENT_RESPONSE_TEMPLATE),
                "language": "en",
                "channel": "mobile",
                "source": source_name,
            }
        )

    def _infer_intent_from_text(self, text: str) -> Optional[str]:
        t = f" {text.lower()} "

        def has_any(words: List[str]) -> bool:
            return any(w in t for w in words)

        if has_any([" wallet top up", " top up", "topup "]):
            return "wallet_topup"
        if has_any([" loan ", " interest rate", "eligible for quick loan", "loan approval"]):
            return "loan_info"
        if has_any([" kyc", " verify my identity", "documents needed", "verification"]):
            return "kyc_verification"
        if has_any([" open account", " account registration", "requirements to open", "new account"]):
            return "account_opening"
        if has_any([" bill ", " electricity", " water bill", " internet bill", " auto pay"]):
            return "bill_payment"
        if has_any([" balance ", "money left", "account balance"]):
            return "balance_inquiry"
        if has_any([" fraud", " suspicious", " unauthorized", "unrecognized", "unrecognised", "not mine", "stolen"]):
            return "fraud_report"
        if has_any([" fee", " charged", " charge", " extra cost", " service fee"]):
            return "fee_inquiry"
        if has_any([" card "]) and has_any(
            [
                " not working",
                " declined",
                " broken",
                " damaged",
                " lost",
                " blocked",
                " not received",
                " hasn t arrived",
                " not arrived",
                " card arrival",
                " track my card",
                " delivery",
            ]
        ):
            return "card_issue"
        if has_any([" transfer ", " beneficiary", " recipient", " pending", " transaction status", " not received "]):
            if has_any([" how to transfer", " send money", " transfer money", " move money"]):
                return "fund_transfer"
            return "transaction_status"
        if has_any([" withdrawal", " withdraw", " atm", " cash out", " take out money"]):
            return "cash_withdrawal"

        return None

    def _repair_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        repaired = df.copy()

        # Remove low-value synthetic placeholders.
        repaired = repaired[~repaired["text"].str.contains(r"\bsample_\d+\b", regex=True, na=False)].copy()

        inferred = repaired["text"].astype(str).map(self._infer_intent_from_text)
        repaired["intent"] = inferred.fillna(repaired["intent"]).astype(str)

        repaired["response"] = repaired["intent"].map(INTENT_RESPONSE_TEMPLATE).fillna(repaired["response"])
        repaired = repaired.drop_duplicates(subset=["text", "intent"]).reset_index(drop=True)
        return repaired

    def load_banking77_from_local(self, raw_dir: str, max_rows: Optional[int] = None) -> pd.DataFrame:
        train_path = os.path.join(raw_dir, "train.csv")
        test_path = os.path.join(raw_dir, "test.csv")

        if not (os.path.exists(train_path) and os.path.exists(test_path)):
            raise FileNotFoundError(
                f"Local BANKING77 files not found. Expected: {train_path} and {test_path}"
            )

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        df = pd.concat([train_df, test_df], ignore_index=True)

        if "label_name" not in df.columns:
            raise ValueError("Local BANKING77 CSV must include 'label_name'.")

        mapped = self._apply_banking77_mapping(df, source_label_col="label_name", source_name="banking77_local")

        if max_rows is not None and max_rows > 0:
            mapped = mapped.sample(min(max_rows, len(mapped)), random_state=self.random_seed)

        return mapped

    def load_banking77_from_hf(self, max_rows: Optional[int] = None) -> pd.DataFrame:
        if load_dataset is None:
            raise RuntimeError("datasets package is not available.")

        ds_train = load_dataset("banking77", split="train")
        ds_test = load_dataset("banking77", split="test")

        df = pd.concat([ds_train.to_pandas(), ds_test.to_pandas()], ignore_index=True)
        label_feature = ds_train.features["label"]
        id_to_name = {i: name for i, name in enumerate(label_feature.names)}
        df["label_name"] = df["label"].map(id_to_name)

        mapped = self._apply_banking77_mapping(df, source_label_col="label_name", source_name="banking77")

        if max_rows is not None and max_rows > 0:
            mapped = mapped.sample(min(max_rows, len(mapped)), random_state=self.random_seed)

        return mapped

    def load_optional_external_dataset(self, path: str, source_name: str) -> pd.DataFrame:
        if not path or not os.path.exists(path):
            return pd.DataFrame(columns=["text", "intent", "response", "language", "channel", "source"])

        if path.lower().endswith(".jsonl"):
            df = pd.read_json(path, lines=True)
        else:
            df = pd.read_csv(path)

        text_col = self._detect_text_col(df, ["text", "query", "question", "utterance"])
        intent_col = self._detect_text_col(df, ["intent", "label", "category", "class"])
        response_col = self._detect_text_col(df, ["response", "answer", "reply"])

        if text_col is None or intent_col is None:
            return pd.DataFrame(columns=["text", "intent", "response", "language", "channel", "source"])

        data = pd.DataFrame()
        data["text"] = df[text_col].astype(str)
        data["intent"] = df[intent_col].astype(str).str.lower().str.replace(" ", "_", regex=False)

        if response_col is not None:
            data["response"] = df[response_col].astype(str)
        else:
            data["response"] = data["intent"].map(INTENT_RESPONSE_TEMPLATE).fillna(
                "Please check your account dashboard or contact support for this request."
            )

        data["language"] = "en"
        data["channel"] = "mobile"
        data["source"] = source_name
        return data

    def load_regional_augmentation(self, path: str) -> pd.DataFrame:
        if not path or not os.path.exists(path):
            return pd.DataFrame(columns=["text", "intent", "response", "language", "channel", "source"])

        df = pd.read_csv(path)
        expected_cols = ["text", "intent", "response"]
        for col in expected_cols:
            if col not in df.columns:
                raise ValueError(f"Regional file missing required column '{col}': {path}")

        if "language" not in df.columns:
            df["language"] = "en"
        if "channel" not in df.columns:
            df["channel"] = "mobile"
        if "source" not in df.columns:
            df["source"] = "regional_aug"

        return df[["text", "intent", "response", "language", "channel", "source"]]

    def synthesize_missing_examples(self, df: pd.DataFrame, min_per_intent: int) -> pd.DataFrame:
        rows = []
        counts = df["intent"].value_counts().to_dict()

        for intent in INTENT_RESPONSE_TEMPLATE:
            have = counts.get(intent, 0)
            need = max(0, min_per_intent - have)
            if need == 0:
                continue

            patterns = SYNTHETIC_REGIONAL_PATTERNS.get(intent, [intent.replace("_", " ")])
            for i in range(need):
                base = patterns[i % len(patterns)]
                seed = int(hashlib.md5(f"{intent}-{i}".encode("utf-8")).hexdigest()[:8], 16)
                rng = random.Random(seed)
                style = i % 4
                if style == 0:
                    variant = base
                elif style == 1:
                    variant = f"can you help with {base}"
                elif style == 2:
                    variant = f"please check {base}"
                else:
                    closers = ["today", "for me", "right now", "in the app"]
                    variant = f"i need support for {base} {rng.choice(closers)}"
                rows.append(
                    {
                        "text": variant,
                        "intent": intent,
                        "response": INTENT_RESPONSE_TEMPLATE[intent],
                        "language": "en",
                        "channel": "mobile",
                        "source": "synthetic_seed",
                    }
                )

        return pd.DataFrame(rows)

    def stratified_split(self, df: pd.DataFrame):
        train_df, temp_df = train_test_split(
            df,
            test_size=0.2,
            random_state=self.random_seed,
            stratify=df["intent"] if df["intent"].nunique() > 1 else None,
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=self.random_seed,
            stratify=temp_df["intent"] if temp_df["intent"].nunique() > 1 else None,
        )

        return train_df, val_df, test_df

    def save_jsonl(self, df: pd.DataFrame, output_path: str) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                item = {
                    "prompt": row["text"],
                    "response": row["response"],
                    "category": row["intent"],
                }
                f.write(json.dumps(item, ensure_ascii=True) + "\n")

    def process(
        self,
        output_dir: str = "dataset/processed",
        regional_aug: str = "dataset/regional_augmentation_template.csv",
        bitext_path: str = "",
        kaggle_path: str = "",
        max_banking77_rows: int = 0,
        banking77_raw_dir: str = "dataset/raw/banking77",
        min_samples_per_intent: int = 40,
        no_synthetic_fill: bool = False,
    ) -> dict:
        os.makedirs(output_dir, exist_ok=True)

        frames = []
        max_rows = max_banking77_rows if max_banking77_rows > 0 else None

        try:
            frames.append(self.load_banking77_from_local(banking77_raw_dir, max_rows=max_rows))
            print(f"Loaded BANKING77 from local raw folder: {banking77_raw_dir}")
        except Exception as local_ex:
            print(f"Warning: Local BANKING77 load failed: {local_ex}")
            try:
                frames.append(self.load_banking77_from_hf(max_rows=max_rows))
                print("Loaded BANKING77 from Hugging Face and mapped to core intents.")
            except Exception as hf_ex:
                print(f"Warning: Hugging Face BANKING77 load failed: {hf_ex}")

        if bitext_path:
            frames.append(self.load_optional_external_dataset(bitext_path, source_name="bitext"))

        if kaggle_path:
            frames.append(self.load_optional_external_dataset(kaggle_path, source_name="kaggle"))

        frames.append(self.load_regional_augmentation(regional_aug))

        df = pd.concat(frames, ignore_index=True)
        if df.empty:
            raise RuntimeError("Combined dataset is empty. Provide at least one valid source.")

        df["text"] = df["text"].astype(str).map(self.clean_text)
        df["intent"] = df["intent"].astype(str).str.strip().str.lower().str.replace(" ", "_", regex=False)
        df["response"] = df["response"].astype(str).str.strip()
        df = df[(df["text"].str.len() > 2) & (df["intent"].str.len() > 0)].copy()

        allowed_intents = set(INTENT_RESPONSE_TEMPLATE.keys())
        df = df[df["intent"].isin(allowed_intents)].copy()

        if not no_synthetic_fill:
            synthetic_df = self.synthesize_missing_examples(df, min_per_intent=min_samples_per_intent)
            if not synthetic_df.empty:
                df = pd.concat([df, synthetic_df], ignore_index=True)

        df = self._repair_quality(df)

        df = df.drop_duplicates(subset=["text", "intent"]).reset_index(drop=True)
        train_df, val_df, test_df = self.stratified_split(df)

        full_path = os.path.join(output_dir, "fintech_intents_prepared.csv")
        train_path = os.path.join(output_dir, "fintech_intents_train.csv")
        val_path = os.path.join(output_dir, "fintech_intents_val.csv")
        test_path = os.path.join(output_dir, "fintech_intents_test.csv")
        gpt_path = os.path.join(output_dir, "fintech_gpt_train.jsonl")

        df.to_csv(full_path, index=False)
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        self.save_jsonl(df, gpt_path)

        stats = {
            "total_samples": int(len(df)),
            "train_samples": int(len(train_df)),
            "val_samples": int(len(val_df)),
            "test_samples": int(len(test_df)),
            "intents": sorted(df["intent"].unique().tolist()),
            "intent_distribution": df["intent"].value_counts().to_dict(),
            "sources": df["source"].value_counts().to_dict(),
            "files": {
                "full": full_path,
                "train": train_path,
                "val": val_path,
                "test": test_path,
                "gpt": gpt_path,
            },
        }

        stats_path = os.path.join(output_dir, "dataset_stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=True)

        stats["stats_path"] = stats_path
        return stats
