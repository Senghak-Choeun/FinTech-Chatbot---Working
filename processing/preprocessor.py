import json
import os
import random
import re
import hashlib
from typing import Dict, List, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from .constants import BANKING77_TO_CORE, INTENT_RESPONSE_TEMPLATE, INTENT_RESPONSE_VARIANTS, SYNTHETIC_REGIONAL_PATTERNS

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

    def _pick_response_variant(self, intent: str, text: str, source: str = "") -> str:
        variants = INTENT_RESPONSE_VARIANTS.get(intent, [])
        if not variants:
            return INTENT_RESPONSE_TEMPLATE.get(
                intent,
                "Please check your account dashboard or contact support for this request.",
            )

        key = f"{intent}|{text}|{source}".encode("utf-8")
        idx = int(hashlib.sha256(key).hexdigest(), 16) % len(variants)
        return variants[idx]

    def _extract_focus_phrase(self, text: str) -> str:
        tokens = [t for t in str(text).split() if len(t) > 2]
        if not tokens:
            return "your request"

        stopwords = {
            "please",
            "help",
            "with",
            "from",
            "that",
            "this",
            "your",
            "what",
            "when",
            "where",
            "which",
            "need",
            "want",
            "about",
            "still",
            "now",
            "how",
            "can",
        }
        keywords = [t for t in tokens if t not in stopwords]
        if not keywords:
            keywords = tokens

        return " ".join(keywords[:3])

    def _compose_multisentence_response(self, intent: str, text: str, source: str = "") -> str:
        base = self._pick_response_variant(intent=intent, text=text, source=source)
        focus = self._extract_focus_phrase(text)

        base_sentences = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", str(base).strip()) if s.strip()
        ]
        if not base_sentences:
            base_sentences = [
                "Please check your account dashboard or contact support for this request."
            ]
        base_sentence_count = len(base_sentences)

        follow_up_options = [
            f"For {focus}, check the in-app details and confirm all recipient information before you proceed.",
            f"For {focus}, review the latest status in your app history and use the provided reference when contacting support.",
            f"For {focus}, keep your notifications on so you can confirm the next update quickly.",
            f"For {focus}, make sure your profile and security settings are up to date to avoid delays.",
        ]

        regional_options = [
            "This flow is commonly used with local rails like KHQR, Bakong, PromptPay, DuitNow, and other regional payment services.",
            "If this involves cross-border payments in ASEAN, processing speed may vary by bank settlement windows and partner networks.",
            "For Cambodia and nearby markets, wallet and bank transfers can have cutoff times depending on provider and channel.",
            "Regional e-wallet and QR payments may apply provider-specific limits, so check the fee and limit preview before confirmation.",
        ]

        key = f"{intent}|{text}|{source}".encode("utf-8")
        digest = int(hashlib.sha256(key).hexdigest(), 16)
        follow_up = follow_up_options[digest % len(follow_up_options)]
        regional = regional_options[(digest // 7) % len(regional_options)]

        target_sentences = 2 + (digest % 2)
        if source in {"asia_custom", "regional_aug", "synthetic_seed"}:
            target_sentences = 3

        room = max(0, 3 - base_sentence_count)
        need_extra = min(room, max(0, target_sentences - base_sentence_count))

        if need_extra <= 0:
            return " ".join(base_sentences[:3])
        if need_extra == 1:
            return f"{' '.join(base_sentences)} {follow_up}"
        return f"{' '.join(base_sentences)} {follow_up} {regional}"

    def diversify_responses(self, df: pd.DataFrame) -> pd.DataFrame:
        work_df = df.copy()
        if "source" not in work_df.columns:
            work_df["source"] = "unknown"

        work_df["response"] = work_df.apply(
            lambda row: self._compose_multisentence_response(
                intent=str(row["intent"]),
                text=str(row["text"]),
                source=str(row.get("source", "")),
            ),
            axis=1,
        )
        return work_df

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
                variant = f"{base} sample_{i + 1}"
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

    def synthesize_missing_examples_cambodia(self, df: pd.DataFrame, min_per_intent: int) -> pd.DataFrame:
        rows = []
        counts = df["intent"].value_counts().to_dict()

        for intent in INTENT_RESPONSE_TEMPLATE:
            have = counts.get(intent, 0)
            need = max(0, min_per_intent - have)
            if need == 0:
                continue

            base = intent.replace("_", " ")
            for i in range(need):
                rows.append(
                    {
                        "text": f"cambodia {base} in phnom penh sample_{i + 1}",
                        "intent": intent,
                        "response": INTENT_RESPONSE_TEMPLATE[intent],
                        "language": "en",
                        "channel": "mobile",
                        "source": "synthetic_seed_cambodia",
                    }
                )

        return pd.DataFrame(rows)

    def filter_cambodia_focus(self, df: pd.DataFrame) -> pd.DataFrame:
        cambodia_keywords = [
            "cambodia",
            "phnom penh",
            "siem reap",
            "battambang",
            "kampot",
            "khqr",
            "bakong",
            "aba",
            "acleda",
            "wing",
            "riel",
            "khmer",
            "knyom",
            "ban te",
            "ot ban",
        ]
        pattern = "|".join(re.escape(k) for k in cambodia_keywords)

        work_df = df.copy()
        source_mask = work_df["source"].astype(str).str.lower().eq("asia_custom")
        text_mask = work_df["text"].astype(str).str.contains(pattern, case=False, regex=True)
        return work_df[source_mask | text_mask].copy()

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
        regional_aug: str = "dataset/cambodian_asia_banking_qa_2016.csv",
        asia_custom_aug: str = "dataset/cambodian_asia_banking_custom.csv",
        bitext_path: str = "",
        kaggle_path: str = "",
        max_banking77_rows: int = 0,
        banking77_raw_dir: str = "dataset/raw/banking77",
        min_samples_per_intent: int = 40,
        no_synthetic_fill: bool = False,
        cambodia_only: bool = False,
    ) -> dict:
        os.makedirs(output_dir, exist_ok=True)

        frames = []
        max_rows = max_banking77_rows if max_banking77_rows > 0 else None

        if cambodia_only:
            print("Cambodia-only mode enabled: skipping BANKING77 and non-Cambodia external sources.")
            frames.append(self.load_regional_augmentation(regional_aug))
            frames.append(self.load_regional_augmentation(asia_custom_aug))
        else:
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
            frames.append(self.load_regional_augmentation(asia_custom_aug))

        df = pd.concat(frames, ignore_index=True)
        if df.empty:
            raise RuntimeError("Combined dataset is empty. Provide at least one valid source.")

        df["text"] = df["text"].astype(str).map(self.clean_text)
        df["intent"] = df["intent"].astype(str).str.strip().str.lower().str.replace(" ", "_", regex=False)
        df["response"] = df["response"].astype(str).str.strip()
        df = df[(df["text"].str.len() > 2) & (df["intent"].str.len() > 0)].copy()

        if cambodia_only:
            df = self.filter_cambodia_focus(df)
            if df.empty:
                raise RuntimeError("Cambodia-only mode produced empty dataset. Add Cambodia-focused rows to augmentation files.")

        allowed_intents = set(INTENT_RESPONSE_TEMPLATE.keys())
        df = df[df["intent"].isin(allowed_intents)].copy()

        if not no_synthetic_fill:
            if cambodia_only:
                synthetic_df = self.synthesize_missing_examples_cambodia(df, min_per_intent=min_samples_per_intent)
            else:
                synthetic_df = self.synthesize_missing_examples(df, min_per_intent=min_samples_per_intent)
            if not synthetic_df.empty:
                df = pd.concat([df, synthetic_df], ignore_index=True)

        # Diversify assistant responses per sample while keeping intent-consistent semantics.
        df = self.diversify_responses(df)

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
            "cambodia_only": bool(cambodia_only),
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
