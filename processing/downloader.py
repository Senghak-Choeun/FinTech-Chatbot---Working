import json
import os

from datasets import load_dataset


class Banking77Downloader:
    def __init__(self, output_dir: str = "dataset/raw/banking77"):
        self.output_dir = output_dir

    def download(self) -> dict:
        os.makedirs(self.output_dir, exist_ok=True)

        ds = load_dataset("banking77")
        train = ds["train"].to_pandas()
        test = ds["test"].to_pandas()
        label_names = ds["train"].features["label"].names

        train["label_name"] = train["label"].map(lambda i: label_names[i])
        test["label_name"] = test["label"].map(lambda i: label_names[i])

        train_path = os.path.join(self.output_dir, "train.csv")
        test_path = os.path.join(self.output_dir, "test.csv")
        labels_path = os.path.join(self.output_dir, "label_names.json")

        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)

        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(label_names, f, indent=2, ensure_ascii=True)

        return {
            "output_dir": self.output_dir,
            "train_path": train_path,
            "test_path": test_path,
            "labels_path": labels_path,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "num_intents": int(len(label_names)),
        }
