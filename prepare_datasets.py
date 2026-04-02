import argparse
import json

from processing.preprocessor import FintechDatasetProcessor


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive and preprocess fintech datasets into training templates")
    parser.add_argument("--output_dir", type=str, default="dataset/processed")
    parser.add_argument("--regional_aug", type=str, default="dataset/regional_augmentation_template.csv")
    parser.add_argument("--bitext_path", type=str, default="")
    parser.add_argument("--kaggle_path", type=str, default="")
    parser.add_argument("--max_banking77_rows", type=int, default=0)
    parser.add_argument("--banking77_raw_dir", type=str, default="dataset/raw/banking77")
    parser.add_argument("--min_samples_per_intent", type=int, default=40)
    parser.add_argument("--no_synthetic_fill", action="store_true")
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

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

    print("Dataset preparation complete.")
    print(json.dumps({k: v for k, v in stats.items() if k != "files"}, indent=2, ensure_ascii=True))
    print(f"Saved full dataset: {stats['files']['full']}")
    print(
        "Saved split datasets: "
        f"{stats['files']['train']}, {stats['files']['val']}, {stats['files']['test']}"
    )
    print(f"Saved GPT train file: {stats['files']['gpt']}")


if __name__ == "__main__":
    main()
