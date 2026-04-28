"""
Script to rebuild responses_by_intent.json with the new structure:
{
  "intent": [
    {"text": "training question", "response": "response"},
    ...
  ]
}

This structure enables similarity-based response retrieval instead of random selection.
"""

import json
import pandas as pd
from pathlib import Path


def rebuild_responses_from_dataset(
    dataset_csv: str,
    output_json: str,
) -> None:
    """
    Build responses_by_intent.json from training dataset CSV.

    Args:
        dataset_csv: Path to fintech_intents_train.csv
        output_json: Path to save responses_by_intent.json
    """
    print(f"Reading training data from {dataset_csv}...")
    df = pd.read_csv(dataset_csv)

    # Group by intent and collect text + response pairs
    responses_by_intent = {}
    for intent in df["intent"].unique():
        intent_data = df[df["intent"] == intent]
        examples = [
            {
                "text": row["text"],
                "response": row["response"]
            }
            for _, row in intent_data.iterrows()
        ]
        responses_by_intent[intent] = examples
        print(f"  {intent}: {len(examples)} examples")

    print(f"\nSaving to {output_json}...")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(responses_by_intent, f, ensure_ascii=False, indent=2)

    print(f"✓ Built responses_by_intent.json with {len(responses_by_intent)} intents")
    print(f"  Total training examples: {len(df)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Rebuild responses_by_intent.json for similarity-based retrieval"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset/processed/fintech_intents_final_train.csv",
        help="Path to training dataset CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/classical/responses_by_intent.json",
        help="Path to output responses_by_intent.json",
    )
    args = parser.parse_args()

    rebuild_responses_from_dataset(args.dataset, args.output)
