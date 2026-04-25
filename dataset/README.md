# Dataset Guide

This folder contains raw data, augmentation inputs, templates, and processed outputs for the fintech chatbot pipeline.

## Main Input Files

- `cambodian_asia_banking_custom.csv`
  - Custom local augmentation phrases and responses.
- `cambodian_asia_banking_qa_1008.csv`
  - QA-style augmentation source.
- `cambodian_asia_banking_qa_2016.csv`
  - Regional augmentation source used by default in preprocessing.
- `core_intents_guide.csv`
  - Core intents reference.
- `fintech_intents_template.csv`
  - Starter intent dataset template.
- `label_response_template.csv`
  - Optional intent-to-response mapping template.
- `regional_augmentation_template.csv`
  - Template for adding Southeast Asia phrasing.

## BANKING77 Raw Data

- `raw/banking77/train.csv`
- `raw/banking77/test.csv`
- `raw/banking77/label_names.json`

These files are created by running:

```bash
python main.py download --output_dir dataset/raw/banking77
```

## Processed Output Folders

This repository currently contains multiple processed sets:

- `processed/`
- `processed_cambodia/`
- `processed_from_raw/`

Each processed folder includes:

- `fintech_intents_prepared.csv`
- `fintech_intents_train.csv`
- `fintech_intents_val.csv`
- `fintech_intents_test.csv`
- `fintech_gpt_train.jsonl`
- `dataset_stats.json`

## Preprocessing Commands

From the project root:

```bash
python prepare_datasets.py --output_dir dataset/processed
```

Equivalent command through the unified CLI:

```bash
python main.py process --output_dir dataset/processed
```

Useful options:

- `--banking77_raw_dir dataset/raw/banking77`
- `--regional_aug dataset/cambodian_asia_banking_qa_2016.csv`
- `--asia_custom_aug dataset/cambodian_asia_banking_custom.csv`
- `--min_samples_per_intent 40`
- `--max_banking77_rows 0`
- `--cambodia_only`
- `--no_synthetic_fill`

## Data Quality Notes

- Keep intent names stable and consistent across all files.
- Ensure required columns are present (`text`, `intent`, optional `response` for classical/BERT CSV workflows; `prompt` and `response` for GPT JSONL).
- Exclude real account numbers, personal IDs, and other sensitive financial data.
