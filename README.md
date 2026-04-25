# FinTech Chatbot Project (Classical ML + Transfer Learning)

This repository provides an end-to-end pipeline for a fintech chatbot with:

- Classical intent classification: Naive Bayes, Logistic Regression, SVM
- Transfer learning: BERT intent classifier and GPT-style response model
- Dataset preparation with BANKING77 + local Cambodia/SEA augmentation data

## Project Structure

- `main.py`: unified CLI for download, preprocessing, and training
- `prepare_datasets.py`: direct dataset preparation script
- `chatbot.py`: run chatbot backends (classical, BERT, GPT)
- `processing/`: downloader, preprocessor, and trainer implementations
- `dataset/`: raw inputs, templates, and processed outputs
- `models/`: trained model artifacts and metadata

## Setup

### Local environment (Windows PowerShell)

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Start interactive mode

```bash
python main.py
```

If no subcommand is provided, `main.py` launches an interactive flow that can:

- Optionally download BANKING77
- Prepare dataset files
- Train Classical, BERT, or GPT models
- Resume from checkpoint or continue from best weights (BERT/GPT)

## CLI Workflows

### 1) Download BANKING77 locally

```bash
python main.py download --output_dir dataset/raw/banking77
```

### 2) Prepare datasets

```bash
python main.py process \
  --output_dir dataset/processed \
  --banking77_raw_dir dataset/raw/banking77
```

Key options:

- `--regional_aug` (default: `dataset/cambodian_asia_banking_qa_2016.csv`)
- `--asia_custom_aug` (default: `dataset/cambodian_asia_banking_custom.csv`)
- `--min_samples_per_intent` (default: `40`)
- `--no_synthetic_fill`
- `--cambodia_only`

You can run the same preprocessing via:

```bash
python prepare_datasets.py --output_dir dataset/processed
```

### 3) Train classical models

```bash
python main.py train-classical --model all --data dataset/processed/fintech_intents_train.csv
```

Model choices: `logreg`, `naive_bayes`, `svm`, `all`

### 4) Train BERT intent model

```bash
python main.py train-intent \
  --data dataset/processed/fintech_intents_train.csv \
  --model_name distilbert-base-uncased \
  --output_dir models/bert_intent
```

Resume from checkpoint:

```bash
python main.py train-intent \
  --data dataset/processed/fintech_intents_train.csv \
  --output_dir models/bert_intent \
  --resume_from_checkpoint models/bert_intent/checkpoint-XXX
```

Continue from best weights:

```bash
python main.py train-intent \
  --data dataset/processed/fintech_intents_train.csv \
  --init_model_path models/bert_intent/best_model \
  --output_dir models/bert_intent_v2
```

### 5) Train GPT-style model

```bash
python main.py train-gpt \
  --data dataset/processed/fintech_gpt_train.jsonl \
  --model_name distilgpt2 \
  --output_dir models/gpt_finetuned
```

Resume from checkpoint:

```bash
python main.py train-gpt \
  --data dataset/processed/fintech_gpt_train.jsonl \
  --output_dir models/gpt_finetuned \
  --resume_from_checkpoint models/gpt_finetuned/checkpoint-XXX
```

### 6) Run full pipeline (download + process + classical)

```bash
python main.py all --classical_model all
```

## Generated Outputs

Typical preprocessing outputs under `dataset/processed*/`:

- `fintech_intents_prepared.csv`
- `fintech_intents_train.csv`
- `fintech_intents_val.csv`
- `fintech_intents_test.csv`
- `fintech_gpt_train.jsonl`
- `dataset_stats.json`

Classical training output directory includes:

- `<model>_pipeline.joblib`
- `responses_by_intent.json`
- `metadata.json`

## Run Chatbot

Classical backend:

```bash
python chatbot.py --backend classical \
  --model_path models/classical_main_check/naive_bayes_pipeline.joblib \
  --responses_path models/classical_main_check/responses_by_intent.json
```

BERT backend:

```bash
python chatbot.py --backend bert \
  --model_dir models/bert_intent \
  --responses_path models/classical_main_check/responses_by_intent.json
```

GPT backend:

```bash
python chatbot.py --backend gpt --model_dir models/gpt_finetuned
```

## Notes

- This project is intended for educational and prototyping use.
- Do not use real customer secrets or sensitive personal data in training files.
- For production use, add strict security, privacy, audit logging, and policy guardrails.
