# FinTech Chatbot Starter (Classical ML + Transfer Learning)

This project gives you a ready baseline to build a fintech chatbot using:

- Classical ML: Naive Bayes, Logistic Regression, SVM
- Transfer Learning: BERT intent classifier and GPT-style causal language model

It is designed to run in both:

- Google Colab
- Custom desktop/laptop Python environment

## Project Structure

- `main.py` -> Single entrypoint to run download, processing, and training
- `processing/` -> Class-based modular pipeline (downloader, preprocessor, trainers)
- `train_classical.py` -> Train Naive Bayes / Logistic Regression / SVM intent classifier
- `train_transfer.py` -> Train BERT classifier or GPT-style model
- `chatbot.py` -> Run chatbot in interactive terminal
- `dataset/` -> Dataset templates

## 1) Setup

### Desktop setup

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Google Colab setup

```python
!pip install -q -r requirements.txt
```

If your files are on Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/AI-Project
```

## 2) Train Classical ML Models

### 2.0 Prepare dataset from proposal sources

This step derives data into your template from:

- BANKING77 (auto download if `datasets` is installed and internet is available)
- Optional Bitext/Kaggle files (if you provide local paths)
- Regional augmentation template for Southeast Asia usage

To explicitly download BANKING77 into a local raw folder first:

```bash
python main.py download --output_dir dataset/raw/banking77
```

```bash
python main.py process --output_dir dataset/processed --banking77_raw_dir dataset/raw/banking77
```

Optional with extra files:

```bash
python prepare_datasets.py \
  --output_dir dataset/processed \
  --bitext_path dataset/bitext_retail_banking.csv \
  --kaggle_path dataset/kaggle_banking.csv
```

Generated files:

- `dataset/processed/fintech_intents_prepared.csv`
- `dataset/processed/fintech_intents_train.csv`
- `dataset/processed/fintech_intents_val.csv`
- `dataset/processed/fintech_intents_test.csv`
- `dataset/processed/fintech_gpt_train.jsonl`
- `dataset/processed/dataset_stats.json`

### 2.1 Train models on prepared data

```bash
python main.py train-classical --model logreg --data dataset/processed/fintech_intents_train.csv
python main.py train-classical --model naive_bayes --data dataset/processed/fintech_intents_train.csv
python main.py train-classical --model svm --data dataset/processed/fintech_intents_train.csv
```

Saved artifacts:

- `models/classical/<model>_pipeline.joblib`
- `models/classical/responses_by_intent.json`
- `models/classical/metadata.json`

## 3) Train Transfer Learning Models

### 3.1 BERT intent classification

```bash
python main.py train-intent \
  --data dataset/processed/fintech_intents_train.csv \
  --model_name distilbert-base-uncased \
  --output_dir models/bert_intent
```

### 3.2 GPT-style chatbot fine-tuning

```bash
python main.py train-gpt \
  --data dataset/processed/fintech_gpt_train.jsonl \
  --model_name distilgpt2 \
  --output_dir models/gpt_finetuned
```

### 3.3 One-command full pipeline

```bash
python main.py all --classical_model logreg
```

## 4) Run Chatbot

### Classical model chatbot

```bash
python chatbot.py --backend classical \
  --model_path models/classical/logreg_pipeline.joblib \
  --responses_path models/classical/responses_by_intent.json
```

### BERT intent chatbot

```bash
python chatbot.py --backend bert \
  --model_dir models/bert_intent \
  --responses_path models/classical/responses_by_intent.json
```

### GPT chatbot

```bash
python chatbot.py --backend gpt --model_dir models/gpt_finetuned
```

## 5) What to customize after you send your prompt

- Edit `dataset/regional_augmentation_template.csv` to add Cambodia/SEA phrasing.
- Keep `dataset/core_intents_guide.csv` as your intent/rule reference sheet.
- Increase training examples per intent for better quality.
- Tune model hyperparameters: `epochs`, `batch_size`, `learning_rate`, and `max_length`.
- Add guardrails for compliance and safe financial advice handling.

## Notes

- This starter is for educational/prototyping use.
- For production fintech use, include full security, privacy, audit logs, and strict policy checks.
