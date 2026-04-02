# FinTech Chatbot Starter

This project supports three chatbot backends:

- Classical ML (LogReg, Naive Bayes, SVM)
- BERT intent classifier
- GPT-style response model

It runs on local Python or Google Colab.

## Quick Start

### Local (Windows PowerShell)

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

`python main.py` opens an interactive menu:

1. Best Classical Model
2. GPT
3. BERT
4. Exit

Then choose whether to train first (`y/n`) before chatting.

### Google Colab

```python
!pip install -q -r requirements.txt
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/AI-Project
!python main.py
```

## Clean Data Pipeline

To rebuild cleaned processed data:

```bash
python main.py process --output_dir dataset/processed --banking77_raw_dir dataset/raw/banking77
```

Generated files:

- `dataset/processed/fintech_intents_prepared.csv`
- `dataset/processed/fintech_intents_train.csv`
- `dataset/processed/fintech_intents_val.csv`
- `dataset/processed/fintech_intents_test.csv`
- `dataset/processed/fintech_gpt_train.jsonl`
- `dataset/processed/dataset_stats.json`

## Training Commands

### Classical

```bash
python main.py train-classical --model logreg --data dataset/processed/fintech_intents_train.csv --output_dir models/classical
python main.py train-classical --model naive_bayes --data dataset/processed/fintech_intents_train.csv --output_dir models/classical
python main.py train-classical --model svm --data dataset/processed/fintech_intents_train.csv --output_dir models/classical
```

### BERT

```bash
python main.py train-intent --data dataset/processed/fintech_intents_train.csv --model_name distilbert-base-uncased --output_dir models/bert_intent
```

### GPT

```bash
python main.py train-gpt --data dataset/processed/fintech_gpt_train.jsonl --model_name distilgpt2 --output_dir models/gpt_finetuned
```

## One-Command Pipeline

```bash
python main.py quickstart --mode best_classical
python main.py quickstart --mode bert
python main.py quickstart --mode gpt
```

Modes:

- `best_classical`: trains LogReg
- `classical`: trains selected classical model (`--classical_model`)
- `bert`: trains BERT intent model
- `gpt`: trains GPT model

## Terminal Output Style

Commands now print organized sections:

- `Download Summary`
- `Dataset Summary`
- `Classical Training Summary` or `BERT/GPT Training Summary`
- `Saved Artifacts`

For full raw JSON output, add `--json_output` to supported commands.

## Direct Chatbot Commands

```bash
python chatbot.py --backend classical --model_path models/classical/logreg_pipeline.joblib --responses_path models/classical/responses_by_intent.json
python chatbot.py --backend bert --model_dir models/bert_intent --responses_path models/classical/responses_by_intent.json
python chatbot.py --backend gpt --model_dir models/gpt_finetuned
```

## Notes

- This project is for education/prototyping.
- For production fintech, add compliance, audit logs, robust security, and policy controls.
