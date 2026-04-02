# Dataset Templates

This folder contains starter templates for your fintech chatbot project.

## Files

- `fintech_intents_template.csv`
  - For intent classification models (Naive Bayes, Logistic Regression, SVM, BERT).
  - Required columns: `text`, `intent`
  - Optional but recommended: `response`

- `regional_augmentation_template.csv`
  - Small Southeast Asia focused augmentation examples.
  - Edit this file with your own Cambodia/SEA user phrasing.

- `core_intents_guide.csv`
  - Reference sheet for 12 core intents, keywords, and safe response templates.

- `fintech_gpt_train_template.jsonl`
  - For GPT-style causal language model fine-tuning.
  - Required fields per line: `prompt`, `response`

- `fintech_faq_template.jsonl`
  - Optional FAQ style dataset for retrieval or future expansions.

- `label_response_template.csv`
  - Optional intent-to-response mapping if you want to separate responses from training data.

## Data Rules

- Keep intent names consistent and lowercase with underscores, e.g. `balance_check`.
- Add at least 30 examples per intent for baseline classical model quality.
- For BERT/GPT fine-tuning, 200+ examples is recommended for better performance.
- Remove sensitive data such as real account numbers, cards, and personal IDs.

## One-command Preprocessing

From project root, generate cleaned and split datasets aligned to your proposal:

```bash
python prepare_datasets.py --output_dir dataset/processed
```

Output files are saved in `dataset/processed/` for direct training.
