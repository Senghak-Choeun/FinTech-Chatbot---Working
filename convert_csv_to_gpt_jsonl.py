"""
Convert fintech_intents_final_train.csv to JSONL format for DistilGPT2 fine-tuning.

Format: {"text":"User: <question>\nAssistant: <response>"}
"""

import pandas as pd
import json
from pathlib import Path
from collections import Counter

def convert_csv_to_gpt_jsonl(
    csv_path: str = "dataset/processed/fintech_intents_final_train.csv",
    output_path: str = "dataset/processed/fintech_gpt_train.jsonl",
    include_intent: bool = False,
    is_test: bool = False
):
    """
    Convert training CSV to JSONL format for GPT fine-tuning.
    
    Args:
        csv_path: Path to input CSV file
        output_path: Path to output JSONL file
        include_intent: If True, include intent in the format:
                       "Intent: <intent>\nUser: ...\nAssistant: ..."
        is_test: If True, treat as test data (text only, no response)
    """
    
    # Load CSV
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  Total rows in CSV: {len(df)}")
    
    # Show columns
    print(f"  Columns: {list(df.columns)}")
    
    if is_test:
        # Test data: just text, no response column
        df_clean = df.dropna(subset=['text']).copy()
        missing_count = len(df) - len(df_clean)
        if missing_count > 0:
            print(f"  Skipped {missing_count} rows with missing text")
        
        # For test data, just keep the text as-is
        df_unique = df_clean.copy()
        intent_counts = Counter(df_unique.get('intent', []))
        print(f"  Number of unique intents: {len(intent_counts)}")
        if intent_counts:
            print(f"  Intent distribution:")
            for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
                print(f"    - {intent}: {count} examples")
        
        # Convert to JSONL format (test data doesn't need Assistant response)
        print(f"\nConverting to JSONL format...")
        jsonl_lines = []
        
        for idx, row in df_unique.iterrows():
            text = row['text'].strip()
            
            if include_intent and 'intent' in row and pd.notna(row['intent']):
                intent = row['intent'].strip()
                formatted_text = f"Intent: {intent}\nUser: {text}"
            else:
                formatted_text = f"User: {text}"
            
            jsonl_obj = {"text": formatted_text}
            jsonl_lines.append(json.dumps(jsonl_obj))
        
        duplicates_count = 0
    else:
        # Training data: has text and response
        df_clean = df.dropna(subset=['text', 'response']).copy()
        missing_count = len(df) - len(df_clean)
        if missing_count > 0:
            print(f"  Skipped {missing_count} rows with missing text/response")
        
        # Remove duplicate (text, response) pairs - keep first occurrence
        df_clean['pair'] = df_clean['text'] + " || " + df_clean['response']
        duplicates_count = len(df_clean) - len(df_clean.drop_duplicates(subset=['pair']))
        df_unique = df_clean.drop_duplicates(subset=['pair']).copy()
        print(f"  Removed {duplicates_count} duplicate pairs")
        
        # Count intents
        intent_counts = Counter(df_unique['intent'])
        print(f"  Number of unique intents: {len(intent_counts)}")
        print(f"  Intent distribution:")
        for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
            print(f"    - {intent}: {count} examples")
        
        # Convert to JSONL format
        print(f"\nConverting to JSONL format...")
        jsonl_lines = []
        
        for idx, row in df_unique.iterrows():
            text = row['text'].strip()
            response = row['response'].strip()
            
            if include_intent:
                # Format with intent
                intent = row['intent'].strip()
                formatted_text = f"Intent: {intent}\nUser: {text}\nAssistant: {response}"
            else:
                # Simple format
                formatted_text = f"User: {text}\nAssistant: {response}"
            
            jsonl_obj = {"text": formatted_text}
            jsonl_lines.append(json.dumps(jsonl_obj))
    
    # Write JSONL file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in jsonl_lines:
            f.write(line + '\n')
    
    print(f"  ✅ Saved {len(jsonl_lines)} examples to {output_path}")
    
    # Summary statistics
    data_type = "TEST" if is_test else "TRAINING"
    print(f"\n{'='*60}")
    print(f"{data_type} DATA CONVERSION SUMMARY")
    print(f"{'='*60}")
    print(f"Original rows:          {len(df)}")
    print(f"Rows after cleaning:    {len(df_clean)}")
    print(f"Duplicates removed:     {duplicates_count}")
    print(f"Final {data_type.lower()} examples: {len(jsonl_lines)}")
    print(f"Unique intents:         {len(intent_counts)}")
    print(f"Output file:            {output_path}")
    print(f"{'='*60}\n")
    
    # Show sample
    if len(jsonl_lines) > 0:
        print(f"Sample JSONL entries (first 2):")
        for i, line in enumerate(jsonl_lines[:2]):
            obj = json.loads(line)
            print(f"\n  [{i+1}] {obj['text'][:80]}...")
    
    return len(jsonl_lines), len(intent_counts), duplicates_count


def explain_test_evaluation():
    """
    Explain how to use fintech_intents_final_test.csv for GPT evaluation.
    """
    explanation = """
    ╔════════════════════════════════════════════════════════════════╗
    ║  USING fintech_intents_final_test.csv FOR GPT EVALUATION       ║
    ╚════════════════════════════════════════════════════════════════╝

    After fine-tuning DistilGPT2 on fintech_gpt_train.jsonl, evaluate 
    using fintech_intents_final_test.csv:

    EVALUATION METHODS:
    ─────────────────────────────────────────────────────────────

    1. PERPLEXITY (Language Model Quality)
       - Convert test CSV to JSONL (same format as training)
       - Load fine-tuned model and evaluate on test set
       - Perplexity measures how well model predicts next tokens
       - Lower perplexity = better generalization
       
       Code:
       ```python
       from transformers import AutoModelForCausalLM, AutoTokenizer
       import torch
       
       model = AutoModelForCausalLM.from_pretrained("path/to/finetuned/model")
       tokenizer = AutoTokenizer.from_pretrained("path/to/finetuned/model")
       
       # Load test JSONL
       test_texts = [json.loads(line)["text"] for line in open("test.jsonl")]
       
       # Calculate perplexity
       perplexities = []
       for text in test_texts:
           inputs = tokenizer(text, return_tensors="pt")
           loss = model(**inputs, labels=inputs["input_ids"]).loss
           perplexity = torch.exp(loss)
           perplexities.append(perplexity.item())
       
       avg_perplexity = sum(perplexities) / len(perplexities)
       print(f"Average Perplexity: {avg_perplexity:.2f}")
       ```

    2. RESPONSE QUALITY (Semantic Evaluation)
       - Generate responses for test prompts
       - Compare with ground truth responses
       - Calculate BLEU, ROUGE, or similarity scores
       
       Code:
       ```python
       model = AutoModelForCausalLM.from_pretrained("path/to/finetuned/model")
       tokenizer = AutoTokenizer.from_pretrained("path/to/finetuned/model")
       
       prompt = "User: How do I transfer money?"
       inputs = tokenizer(prompt, return_tensors="pt")
       outputs = model.generate(**inputs, max_length=100)
       response = tokenizer.decode(outputs[0], skip_special_tokens=True)
       ```

    3. HELD-OUT TEST SET (Best Practice)
       - Keep fintech_intents_final_test.csv completely separate
       - Don't use for training or hyperparameter tuning
       - Use ONLY for final evaluation after training completes
       - Reports true generalization performance

    WORKFLOW:
    ─────────────────────────────────────────────────────────────
    1. Convert test CSV to JSONL (same process, different file)
    2. Fine-tune model on fintech_gpt_train.jsonl
    3. Evaluate on fintech_gpt_test.jsonl (new file, not included in training)
    4. Compare training loss vs test perplexity:
       - Training loss: ~0.8-1.2 (expected)
       - Test perplexity: 20-50 (typical range)
       - Big gap = overfitting → may need regularization

    DO NOT:
    ─────────────────────────────────────────────────────────────
    ❌ Mix test data into training JSONL (data leakage)
    ❌ Use test set during training for early stopping
    ❌ Evaluate on test set during hyperparameter tuning
    ✅ Keep test completely separate until final evaluation
    """
    print(explanation)


if __name__ == "__main__":
    # Convert training data to JSONL (simple format)
    print("="*70)
    print("CONVERTING TRAINING DATA")
    print("="*70 + "\n")
    total_train, intents_train, dupes_train = convert_csv_to_gpt_jsonl(
        csv_path="dataset/processed/fintech_intents_final_train.csv",
        output_path="dataset/processed/fintech_gpt_train.jsonl",
        include_intent=False,
        is_test=False
    )
    
    # Convert test data to JSONL (test data format - no responses)
    print("\n" + "="*70)
    print("CONVERTING TEST DATA")
    print("="*70 + "\n")
    total_test, intents_test, dupes_test = convert_csv_to_gpt_jsonl(
        csv_path="dataset/processed/fintech_intents_final_test.csv",
        output_path="dataset/processed/fintech_gpt_test.jsonl",
        include_intent=False,
        is_test=True
    )
    
    # Show test data evaluation explanation
    print("\n")
    explain_test_evaluation()
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"✅ Training JSONL: {total_train:,} examples")
    print(f"✅ Test JSONL:     {total_test:,} examples")
    print(f"📊 Intents:       {intents_train} unique intents")
    print(f"📁 Output files:")
    print(f"   - dataset/processed/fintech_gpt_train.jsonl")
    print(f"   - dataset/processed/fintech_gpt_test.jsonl")
    print("="*70 + "\n")
    
    print("NEXT STEPS:")
    print("──────────────────────────────────────────────")
    print("1. Train GPT with separate test set:")
    print("   python main.py train-gpt \\")
    print("     --data dataset/processed/fintech_gpt_train.jsonl \\")
    print("     --test_data dataset/processed/fintech_gpt_test.jsonl")
    print("\n2. Model will evaluate on held-out test set")
    print("   (no data leakage, fair comparison)")
    print("\n3. Results saved to models/gpt_finetuned/")
    print("──────────────────────────────────────────────\n")
