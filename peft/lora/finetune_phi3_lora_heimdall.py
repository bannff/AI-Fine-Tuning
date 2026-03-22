#!/usr/bin/env python3
"""
LoRA Fine-tuning for Phi-3-mini on Heimdall 7-Variant Dataset
Professional version (under 200 LOC) following factory standards.
"""

import argparse
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer
from peft.runtime.training_logic import load_local_jsonl, apply_lora, get_training_args

# ============================================================================
# Configuration
# ============================================================================

DATASET_PATH = "/Volumes/Crucial X9/data-sets/heimdall/processed/heimdall-enhanced-strands-7variants.jsonl"
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
MAX_SEQ_LENGTH = 1024
OUTPUT_BASE = Path("/Volumes/Crucial X9/ai-models")
OUTPUT_DIR = OUTPUT_BASE / "phi3_mini_4k_heimdall_lora"

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-tune Phi-3-mini on Heimdall")
    parser.add_argument("--train-size", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--test", action="store_true")
    
    args = parser.parse_args()
    train_size = 50 if args.test else args.train_size
    
    # 1. Load Data
    dataset = load_local_jsonl(DATASET_PATH)
    if train_size < len(dataset):
        dataset = dataset.select(range(train_size))
    
    # 2. Setup Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map="auto", torch_dtype="auto", trust_remote_code=True
    )
    model = apply_lora(model, args.lora_rank)
    
    # 3. Preprocess
    def preprocess(example):
        messages = example.get("messages", [])
        full_text = "".join([f"<|{m['role']}|>\n{m['content']}\n" for m in messages])
        encoded = tokenizer(full_text, truncation=True, padding="max_length", max_length=MAX_SEQ_LENGTH, return_tensors="pt")
        return {"input_ids": encoded["input_ids"].squeeze(), "attention_mask": encoded["attention_mask"].squeeze(), "labels": encoded["input_ids"].squeeze()}
    
    processed_ds = dataset.map(preprocess, remove_columns=dataset.column_names)
    
    # 4. Train
    args_train = get_training_args(str(OUTPUT_DIR), args.batch_size, args.epochs)
    trainer = Trainer(model=model, args=args_train, train_dataset=processed_ds)
    
    print("\n▶️  Starting training...")
    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    print(f"✅ Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    sys.exit(main())
