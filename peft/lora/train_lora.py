#!/usr/bin/env python3
"""
LoRA Fine-tuning using standard HuggingFace tools.
Professional version (under 200 LOC) following factory standards.
"""

import argparse
import sys
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer

from peft.runtime.training_logic import apply_lora, get_training_args, load_local_jsonl


def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-tune a causal language model")
    
    # Required arguments
    parser.add_argument("--model", type=str, required=True, help="HF model ID or path to local model")
    parser.add_argument("--dataset", type=str, required=True, help="Path to JSONL dataset")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the trained adapters")
    
    # Optional hyperparams
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--train-size", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--test", action="store_true", help="Run with a tiny dataset for testing")

    args = parser.parse_args()
    
    train_size = 50 if args.test else args.train_size
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🚀 Starting LoRA training on {args.model}")
    print(f"📂 Dataset: {args.dataset}")
    print(f"💾 Output: {output_dir}")

    # 1. Load Data
    dataset = load_local_jsonl(args.dataset)
    if train_size < len(dataset):
        dataset = dataset.select(range(train_size))

    # 2. Setup Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype="auto", trust_remote_code=True
    )
    model = apply_lora(model, args.lora_rank)

    # 3. Preprocess
    def preprocess(example):
        messages = example.get("messages", [])
        # Simple format, similar to original Heimdall script
        full_text = "".join([f"<|{m.get('role', 'user')}|>\n{m.get('content', '')}\n" for m in messages])
        encoded = tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=args.max_seq_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "labels": encoded["input_ids"].squeeze(),
        }

    processed_ds = dataset.map(preprocess, remove_columns=dataset.column_names)

    # 4. Train
    args_train = get_training_args(str(output_dir), args.batch_size, args.epochs)
    trainer = Trainer(model=model, args=args_train, train_dataset=processed_ds)

    print("\n▶️  Starting training...")
    trainer.train()
    trainer.save_model(str(output_dir))
    print(f"✅ Model saved to {output_dir}")


if __name__ == "__main__":
    sys.exit(main())
