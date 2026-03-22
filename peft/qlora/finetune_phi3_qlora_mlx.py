#!/usr/bin/env python3
"""
MLX-LM QLoRA Fine-tuning for Phi-3-mini on Heimdall Dataset
Professional version (under 200 LOC) following factory standards.
"""

import json
import argparse
import sys
from pathlib import Path
from peft.runtime.mlx_logic import prepare_mlx_data, train_with_mlx_lm

# ============================================================================
# Configuration
# ============================================================================

DATASET_PATH = "/Volumes/Crucial X9/data-sets/heimdall/processed/heimdall-enhanced-strands-7variants.jsonl"
OUTPUT_BASE = Path("/Volumes/Crucial X9/ai-models")
OUTPUT_DIR = OUTPUT_BASE / "phi3_mini_heimdall_qlora_mlx"
ADAPTERS_DIR = OUTPUT_DIR / "adapters"

MODEL_NAME = "mlx-community/Phi-3-mini-4k-instruct-4bit"
DEFAULT_TRAIN_SIZE = 500
DEFAULT_ITERS = 600
DEFAULT_BATCH_SIZE = 1

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Phi-3-mini with QLoRA using MLX-LM")
    parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--test", action="store_true", help="Quick test mode")
    parser.add_argument("--skip-prepare", action="store_true", help="Skip data preparation")
    
    args = parser.parse_args()
    train_size = 50 if args.test else args.train_size
    iters = 100 if args.test else args.iters
    
    print(f"\n🎯 Training records: {train_size} | Iterations: {iters}")
    
    # 1. Prepare data
    if not args.skip_prepare:
        print("📝 Step 1: Preparing data...\n")
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            subset = [json.loads(line) for line in f][:train_size]
        
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        temp_jsonl = OUTPUT_DIR / "temp_subset.jsonl"
        with open(temp_jsonl, "w", encoding="utf-8") as f:
            for record in subset:
                f.write(json.dumps(record) + "\n")
        
        data_dir = prepare_mlx_data(str(temp_jsonl), str(OUTPUT_DIR / "mlx_data"))
    else:
        data_dir = str(OUTPUT_DIR / "mlx_data")
    
    # 2. Train
    print("✅ Step 2: Starting QLoRA training...\n")
    success = train_with_mlx_lm(
        model=MODEL_NAME,
        data_dir=data_dir,
        adapter_path=str(ADAPTERS_DIR),
        iters=iters,
        batch_size=args.batch_size
    )
    
    if success:
        print(f"\n🎉 Training Completed! Adapters saved to: {ADAPTERS_DIR}")
        return 0
    return 1

if __name__ == "__main__":
    sys.exit(main())
