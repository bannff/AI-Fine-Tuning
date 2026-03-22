#!/usr/bin/env python3
"""MLX-LM SFT Fine-tuning for TinyLlama on Heimdall Dataset"""

import json
import argparse
import subprocess
from pathlib import Path
import sys

DATASET_PATH = "/Volumes/Crucial X9/data-sets/heimdall/processed/heimdall-enhanced-strands-7variants.jsonl"
OUTPUT_BASE = Path("/Volumes/Crucial X9/ai-models")
OUTPUT_DIR = OUTPUT_BASE / "tinyllama_heimdall_sft_mlx"
ADAPTERS_DIR = OUTPUT_DIR / "adapters"

MODEL_NAME = "mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit"

DEFAULT_TRAIN_SIZE = 500
DEFAULT_ITERS = 600
DEFAULT_BATCH_SIZE = 1

def prepare_mlx_data(input_jsonl: str, output_dir: str, train_size: int):
    """Convert Heimdall JSONL to MLX-LM format"""
    with open(input_jsonl, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    
    subset = data[:train_size]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_count = int(len(subset) * 0.8)
    valid_count = int(len(subset) * 0.1)
    
    train_data = subset[:train_count]
    valid_data = subset[train_count:train_count + valid_count]
    test_data = subset[train_count + valid_count:]
    
    for filename, records in [
        ("train.jsonl", train_data),
        ("valid.jsonl", valid_data),
        ("test.jsonl", test_data)
    ]:
        filepath = output_path / filename
        with open(filepath, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
    
    return str(output_path)

def train_sft(model: str, data_dir: str, adapter_path: str, iters: int, batch_size: int):
    """Train with MLX-LM LoRA"""
    cmd = [
        "mlx_lm.lora",
        "--model", model,
        "--train",
        "--data", data_dir,
        "--adapter-path", adapter_path,
        "--batch-size", str(batch_size),
        "--iters", str(iters),
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {e}")
        return False
    except FileNotFoundError:
        print("❌ mlx_lm not found. Install: pip install mlx-lm")
        return False

def main():
    parser = argparse.ArgumentParser(description="TinyLlama SFT with MLX-LM")
    parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    
    args = parser.parse_args()
    
    train_size = 50 if args.test else args.train_size
    iters = 100 if args.test else args.iters
    
    if not args.skip_prepare:
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            all_data = [json.loads(line) for line in f]
        
        subset = all_data[:train_size]
        temp_jsonl = OUTPUT_DIR / "temp_subset.jsonl"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        with open(temp_jsonl, "w", encoding="utf-8") as f:
            for record in subset:
                f.write(json.dumps(record) + "\n")
        
        data_dir = prepare_mlx_data(str(temp_jsonl), str(OUTPUT_DIR / "mlx_data"), train_size)
    else:
        data_dir = str(OUTPUT_DIR / "mlx_data")
    
    success = train_sft(
        model=MODEL_NAME,
        data_dir=data_dir,
        adapter_path=str(ADAPTERS_DIR),
        iters=iters,
        batch_size=args.batch_size
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
