#!/usr/bin/env python3
"""
MLX-LM QLoRA Fine-tuning using a polymorphic script.

This module provides a streamlined, memory-efficient QLoRA fine-tuning
pipeline optimized for Apple Silicon via the MLX framework.

Usage:
    ./train_qlora_mlx.py --model <model> --dataset <dataset.jsonl> --output-dir <dir>
"""

import argparse
import json
import sys
from pathlib import Path

from peft.runtime.mlx_logic import prepare_mlx_data, train_with_mlx_lm


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model with QLoRA using MLX-LM")
    
    # Required arguments
    parser.add_argument("--model", type=str, required=True, help="MLX format model identifier or path")
    parser.add_argument("--dataset", type=str, required=True, help="Path to JSONL dataset")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the adapters")
    
    # Hyperparams
    parser.add_argument("--train-size", type=int, default=500)
    parser.add_argument("--iters", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=1)
    
    # Utilities
    parser.add_argument("--test", action="store_true", help="Quick test mode (50 rows, 100 iters)")
    parser.add_argument("--skip-prepare", action="store_true", help="Skip data preparation")

    args = parser.parse_args()
    
    train_size = 50 if args.test else args.train_size
    iters = 100 if args.test else args.iters
    output_dir = Path(args.output_dir)
    adapters_dir = output_dir / "adapters"
    mlx_data_dir = output_dir / "mlx_data"

    print(f"\n🎯 Training records: {train_size} | Iterations: {iters}")
    print(f"🤖 Model: {args.model}")

    # 1. Prepare data
    if not args.skip_prepare:
        print("📝 Step 1: Preparing data...\n")
        with open(args.dataset, "r", encoding="utf-8") as f:
            subset = [json.loads(line) for line in f][:train_size]

        output_dir.mkdir(parents=True, exist_ok=True)
        temp_jsonl = output_dir / "temp_subset.jsonl"
        with open(temp_jsonl, "w", encoding="utf-8") as f:
            for record in subset:
                f.write(json.dumps(record) + "\n")

        # split train/val/test using logic layer
        data_dir = prepare_mlx_data(str(temp_jsonl), str(mlx_data_dir))
    else:
        data_dir = str(mlx_data_dir)

    # 2. Train
    print("✅ Step 2: Starting QLoRA training...\n")
    success = train_with_mlx_lm(
        model=args.model,
        data_dir=data_dir,
        adapter_path=str(adapters_dir),
        iters=iters,
        batch_size=args.batch_size,
    )

    if success:
        print(f"\n🎉 Training Completed! Adapters saved to: {adapters_dir}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
