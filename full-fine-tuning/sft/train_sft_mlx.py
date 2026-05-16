#!/usr/bin/env python3
"""
Full Supervised Fine-Tuning (SFT) using MLX-LM.
Uses full parameter fine-tuning via `--fine-tune-type full`.

Usage:
    ./train_sft_mlx.py --model <model> --data-dir <dir> --output-dir <dir>
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Full SFT Fine-Tuning with MLX-LM")
    
    # Required arguments
    parser.add_argument("--model", type=str, required=True, help="Local full-precision model path")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to prepared MLX data directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the trained model")
    
    # Hyperparameters
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--iters", type=int, default=600)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    
    # Flags
    parser.add_argument("--test", action="store_true", help="Run a quick test (50 iterations)")

    args = parser.parse_args()

    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"❌ Error: Data directory not found: {args.data_dir}")
        print("Please run the data preparation script first.")
        sys.exit(1)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    iters = 50 if args.test else args.iters

    print(f"🚀 Starting full SFT fine-tuning on {args.model}")
    print(f"📂 Data: {args.data_dir}")
    print(f"💾 Output: {args.output_dir}")
    print(f"⚙️  Batch size: {args.batch_size} | LR: {args.learning_rate} | Iters: {iters}")
    print("\nNote: Full fine-tuning requires significant memory. Gradient checkpointing enabled.")

    cmd = [
        "mlx_lm.lora",
        "--model", args.model,
        "--train",
        "--data", args.data_dir,
        "--fine-tune-type", "full",  # KEY: Full parameter fine-tuning, not LoRA
        "--adapter-path", args.output_dir,
        "--batch-size", str(args.batch_size),
        "--iters", str(iters),
        "--learning-rate", str(args.learning_rate),
        "--optimizer", "adamw",
        "--steps-per-report", str(args.eval_steps),
        "--steps-per-eval", str(args.eval_steps),
        "--save-every", str(args.save_every),
        "--max-seq-length", str(args.max_seq_length),
        "--grad-checkpoint",  # Use gradient checkpointing to save memory
    ]

    print("\nExecuting command:")
    print(" ".join(cmd))
    print()

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {e}")
        return False
    except FileNotFoundError:
        print("❌ mlx_lm not found. Install: pip install mlx-lm")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
