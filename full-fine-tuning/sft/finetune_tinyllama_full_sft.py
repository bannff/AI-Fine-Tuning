#!/usr/bin/env python3
"""
Full Supervised Fine-Tuning (SFT) for TinyLlama-1.1B on Heimdall dataset using MLX-LM.
Uses full parameter fine-tuning (not LoRA/QLoRA).

Models: TinyLlama-1.1B-Chat-v1.0 (regular bfloat16, ~2.0 GB)
Output: /Volumes/Crucial X9/ai-models/tinyllama_heimdall_full_sft/
"""

import subprocess
import sys
from pathlib import Path

# Configuration
MODEL = "/Volumes/Crucial X9/ai-models/tinyllama_mlx"  # Local full-precision model
DATA_DIR = "/Volumes/Crucial X9/ai-models/heimdall_sft_data"
OUTPUT_DIR = "/Volumes/Crucial X9/ai-models/tinyllama_heimdall_full_sft"

# Training hyperparameters
BATCH_SIZE = 4
LEARNING_RATE = 2e-6  # Lower learning rate for full fine-tuning to avoid NaN
ITERS = 600  # Full training iterations
EVAL_STEPS = 50
SAVE_EVERY = 100
MAX_SEQ_LENGTH = 2048


def main():
    # Ensure data directory exists
    data_path = Path(DATA_DIR)
    if not data_path.exists():
        print(f"Error: Data directory not found: {DATA_DIR}")
        print("Please run the data preparation script first.")
        sys.exit(1)

    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    # Build MLX-LM lora command for full fine-tuning
    cmd = [
        "mlx_lm.lora",
        "--model",
        MODEL,
        "--train",
        "--data",
        DATA_DIR,
        "--fine-tune-type",
        "full",  # Full parameter fine-tuning, not LoRA
        "--adapter-path",
        OUTPUT_DIR,
        "--batch-size",
        str(BATCH_SIZE),
        "--iters",
        str(ITERS),
        "--learning-rate",
        str(LEARNING_RATE),
        "--optimizer",
        "adamw",
        "--steps-per-report",
        str(EVAL_STEPS),
        "--steps-per-eval",
        str(EVAL_STEPS),
        "--save-every",
        str(SAVE_EVERY),
        "--max-seq-length",
        str(MAX_SEQ_LENGTH),
        "--grad-checkpoint",  # Use gradient checkpointing to save memory
    ]

    # Check for test mode
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("Running in TEST mode (50 iterations)...")
        # For test mode, just reduce iterations
        cmd[cmd.index("--iters") + 1] = "50"
    else:
        print("Starting full SFT fine-tuning on TinyLlama-1.1B")
        print(f"Model: {MODEL}")
        print(f"Data: {DATA_DIR}")
        print(f"Output: {OUTPUT_DIR}")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"Learning rate: {LEARNING_RATE}")
        print(f"Total iterations: {ITERS}")
        print(f"Max sequence length: {MAX_SEQ_LENGTH}")
        print(
            "\nNote: Full fine-tuning uses ~8 GB memory. Gradient checkpointing enabled."
        )

    # Run training
    print("\nExecuting command:")
    print(" ".join(cmd))
    print()

    result = subprocess.run(cmd, cwd=str(Path.home()))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
