#!/usr/bin/env python3
"""
PHI-3 Cybersecurity LoRA Fine-Tuning
MLX-LM optimized for Mac Studio M4 Max
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Project configuration
PROJECT_ROOT = Path(__file__).parent
ADAPTER_OUTPUT = PROJECT_ROOT / "adapters"


def validate_paths(model_path, data_path):
    """Validate all required paths exist"""
    # Note: model_path could be a HF ID, so we only check if it's a local path
    if os.path.exists(model_path):
        print(f"✅ Local model found: {model_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    if not os.path.exists(f"{data_path}/train.jsonl"):
        raise FileNotFoundError(f"Training data not found: {data_path}/train.jsonl")
    if not os.path.exists(f"{data_path}/valid.jsonl"):
        raise FileNotFoundError(f"Validation data not found: {data_path}/valid.jsonl")


def create_config_file(model_path, data_path):
    """Create MLX-LM training configuration"""
    config = {
        "model": model_path,
        "data": str(data_path),
        "train": True,
        "batch_size": 2,
        "iters": 1000,
        "learning_rate": 2e-4,
        "steps_per_report": 10,
        "steps_per_eval": 50,
        "save_every": 100,
        "max_seq_length": 512,
        "num_layers": -1,
        "adapter_path": str(ADAPTER_OUTPUT),
        "grad_checkpoint": True,
        "seed": 42,
        "val_batches": 50,
        "fine_tune_type": "lora",
    }

    config_path = PROJECT_ROOT / "lora_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return config_path


def main():
    parser = argparse.ArgumentParser(description="PHI-3 Cybersecurity LoRA Training")
    parser.add_argument("--model", type=str, default="microsoft/Phi-3-mini-128k-instruct", help="Base model identifier or path")
    parser.add_argument("--data", type=str, default="data", help="Path to prepared MLX data directory")
    parser.add_argument(
        "--test", action="store_true", help="Run test with 20 iterations"
    )
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    args = parser.parse_args()

    print("🚀 PHI-3 Cybersecurity LoRA Fine-Tuning")
    print("=" * 60)

    try:
        validate_paths(args.model, args.data)
        print("✅ All paths validated")

        # Create directories
        ADAPTER_OUTPUT.mkdir(parents=True, exist_ok=True)

        # Create config
        config_path = create_config_file(args.model, args.data)
        print(f"✅ Config created: {config_path}")

        # Build command
        cmd = [
            "python3",
            "-m",
            "mlx_lm",
            "lora",
            "--model",
            args.model,
            "--data",
            str(args.data),
            "--train",
            "--batch-size",
            "2",
            "--iters",
            "20" if args.test else "1000",
            "--learning-rate",
            "2e-4",
            "--steps-per-report",
            "5" if args.test else "10",
            "--steps-per-eval",
            "10" if args.test else "50",
            "--save-every",
            "10" if args.test else "100",
            "--max-seq-length",
            "512",
            "--num-layers",
            "-1",
            "--adapter-path",
            str(ADAPTER_OUTPUT) + ("_test" if args.test else ""),
            "--grad-checkpoint",
            "--seed",
            "42",
            "--val-batches",
            "10" if args.test else "50",
        ]

        if args.resume:
            cmd.extend(["--resume-adapter-file", args.resume])

        mode = "TEST" if args.test else "FULL"
        iterations = "20" if args.test else "1000"

        print(f"\n🎯 Starting {mode} training ({iterations} iterations)")
        print(f"📊 Dataset: {args.data}")
        print(f"🔧 Adapters: {ADAPTER_OUTPUT}")
        print(f"💾 Base model: {args.model}")
        print("\n" + "=" * 60)

        # Execute training
        subprocess.run(cmd, check=True)

        print("\n" + "=" * 60)
        print("🎉 Training completed successfully!")

        if not args.test:
            print(f"\n📂 LoRA adapters saved to: {ADAPTER_OUTPUT}")
            print("🔥 Your cybersecurity-specialized PHI-3 model is ready!")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
