#!/usr/bin/env python3
"""
PHI-3 Cybersecurity LoRA Fine-Tuning
MLX-LM optimized for Mac Studio M4 Max
"""
import os
import sys
import json
import subprocess
import argparse
from pathlib import Path

# Project configuration
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT
BASE_MODEL = "/Volumes/Crucial X9/ai-models/Phi-3-mini-128k-instruct-mlx"
DATASET_DIR = "../../../../../datasets/cybersec_data"
ADAPTER_OUTPUT = MODELS_DIR / "adapters"

def validate_paths():
    """Validate all required paths exist"""
    if not os.path.exists(BASE_MODEL):
        raise FileNotFoundError(f"Base model not found: {BASE_MODEL}")
    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(f"Dataset not found: {DATASET_DIR}")
    if not os.path.exists(f"{DATASET_DIR}/train.jsonl"):
        raise FileNotFoundError(f"Training data not found: {DATASET_DIR}/train.jsonl")
    if not os.path.exists(f"{DATASET_DIR}/valid.jsonl"):
        raise FileNotFoundError(f"Validation data not found: {DATASET_DIR}/valid.jsonl")

def create_config_file():
    """Create MLX-LM training configuration"""
    config = {
        "model": BASE_MODEL,
        "data": str(DATASET_DIR),
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
        "fine_tune_type": "lora"
    }
    
    config_path = MODELS_DIR / "lora_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_path

def main():
    parser = argparse.ArgumentParser(description="PHI-3 Cybersecurity LoRA Training")
    parser.add_argument("--test", action="store_true", help="Run test with 20 iterations")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    args = parser.parse_args()
    
    print("🚀 PHI-3 Cybersecurity LoRA Fine-Tuning")
    print("=" * 60)
    
    try:
        validate_paths()
        print("✅ All paths validated")
        
        # Create directories
        ADAPTER_OUTPUT.mkdir(parents=True, exist_ok=True)
        
        # Create config
        config_path = create_config_file()
        print(f"✅ Config created: {config_path}")
        
        # Build command
        cmd = [
            "python3", "-m", "mlx_lm", "lora",
            "--model", BASE_MODEL,
            "--data", str(DATASET_DIR),
            "--train",
            "--batch-size", "2",
            "--iters", "20" if args.test else "1000",
            "--learning-rate", "2e-4",
            "--steps-per-report", "5" if args.test else "10",
            "--steps-per-eval", "10" if args.test else "50",
            "--save-every", "10" if args.test else "100",
            "--max-seq-length", "512",
            "--num-layers", "-1",
            "--adapter-path", str(ADAPTER_OUTPUT) + ("_test" if args.test else ""),
            "--grad-checkpoint",
            "--seed", "42",
            "--val-batches", "10" if args.test else "50",
        ]
        
        if args.resume:
            cmd.extend(["--resume-adapter-file", args.resume])
        
        mode = "TEST" if args.test else "FULL"
        iterations = "20" if args.test else "1000"
        
        print(f"\n🎯 Starting {mode} training ({iterations} iterations)")
        print(f"📊 Dataset: {DATASET_DIR}")
        print(f"🔧 Adapters: {ADAPTER_OUTPUT}")
        print(f"💾 Base model: {BASE_MODEL}")
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
