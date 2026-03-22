#!/usr/bin/env python3
"""
Train TinyLlama with LoRA on enhanced_data/train.jsonl
Fresh run, not resuming from any adapter checkpoint
"""

import subprocess
import sys
import os

def train_tinyllama_lora():
    """Train TinyLlama with LoRA on enhanced_data/train.jsonl"""
    print("\U0001F981 TINYLLAMA LoRA TRAINING (Fresh Run)")
    print("=" * 50)
    print("\U0001F4DA Using /enhanced_data/train.jsonl as training data")
    print("")
    
    # Paths
    model_path = "/Volumes/Crucial X9/ai-models/tinyllama_mlx"
    dataset_path = "../../../../datasets/enhanced_data/train.jsonl"
    adapter_path = "./tinyllama_lora_adapter"
    
    # Verify dataset exists
    if not os.path.exists(dataset_path):
        print(f"\u274c Training dataset not found: {dataset_path}")
        return False
    
    # Training parameters - correct MLX-LM syntax
    cmd = [
        "python3", "-m", "mlx_lm", "lora",
        "--model", model_path,
        "--train",
        "--data", dataset_path,
        "--adapter-path", adapter_path,
        "--batch-size", "16",
        "--iters", "1000",
        "--learning-rate", "2e-5",
        # LoRA alpha/scale not set via CLI in current MLX-LM
        "--steps-per-report", "20",
        "--save-every", "200",
        "--max-seq-length", "1024",
        "--seed", "42"
    ]
    
    print("\U0001F680 Starting TinyLlama LoRA training...")
    print(f"\U0001F4C8 Command: {' '.join(cmd)}")
    print("")
    print("\U0001F4CA Training Progress:")
    print("-" * 50)
    
    try:
        # Run training
        result = subprocess.run(cmd, check=True, cwd=str(Path(__file__).parent))
        
        print("\n" + "=" * 50)
        print("\u2705 TINYLLAMA LoRA TRAINING COMPLETED SUCCESSFULLY!")
        print(f"\U0001F4BE LoRA adapters saved to: {adapter_path}")
        print("")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n\u274c Training failed with error: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\u23f9\ufe0f  Training interrupted by user")
        return False
    except Exception as e:
        print(f"\n\u274c Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = train_tinyllama_lora()
    if success:
        print("\n\U0001F389 Ready to test TinyLlama LoRA model!")
        print("\U0001F4AC Run: python3 tinyllama_chat.py")
    else:
        print("\n\U0001F4A5 Training failed - check logs above")
        sys.exit(1)
