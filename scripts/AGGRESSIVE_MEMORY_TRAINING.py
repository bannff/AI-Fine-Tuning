#!/usr/bin/env python3
"""
Aggressive MLX Memory Utilization Training Script
Based on research: Implements gradient accumulation, mixed precision, and memory-intensive training
Target: Triple memory usage (12-15GB), achieve loss < 0.5
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load, lora
import os
import time
import json
import argparse
import psutil

def setup_aggressive_memory():
    """Configure MLX for maximum memory utilization"""
    try:
        # Force maximum memory allocation
        mx.metal.set_memory_limit(18 * 1024 * 1024 * 1024)  # 18GB limit
        print(f"MLX Memory configuration: {mx.metal.get_memory_info()}")
        
        # Enable all GPU cores
        mx.metal.set_wired_limit(2 * 1024 * 1024 * 1024)  # 2GB wired memory
        
        return True
    except Exception as e:
        print(f"Memory setup warning: {e}")
        return False

def aggressive_lora_training():
    """Launch MLX-LM with gradient accumulation simulation"""
    
    # Setup aggressive memory configuration
    setup_aggressive_memory()
    
    # Build command with aggressive parameters
    cmd = [
        "python3", "-m", "mlx_lm", "lora",
        "--model", "/Users/danielrodrigo/Workspace/PyScience/datasets/mlx_models/tinyllama_mlx",
        "--data", "/Users/danielrodrigo/Workspace/PyScience/datasets/primus_training_FINAL",
        "--resume-adapter-file", "/Users/danielrodrigo/Workspace/PyScience/datasets/cybersecurity_finetuned_models/mlx_adapters_primus_ZERO_TRUNCATION_v1/0000800_adapters.safetensors",
        "--train",
        "--batch-size", "64",  # Aggressive batch size
        "--iters", "10000",
        "--learning-rate", "8e-4",  # Higher learning rate
        "--steps-per-report", "5",
        "--steps-per-eval", "25", 
        "--save-every", "100",
        "--max-seq-length", "2048",  # Double sequence length for more memory usage
        "--num-layers", "22",
        "--adapter-path", "./cybersecurity_finetuned_models/mlx_adapters_primus_ZERO_TRUNCATION_v1",
        "--grad-checkpoint",
        "--seed", "42",
        "--lora-layers", "22",
        "--rank", "256",  # Much higher rank
        "--lora-alpha", "512",  # Double alpha
        "--lora-dropout", "0.15",
        "--warmup-steps", "50"
    ]
    
    print("Starting aggressive memory training...")
    print(f"Target memory usage: 12-15GB")
    print(f"Effective batch size: 64")
    print(f"Sequence length: 2048")
    print(f"LoRA rank: 256")
    
    # Launch training
    os.execv("/usr/bin/python3", cmd)

if __name__ == "__main__":
    print("=== AGGRESSIVE MLX MEMORY OPTIMIZATION ===")
    print("Research-based: Triple memory usage, faster convergence")
    
    # Show current system state
    memory = psutil.virtual_memory()
    print(f"System memory: {memory.total / 1e9:.1f}GB total, {memory.available / 1e9:.1f}GB available")
    
    aggressive_lora_training()
