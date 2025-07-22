#!/usr/bin/env python3
"""
Resume PRIMUS-Enhanced Cybersecurity Training
Continues training from the last saved checkpoint
"""

import os
import json
from pathlib import Path

def find_latest_checkpoint():
    """Find the latest checkpoint to resume from"""
    checkpoint_dir = "/Users/danielrodrigo/Workspace/PyScience/datasets/cybersecurity_finetuned_models/mlx_adapters_primus_ZERO_TRUNCATION_v1"
    
    # Look for numbered checkpoint files
    checkpoint_files = []
    for file in os.listdir(checkpoint_dir):
        if file.startswith("0000") and file.endswith("_adapters.safetensors"):
            iteration = int(file[4:7])  # Extract iteration number
            checkpoint_files.append((iteration, file))
    
    if checkpoint_files:
        latest_iter, latest_file = max(checkpoint_files)
        return latest_iter, os.path.join(checkpoint_dir, latest_file)
    else:
        return 0, None

def create_resume_command():
    """Create the command to resume training"""
    
    latest_iter, checkpoint_path = find_latest_checkpoint()
    
    if checkpoint_path:
        print(f"🔄 Found checkpoint at iteration {latest_iter}")
        print(f"📁 Checkpoint: {checkpoint_path}")
        
        # Calculate remaining iterations
        total_iters = 10000
        remaining_iters = total_iters - latest_iter
        
        print(f"📊 Progress: {latest_iter}/{total_iters} ({latest_iter/total_iters*100:.1f}% complete)")
        print(f"⏳ Remaining: {remaining_iters} iterations")
        
        # Create resume command
        resume_command = f"""cd /Users/danielrodrigo/Workspace/PyScience/datasets && python3 -m mlx_lm lora \\
  --model /Users/danielrodrigo/Workspace/PyScience/datasets/mlx_models/tinyllama_mlx \\
  --data /Users/danielrodrigo/Workspace/PyScience/datasets/primus_training_FINAL \\
  --resume-adapter-file {checkpoint_path} \\
  --train \\
  --batch-size 4 \\
  --iters {total_iters} \\
  --learning-rate 1e-4 \\
  --steps-per-report 50 \\
  --steps-per-eval 200 \\
  --save-every 400 \\
  --max-seq-length 512 \\
  --num-layers 22 \\
  --adapter-path ./cybersecurity_finetuned_models/mlx_adapters_primus_ZERO_TRUNCATION_v1 \\
  --grad-checkpoint \\
  --seed 42"""
        
        return resume_command, latest_iter, remaining_iters
    else:
        print("❌ No checkpoint found - would need to start from beginning")
        return None, 0, 10000

def main():
    print("=== PRIMUS Training Resume Helper ===\\n")
    
    # Check current status
    resume_command, latest_iter, remaining_iters = create_resume_command()
    
    if resume_command:
        print(f"\\n🚀 RESUME COMMAND:")
        print(f"Copy and paste this command to resume training:\\n")
        print(resume_command)
        
        # Save to file for easy access
        with open("/Users/danielrodrigo/Workspace/PyScience/datasets/RESUME_TRAINING.sh", "w") as f:
            f.write("#!/bin/bash\\n")
            f.write("# Resume PRIMUS-Enhanced Cybersecurity Training\\n")
            f.write(f"# Resuming from iteration {latest_iter}\\n")
            f.write(f"# Remaining iterations: {remaining_iters}\\n\\n")
            f.write(resume_command)
        
        print(f"\\n📁 Also saved to: RESUME_TRAINING.sh")
        print(f"\\n💡 To resume later, just run:")
        print(f"bash /Users/danielrodrigo/Workspace/PyScience/datasets/RESUME_TRAINING.sh")
        
    print(f"\\n📈 TRAINING PROGRESS SO FAR:")
    print(f"   - Started with loss: 1.940 (validation)")
    print(f"   - Latest train loss: 1.337 (iteration 900)")  
    print(f"   - Latest val loss: 1.388 (iteration 800)")
    print(f"   - Excellent convergence trend! 📉")
    print(f"   - Total tokens trained: 1,111,104")
    print(f"   - Memory usage: 3.339GB (very efficient)")

if __name__ == "__main__":
    main()
