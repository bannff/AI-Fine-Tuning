#!/usr/bin/env bash
"""
Cleanup script for MLX-LM training checkpoints
This will keep the final adapters and delete intermediate checkpoints to save space
"""

echo "🧹 Cleaning up training checkpoints..."

CHECKPOINT_DIR="/Users/danielrodrigo/Workspace/PyScience/datasets/cybersecurity_finetuned_models/mlx_adapters_v3"

# Keep these important files:
# - adapters.safetensors (final trained model)
# - adapter_config.json (configuration)
# - 0001000_adapters.safetensors (latest checkpoint as backup)

echo "📁 Current directory contents:"
ls -lh "$CHECKPOINT_DIR"

echo ""
echo "🗑️  Files to delete (intermediate checkpoints):"
ls -lh "$CHECKPOINT_DIR"/00001*_adapters.safetensors 2>/dev/null || echo "No 00001* files"
ls -lh "$CHECKPOINT_DIR"/00002*_adapters.safetensors 2>/dev/null || echo "No 00002* files"
ls -lh "$CHECKPOINT_DIR"/00003*_adapters.safetensors 2>/dev/null || echo "No 00003* files"
ls -lh "$CHECKPOINT_DIR"/00004*_adapters.safetensors 2>/dev/null || echo "No 00004* files"
ls -lh "$CHECKPOINT_DIR"/00005*_adapters.safetensors 2>/dev/null || echo "No 00005* files"
ls -lh "$CHECKPOINT_DIR"/00006*_adapters.safetensors 2>/dev/null || echo "No 00006* files"
ls -lh "$CHECKPOINT_DIR"/00007*_adapters.safetensors 2>/dev/null || echo "No 00007* files"
ls -lh "$CHECKPOINT_DIR"/00008*_adapters.safetensors 2>/dev/null || echo "No 00008* files"
ls -lh "$CHECKPOINT_DIR"/00009*_adapters.safetensors 2>/dev/null || echo "No 00009* files"

echo ""
echo "💾 Files to KEEP:"
echo "- adapters.safetensors (final model)"
echo "- adapter_config.json (config)"
echo "- 0001000_adapters.safetensors (latest checkpoint backup)"

echo ""
read -p "❓ Delete intermediate checkpoints? This will save ~56MB. (y/N): " confirm

if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
    echo "🗑️  Deleting intermediate checkpoints..."
    
    # Delete checkpoints 100-900 but keep 1000
    rm -f "$CHECKPOINT_DIR"/0000[1-9]00_adapters.safetensors
    
    echo "✅ Cleanup complete!"
    echo ""
    echo "📊 Space saved:"
    du -sh "$CHECKPOINT_DIR"
    echo ""
    echo "📁 Remaining files:"
    ls -lh "$CHECKPOINT_DIR"
else
    echo "❌ Cleanup cancelled. All checkpoints preserved."
fi
