#!/bin/bash

# PyScience Quick Setup and Chat Script
# Usage: ./chat_with_model.sh

echo "🚀 Starting PyScience Cybersecurity Model Chat"
echo "=============================================="

# Navigate to PyScience directory
cd "$(dirname "$0")"

# Activate Python environment
echo "📦 Activating Python environment..."
source environments/hf-llm-env/bin/activate

# Check if model and adapters exist
if [ ! -d "datasets/mlx_models/tinyllama_mlx" ]; then
    echo "❌ Error: Base model not found at datasets/mlx_models/tinyllama_mlx"
    exit 1
fi

if [ ! -d "datasets/cybersecurity_finetuned_models/mlx_adapters_v3" ]; then
    echo "❌ Error: Trained adapters not found at datasets/cybersecurity_finetuned_models/mlx_adapters_v3"
    exit 1
fi

echo "✅ Model and adapters found"
echo "🤖 Starting interactive chat with trained cybersecurity model..."
echo ""
echo "Chat Commands:"
echo "  'q' - Exit chat"
echo "  'r' - Reset conversation"
echo "  'h' - Show help"
echo ""
echo "Try asking about:"
echo "  - SQL injection attacks"
echo "  - Network security"
echo "  - Malware analysis"
echo "  - Incident response"
echo "  - Vulnerability assessment"
echo ""
echo "Starting chat in 3 seconds..."
sleep 3

# Start the chat
python3 -m mlx_lm chat \
  --model datasets/mlx_models/tinyllama_mlx \
  --adapter-path datasets/cybersecurity_finetuned_models/mlx_adapters_v3 \
  --max-tokens 512

echo ""
echo "🎯 Chat session ended. Thanks for using PyScience!"
