#!/bin/zsh
# Quick-start commands for fine-tuning

echo "🚀 LoRA Fine-Tuning Quick Reference"
echo ""
echo "TEST MODE (10-20 minutes, 50 records):"
echo "  cd peft/lora"
echo "  python train_lora.py --model tinyllama/TinyLlama-1.1B-Chat-v1.0 --dataset data/dataset.jsonl --output-dir outputs/tinyllama_lora --test"
echo ""
echo "QUICK TRAINING (45-90 minutes, 500 records):"
echo "  python train_lora.py --model tinyllama/TinyLlama-1.1B-Chat-v1.0 --dataset data/dataset.jsonl --output-dir outputs/tinyllama_lora"
echo ""
echo "FULL TRAINING (12+ hours, 148,778 records):"
echo "  python train_lora.py --model tinyllama/TinyLlama-1.1B-Chat-v1.0 --dataset data/dataset.jsonl --output-dir outputs/tinyllama_lora --train-size 148778"
echo ""
echo "CUSTOM (adjust train-size):"
echo "  python train_lora.py --model tinyllama/TinyLlama-1.1B-Chat-v1.0 --dataset data/dataset.jsonl --output-dir outputs/tinyllama_lora --train-size 1000"
echo ""
echo "📖 See docs/ for more details"
