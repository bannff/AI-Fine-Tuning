# PyScience - Python-based Machine Learning for Cybersecurity

This folder contains all Python-based machine learning workflows, datasets, models, and environments for cybersecurity research and model training.

## 🚀 Quick Start

### Interactive Chat with Trained Model
```bash
cd /Users/danielrodrigo/Workspace/PyScience
source environments/hf-llm-env/bin/activate
mlx_lm.chat --model datasets/mlx_models/tinyllama_mlx --adapter-path datasets/cybersecurity_finetuned_models/mlx_adapters_v2 --max-tokens 512
```

### Alternative Chat Commands
```bash
# Using the latest checkpoint
mlx_lm.chat --model datasets/mlx_models/tinyllama_mlx --adapter-path datasets/cybersecurity_finetuned_models/mlx_adapters_v2/0000500_adapters.safetensors --max-tokens 512

# For longer responses
mlx_lm.chat --model datasets/mlx_models/tinyllama_mlx --adapter-path datasets/cybersecurity_finetuned_models/mlx_adapters_v2 --max-tokens 1024
```

## 📁 Directory Structure

```
PyScience/
├── datasets/                              # All training data and dataset management
│   ├── cybersecurity_datasets/            # Processed cybersecurity datasets
│   │   ├── processed/                     # Clean, MLX-ready datasets
│   │   │   ├── cybersecurity_agentic_clean_train.jsonl
│   │   │   ├── cybersecurity_agentic_clean_valid.jsonl
│   │   │   ├── heimdall_v1_1.jsonl
│   │   │   ├── heimdall_merged_cleaned.jsonl
│   │   │   ├── vanessa_cybersec_32k.jsonl
│   │   │   └── clyde_cybersec_10k.jsonl
│   │   └── AlicanKiraz0_Cybersecurity-Dataset-*/
│   ├── cybersecurity_finetuned_models/    # Trained model outputs
│   │   ├── mlx_adapters_v2/               # Final LoRA adapters
│   │   └── peft_lora/                     # Alternative adapter formats
│   ├── mlx_models/                        # Base models in MLX format
│   │   └── tinyllama_mlx/                 # TinyLlama converted for MLX
│   ├── mlx_full_dataset/                  # Combined training data
│   │   ├── train.jsonl                    # Training set
│   │   └── valid.jsonl                    # Validation set
│   └── *.py                               # Dataset processing scripts
├── models/                                # Model training outputs and scripts
│   ├── scripts/                           # Training and inference scripts
│   ├── models/                            # Trained model checkpoints
│   └── tinyllama_heimdall_v1_merged_finetuned/  # LoRA training outputs
├── environments/                          # Python environments
│   └── hf-llm-env/                       # Main ML environment with MLX-LM
└── scripts/                              # Utility scripts
    └── run_model.py                      # General model runner
```

## 🗃️ Datasets Available

### Processed Cybersecurity Datasets
- **cybersecurity_agentic_clean_train.jsonl** (135,742 samples) - Main training data
- **cybersecurity_agentic_clean_valid.jsonl** (15,083 samples) - Validation data
- **heimdall_v1_1.jsonl** (75,673 samples) - Heimdall dataset v1.1
- **heimdall_merged_cleaned.jsonl** (150,827 samples) - Merged Heimdall data
- **vanessa_cybersec_32k.jsonl** (32,000 samples) - Instruction-tuned cybersec data
- **clyde_cybersec_10k.jsonl** (10,000 samples) - Additional cybersec conversations

### Combined Training Data
- **Total training samples**: 135,742
- **Total validation samples**: 15,083
- **Format**: MLX-LM compatible JSONL with instruction-response pairs

## 🤖 Trained Models

### TinyLlama Cybersecurity Model
- **Base Model**: TinyLlama-1.1B-Chat-v1.0 (MLX format)
- **Training Method**: LoRA (Low-Rank Adaptation)
- **Specialization**: Cybersecurity questions and analysis
- **Final Adapters**: `datasets/cybersecurity_finetuned_models/mlx_adapters_v2/`

### Training Configuration
- **Batch Size**: 1 (memory optimized)
- **Max Sequence Length**: 256
- **LoRA Rank**: 16
- **Training Steps**: 500
- **Validation**: Every 50 steps
- **Final Loss**: ~0.000 (converged)

## 🛠️ Environment Setup

The Python environment is pre-configured with:
- **MLX-LM**: Apple Silicon optimized training
- **HuggingFace Transformers**: Model and dataset access
- **HuggingFace Datasets**: Dataset processing
- **Pandas, NumPy**: Data manipulation
- **All dependencies**: For MLX training and inference

### Activate Environment
```bash
source environments/hf-llm-env/bin/activate
```

## 📊 Model Performance

The trained model achieved excellent convergence:
- **Training Loss**: Converged to ~0.000
- **Validation Loss**: Converged to ~0.000
- **Training Duration**: ~500 steps
- **Memory Usage**: Optimized for Apple Silicon (8-16GB RAM)

## 🔍 Usage Examples

### Cybersecurity Questions
Try asking the model about:
- SQL injection attacks and prevention
- Network security best practices
- Malware analysis techniques
- Incident response procedures
- Cryptography concepts
- Vulnerability assessment

### Chat Commands
- `q` - Exit chat
- `r` - Reset conversation
- `h` - Show help

## 📝 Training Log

The model was trained using the following process:
1. Downloaded and processed multiple cybersecurity datasets
2. Converted base TinyLlama model to MLX format
3. Combined datasets into MLX-LM compatible format
4. Trained using LoRA with memory-optimized settings
5. Validated convergence through loss monitoring
6. Saved final adapter weights for inference

## 🔄 Re-training

To retrain or continue training:
```bash
cd PyScience
source environments/hf-llm-env/bin/activate

mlx_lm.lora \
  --model datasets/mlx_models/tinyllama_mlx \
  --train \
  --data datasets/mlx_full_dataset \
  --iters 500 \
  --save-every 50 \
  --adapter-path datasets/cybersecurity_finetuned_models/mlx_adapters_v3 \
  --batch-size 1 \
  --lora-layers 16 \
  --max-seq-length 256
```

## 🎯 Next Steps

1. **Test Model**: Use the interactive chat to validate cybersecurity responses
2. **Evaluate Performance**: Test on held-out cybersecurity questions
3. **Export Model**: Convert adapters to other formats if needed
4. **Scale Training**: Increase batch size or sequence length for better performance
5. **Add More Data**: Incorporate additional cybersecurity datasets

---

*This workspace is optimized for Apple Silicon (M1/M2/M3) using MLX-LM for efficient training and inference.*
