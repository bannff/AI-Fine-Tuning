# 🧩 Agentic Dataset Pipeline for MLX-LM (Cybersecurity)

This workspace provides a robust, agentic, and truth-promoting dataset pipeline for MLX-LM fine-tuning. The pipeline ensures:
- All data is in the recommended `messages` array format (chat-style, user/assistant turns)
- Deduplication, English-only filtering, and removal of invalid/bad values
- Automatic splitting into `train.jsonl` and `valid.jsonl` for MLX-LM

## How to Use the Pipeline

1. Place your raw cybersecurity Q&A or chat-style JSONL file in the desired project folder.
2. Run the combined script to convert, dedupe, validate, and split your data:

```bash
pip3 install langdetect
python3 scripts/convert_and_validate_agentic_dataset.py <input.jsonl> [valid_frac]
```

- By default, this will overwrite `train.jsonl` and `valid.jsonl` in the same folder as your input file.
- The script expects each line in the input to have either `user`/`assistant` or `question`/`answer` fields.
- The output is MLX-LM compatible and ready for fine-tuning.

### Example

```bash
python3 scripts/convert_and_validate_agentic_dataset.py datasets/my_raw_cybersec.jsonl 0.1
# Produces: datasets/train.jsonl and datasets/valid.jsonl
```

### Output Format (for both train and valid)
```jsonl
{"messages": [
    {"role": "user", "content": "What is the latest CVE for OpenSSL?"},
    {"role": "assistant", "content": "I'm not sure of the latest CVE for OpenSSL. Let me research that for you and get back with an accurate answer."}
]}
```

**This ensures your LLM is trained to be both cybersecurity-savvy and agentic/truthful.**
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


## 📁 Directory Structure (2025, Modular MLX-LM Projects)

```
PyScience/
├── mamba/                  # Mamba Cybersecurity Project (self-contained)
│   ├── configs/            # Config files (YAML, JSON)
│   ├── datasets/           # Project-specific dataset scripts
│   ├── models/             # Model weights/checkpoints
│   ├── scripts/            # Training, chat, and utility scripts
│   └── validation/         # Validation and analysis scripts
│   └── README.md           # Project-specific readme
├── falcon_mamba/           # Falcon-Mamba Project (self-contained)
│   ├── configs/
│   ├── data/
│   ├── models/
│   ├── scripts/
│   ├── docs/
│   └── ...
├── datasets/               # Shared and legacy datasets (JSONL, tar.gz, zip)
│   ├── primus_training/    # Main training/validation sets
│   ├── mlx_models/         # Base models (e.g., TinyLlama MLX)
│   ├── pattern_identification_examples.jsonl
│   ├── comprehensive_cybersec_dataset.jsonl
│   ├── enhanced_cybersec_dataset.jsonl
│   └── ...
├── scripts/                # Central utility scripts (cleaning, conversion, etc.)
├── configs/                # Central configs (shared or legacy)
├── docs/                   # Documentation and guides
├── archive/                # Archived scripts, configs, datasets, and research
│   ├── data/
│   ├── configs/
│   ├── docs/
│   ├── datasets_archive/   # Former datasets/archive/ content
│   └── ...
├── environments/           # Python environments
│   └── hf-llm-env/
└── README.md               # Project overview (this file)
```

**Key Points:**
- Each major project (e.g., `mamba`, `falcon_mamba`) is self-contained and follows MLX-LM best practices.
- Shared datasets, scripts, and configs are at the top level for easy access.
- All legacy, research, and unused content is preserved in `/archive` for future reference.
- Documentation is centralized in `/docs/` and per-project as needed.


## 🗃️ Datasets Available

### Main Datasets
- **comprehensive_cybersec_dataset.jsonl** — Large, cleaned cybersecurity dataset
- **enhanced_cybersec_dataset.jsonl** — Enhanced, instruction-tuned dataset
- **pattern_identification_examples.jsonl** — Pattern identification examples
- **primus_training/train.jsonl** — Main training set
- **primus_training/valid.jsonl** — Main validation set
- **primus_training/train_enhanced.jsonl** — Enhanced training set
- **primus_training/valid_enhanced.jsonl** — Enhanced validation set

### Model Files
- **mlx_models/tinyllama_mlx/** — TinyLlama model files (MLX format)

### Format
- All datasets are MLX-LM compatible JSONL with instruction-response pairs or task-specific fields.

## 🤖 Trained Models


### Example: TinyLlama Cybersecurity Model
- **Base Model**: TinyLlama-1.1B-Chat-v1.0 (MLX format)
- **Training Method**: LoRA (Low-Rank Adaptation)
- **Specialization**: Cybersecurity questions and analysis
- **Adapters**: See project or `/archive` for LoRA adapters


### Training Configuration (Example)
- **Batch Size**: 1 (memory optimized)
- **Max Sequence Length**: 256
- **LoRA Rank**: 16
- **Training Steps**: 500
- **Validation**: Every 50 steps


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

To retrain or continue training (example for TinyLlama):
```bash
cd PyScience
source environments/hf-llm-env/bin/activate

mlx_lm.lora \
  --model datasets/mlx_models/tinyllama_mlx \
  --train \
  --data datasets/primus_training \
  --iters 500 \
  --save-every 50 \
  --adapter-path archive/data/mlx_adapters_v3 \
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
6. **Explore Archives**: Review `/archive` for legacy scripts, configs, and research data

---

*This workspace is optimized for Apple Silicon (M1/M2/M3) using MLX-LM for efficient training and inference.*
