# Unified Cybersecurity ML Pipeline

A comprehensive, automated pipeline for fine-tuning cybersecurity models with built-in quality control, error correction, and evaluation. **Updated with PRIMUS-Seed integration and zero-truncation token management.**

## 🎯 **Latest Enhancement: PRIMUS-Seed Integration (July 2025)**

Successfully integrated **PRIMUS-Seed** dataset (170K+ samples) with existing cybersecurity datasets for a total of **840,658 training samples** with zero data loss through intelligent token management.

### Key Features:
- ✅ **Zero Truncation Training:** All samples ≤ 480 tokens using actual TinyLlama tokenizer
- ✅ **PRIMUS-Seed Integration:** 3 configurations (default, companies, wikis) 
- ✅ **LoRA α=128:** Maximum adapter influence for strong cybersecurity specialization
- ✅ **Token-Perfect Pipeline:** No "[WARNING] sequences longer than 512" messages
- ✅ **Massive Scale:** 840K+ samples from multiple premium cybersecurity sources

## 📁 Directory Structure

```
datasets/
├── data_collection/          # Dataset research and downloading
│   ├── unified_dataset_collector.py  # Main collection script
│   ├── download_*.py         # Legacy download scripts
│   └── research_*.py         # Legacy research scripts
├── data_processing/          # Data cleaning and transformation
│   ├── unified_data_processor.py     # Main processing script
│   ├── clean_*.py           # Legacy cleaning scripts
│   └── convert_*.py         # Legacy conversion scripts
├── training/                 # Model training and fine-tuning
│   ├── unified_trainer.py    # Main training script
│   ├── real_cybersecurity_finetuner.py  # Current training run
│   └── *trainer*.py         # Legacy training scripts
├── evaluation/              # Model evaluation and testing
│   ├── unified_evaluator.py  # Main evaluation script
│   ├── test_*.py            # Test scripts
│   └── run_model.py         # Model inference
├── utils/                   # Utilities and helpers
│   ├── unified_utils.py     # Main utilities script
│   └── quantize_model.py    # Model quantization
├── cybersecurity_datasets/  # Downloaded raw datasets
├── processed/               # Processed datasets ready for training
└── trained_models/          # Trained model outputs
```

## 🚀 Quick Start

### 1. Research and Download Datasets

```bash
# Research available cybersecurity datasets
python collect_datasets.py research

# Download a specific dataset
python collect_datasets.py download --dataset "AlicanKiraz0/Cybersecurity-Dataset-Heimdall-v1.1"
```

### 2. Process Data for Training

```bash
# Process downloaded dataset to high-quality format
python clean_datasets.py "AlicanKiraz0_Cybersecurity-Dataset-Heimdall-v1.1"
```

### 3. Enhanced PRIMUS-Seed Training (Recommended)

#### A) Latest Zero-Truncation Training with PRIMUS
```bash
# Use our token-perfect PRIMUS-enhanced dataset (840K samples)
cd /Users/danielrodrigo/Workspace/PyScience/datasets
python3 -m mlx_lm lora 
  --model mlx_models/tinyllama_mlx 
  --data primus_training_FINAL 
  --train 
  --batch-size 4 
  --iters 10000 
  --learning-rate 1e-4 
  --steps-per-report 50 
  --steps-per-eval 200 
  --save-every 400 
  --max-seq-length 512 
  --num-layers 22 
  --adapter-path ./cybersecurity_finetuned_models/mlx_adapters_primus_ZERO_TRUNCATION_v1 
  --grad-checkpoint 
  --seed 42
```

#### B) PRIMUS Integration (if you need to rebuild dataset)
```bash
# Download PRIMUS-Seed (requires HuggingFace access approval)
python3 integrate_primus_datasets.py

# Fix token limits using actual TinyLlama tokenizer
python3 simple_token_fixer.py

# Files created:
# - train_FINAL.jsonl (840,658 samples, max 480 tokens each)
# - valid_FINAL.jsonl (71,189 validation samples)
```

### 4. Legacy Training Methods

#### A) MLX Training (Previous Method)
```bash
python train_comprehensive.py  # Uses comprehensive_cybersec_training.jsonl (232K samples)
```

#### B) HuggingFace Training
```bash
python train_model.py --technique lora --num-epochs 3 --lr 2e-4
```

#### C) PEFT Techniques Comparison (Educational)

#### D) Simple Training (Basic Implementation)
```bash
python train_model.py --technique simple
```

### 4. Evaluate Model

```bash
python evaluate_model.py trained_models/cybersec_tinyllama_20250714_123456 processed/AlicanKiraz0_Cybersecurity-Dataset-Heimdall-v1.1_processed/validation.jsonl --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

## 🎯 Pipeline Features

### PRIMUS-Seed Integration (July 2025)
- **Premium Dataset**: Access to PRIMUS-Seed cybersecurity content (170K+ samples)
- **Zero Truncation**: Perfect token management using actual TinyLlama tokenizer 
- **Smart Splitting**: Long samples intelligently split rather than truncated
- **Format Conversion**: PRIMUS content → conversational format → MLX training
- **Source Attribution**: Each sample includes original source context
- **Quality Control**: All samples verified ≤ 480 tokens (no "[WARNING]" messages)

### Key Dataset Files
- `train_FINAL.jsonl` - 840,658 token-perfect training samples
- `valid_FINAL.jsonl` - 71,189 validation samples  
- `primus_training_FINAL/` - MLX-ready directory structure
- `integrate_primus_datasets.py` - PRIMUS dataset integration script
- `simple_token_fixer.py` - Token limit fixer using TinyLlama tokenizer

### Data Collection
- **Unified Research**: Automatically searches HuggingFace for cybersecurity datasets
- **Size Filtering**: Only downloads datasets under 5GB
- **Quality Assessment**: Analyzes dataset structure before download
- **Batch Processing**: Download multiple datasets efficiently

### Data Processing
- **Format Standardization**: Converts various formats to consistent chat format
- **Quality Enhancement**: Improves responses for agentic behavior
- **Automatic Validation**: Checks data quality and reports issues
- **Train/Val Splitting**: Creates proper training/validation splits

### Training
- **MLX Optimization**: Apple Silicon optimized training (10-20x speedup)
- **Multiple PEFT Methods**: LoRA, AdaLoRA, IA3, Prefix Tuning
- **Full Fine-tuning**: Train all parameters for maximum performance  
- **Automatic Configuration**: Optimized hyperparameters per technique
- **Progress Monitoring**: Real-time training progress tracking
- **Error Recovery**: Built-in error detection and correction

#### 🔧 Training Techniques Comparison

| Technique | Parameters | Speed | Memory | Use Case |
|-----------|------------|-------|--------|----------|
| **LoRA** | 0.1-1% | Fast | Low | General fine-tuning |
| **AdaLoRA** | 0.1-1% | Fast | Low | Adaptive rank selection |
| **IA3** | <0.1% | Very Fast | Very Low | Lightweight adaptation |
| **Prefix Tuning** | 0.1% | Fast | Low | Task-specific prefixes |
| **Full Fine-tuning** | 100% | Slow | High | Maximum performance |
| **MLX (Apple Silicon)** | 0.1-1% | Very Fast | Low | Apple hardware |

### Evaluation
- **Cybersecurity Knowledge**: Tests domain-specific understanding
- **Conversational Quality**: Evaluates helpfulness and coherence
- **Code Generation**: Assesses programming capabilities
- **Automated Metrics**: ROUGE, BLEU, and custom metrics
- **Comprehensive Reports**: Human-readable evaluation summaries

## 🔧 Advanced Usage

### Custom Model Configuration

```python
# Add new model to unified_trainer.py
MODEL_CONFIGS["custom_model"] = {
    "model_name": "your/custom-model",
    "recommended_lr": 1e-4,
    "recommended_batch_size": 4,
    "lora_config": {
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
        "lora_dropout": 0.1,
        "bias": "none",
        "task_type": TaskType.CAUSAL_LM
    }
}
```

### Custom Data Processing

```python
from clean_datasets import DataProcessor

processor = DataProcessor()
# Override methods for custom processing
def custom_enhance_for_agentic_behavior(self, data):
    # Custom enhancement logic
    return enhanced_data

processor.enhance_for_agentic_behavior = custom_enhance_for_agentic_behavior
```

### Batch Operations

```bash
# Process multiple datasets
python utils.py merge dataset1.jsonl dataset2.jsonl --output combined.jsonl

# Create samples for quick testing
python utils.py sample large_dataset.jsonl test_sample.jsonl --size 1000

# Analyze dataset statistics
python utils.py analyze processed_dataset.jsonl
```

## 📊 Quality Standards

The pipeline maintains high data quality through:

1. **Heimdall Standard**: All data is processed to match the high-quality Heimdall format
2. **Conversational Enhancement**: Responses are enhanced for helpful, agentic behavior
3. **Automatic Validation**: Quality scores must exceed 90% to pass validation
4. **Error Correction**: Training issues are automatically detected and corrected
5. **Comprehensive Testing**: Multiple evaluation metrics ensure model quality

## 🔄 Pipeline Workflow

```mermaid
graph LR
    A[Research Datasets] --> B[Download Data]
    B --> C[Process & Clean]
    C --> D[Validate Quality]
    D --> E[Train Model]
    E --> F[Monitor Training]
    F --> G[Evaluate Model]
    G --> H[Generate Report]
```

## 🛠️ Configuration

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables (optional)
export HF_TOKEN="your_huggingface_token"
export WANDB_DISABLED=true  # Disable wandb logging
```

### Training Configuration
Modify `MODEL_CONFIGS` in `training/unified_trainer.py` for custom setups.

### Processing Configuration
Adjust quality thresholds in `data_processing/unified_data_processor.py`.

## 📈 Monitoring and Logging

- **Training Progress**: Real-time loss tracking and step timing
- **Quality Metrics**: Automatic data quality assessment
- **Error Detection**: Training issues caught and reported
- **Evaluation Reports**: Comprehensive model performance analysis

## 🔒 Cybersecurity Focus

This pipeline is specifically designed for cybersecurity applications:

- **Domain Datasets**: Focuses on security-related conversational data
- **Code Security**: Emphasizes secure coding practices in generated code
- **Vulnerability Detection**: Trains models to identify security issues
- **Agentic Behavior**: Promotes helpful, educational responses

## 📝 Current Status

- ✅ **Pipeline Built**: All unified scripts created and organized
- ✅ **Workspace Cleaned**: Removed duplicates, organized by technique
- ✅ **Data Processed**: 18,866 training + 2,097 validation samples ready
- ✅ **Multiple Training Techniques**: LoRA, AdaLoRA, IA3, Prefix Tuning, Full Fine-tuning, MLX
- ✅ **MLX Optimization**: Apple Silicon acceleration installed and configured
- ✅ **HuggingFace Issues Resolved**: Fixed API compatibility, working dataset collector available
- ✅ **New Datasets Downloaded**: 10 additional cybersecurity datasets (8,000+ samples)
- 🔄 **PEFT Training Active**: Testing LoRA technique (5,000 samples)
- ⏳ **Evaluation Pending**: Ready for automated evaluation

### Available Training Files

| File | Technique | Description | Best For |
|------|-----------|-------------|----------|
| `real_cybersecurity_finetuner.py` | LoRA | Standard LoRA implementation | General use |
| `mlx_cybersecurity_finetuner.py` | LoRA+MLX | Apple Silicon optimized | M-series Macs |
| `finetune_phi3_cybersecurity.py` | Full Fine-tuning | All parameters trainable | Maximum performance |
| `peft_techniques_trainer.py` | Multi-PEFT | LoRA/AdaLoRA/IA3/Prefix | Learning/comparison |
| `simple_cybersec_trainer.py` | Basic LoRA | Simplified implementation | Debugging/learning |

### Cleanup Summary (2025-07-16)
- ✅ Removed duplicate Python files from main directory
- ✅ Organized files into proper subfolders (training/, evaluation/, etc.)
- ✅ Removed unused CloScience/scripts/ folder
- ✅ Cleaned up old test directories (test_data/, finetune/, etc.)
- ✅ Consolidated training techniques into individual, focused files
- ✅ Added comprehensive PEFT techniques trainer

## 🔧 Troubleshooting

### Slow Training Performance (Apple Silicon)

If training is extremely slow (>100 seconds per step), try these optimizations:

#### 1. Reduce Context Length
```python
# In real_cybersecurity_finetuner.py, line ~365
'max_length': 512,  # Reduce from 1024 to 512
```

#### 2. Optimize Batch Configuration
```python
# Increase batch size, reduce gradient accumulation
'batch_size': 8,                    # Increase from 2 to 8
'gradient_accumulation_steps': 2,   # Reduce from 8 to 2
```

#### 3. Enable MPS (Apple Silicon GPU)
```python
# Add to training script
import torch
if torch.backends.mps.is_available():
    device_map = {"": 0}
    torch_dtype = torch.float16
```

#### 4. Use Smaller Model for Testing
```python
# For rapid iteration, consider even smaller models:
'model_name': 'distilgpt2',  # Much smaller for testing
```

#### 5. Reduce Dataset Size for Testing
```bash
# Create smaller sample for testing optimizations
python utils/unified_utils.py sample cybersecurity_datasets/processed/cybersecurity_agentic_clean_train.jsonl test_sample.jsonl --size 1000
```

### Memory Issues
- Reduce `max_length` to 256 or 512
- Increase `gradient_accumulation_steps` and reduce `batch_size`
- Use `torch_dtype=torch.float16` if supported

### Training Hangs or Freezes
- Check for MPS compatibility issues
- Try `device_map=None` to force CPU training
- Ensure adequate disk space for checkpoints

## 🎉 Next Steps

### Immediate (Performance Optimization)
1. **Optimize Training Configuration**: Apply Apple Silicon optimizations from troubleshooting section
2. **Test with Smaller Model**: Validate pipeline with distilgpt2 or similar lightweight model
3. **Benchmark Performance**: Measure steps/second with different configurations
4. **MPS Integration**: Properly configure Apple Silicon GPU acceleration

### Short Term
1. **Resume Training**: Once optimized, restart with `python real_cybersecurity_finetuner.py`
2. **Monitor Progress**: Use `python training_monitor.py` to track performance
3. **Automated Evaluation**: Run `python automated_model_evaluator.py` post-training
4. **Performance Analysis**: Analyze model capabilities and limitations

### Long Term
1. **Pipeline Refinement**: Improve based on evaluation results
2. **Production Deployment**: Package for use in other applications
3. **Multi-Model Support**: Add optimized configs for different Apple Silicon variants
4. **Advanced Features**: Add distributed training, quantization, and deployment tools

### Optimization Priority
🔥 **Critical**: Fix training speed (current: 239s/step → target: <10s/step)
⚡ **High**: Enable MPS acceleration for Apple Silicon
📊 **Medium**: Add real-time monitoring dashboard
🚀 **Low**: Advanced features and deployment options

---

*This unified pipeline consolidates all cybersecurity ML training functionality into a clean, maintainable system ready for production use.*

## 🔧 HuggingFace Datasets - Now Working!

### ✅ Issues Resolved
The HuggingFace datasets library is **working correctly**. Previous issues were due to API compatibility:

- **Fixed**: Updated API calls for datasets>=4.0.0
- **Fixed**: Proper handling of current DatasetInfo structure  
- **Fixed**: Rate limiting and error handling for gated datasets
- **Added**: Working dataset collector with comprehensive fallback methods

### 📊 Latest Dataset Collection Results
Using the fixed collector (`working_dataset_collector.py`):

```bash
# Research cybersecurity datasets (working API)
python working_dataset_collector.py research

# Download top datasets (working downloads)
python working_dataset_collector.py download-top --max-downloads 10
```

**Results**: Successfully found **76 cybersecurity datasets**, analyzed **20 in detail**, and downloaded **10 high-quality datasets** with 8,000+ samples total.

### 📂 New Cybersecurity Datasets Available

| Dataset | Type | Samples | Use Case |
|---------|------|---------|----------|
| `zeroshot/cybersecurity-corpus` | Classification | 789 | Binary security classification |
| `schooly/Cyber-Security-Breaches` | Incidents | 1,000 | Security breach analysis |
| `CyberNative/github_cybersecurity_READMEs` | Documentation | 1,000 | Security project docs |
| `bnsapa/cybersecurity-ner` | NER | 1,000 | Security entity recognition |
| `Vanessasml/cybersecurity_32k_instruction` | Instructions | 1,000 | Security Q&A training |
| `jcordon5/cybersecurity-rules` | Rules | 949 | Security guidelines |
| `dattaraj/rag_eval_cybersecurity` | RAG | 33 | Retrieval evaluation |

All datasets downloaded to `cybersecurity_datasets/` with metadata and ready for processing.
