# PHI-3 Cybersecurity LoRA Fine-Tuning

A specialized cybersecurity-focused fine-tuning of Microsoft's PHI-3-mini model using LoRA (Low-Rank Adaptation) technique.

## 📁 Project Structure

```
phi-3-cybersec-lora/
├── train.py              # Main training script
├── README.md             # This file
├── lora_config.json      # Training configuration (auto-generated)
├── adapters/             # LoRA adapter weights (generated after training)
│   ├── adapters.safetensors
│   ├── adapter_config.json
│   └── checkpoints/
└── training/             # Training logs and outputs
```

## 🚀 Quick Start

### Prerequisites
- MLX and MLX-LM installed
- PHI-3-mini-128k-instruct-mlx model
- Cybersecurity training dataset (83K+ examples)
- Mac with Apple Silicon (M1/M2/M3/M4)

### Training

**Full Training (1000 iterations):**
```bash
python3 train.py
```

**Test Run (20 iterations):**
```bash
python3 train.py --test
```

**Resume from Checkpoint:**
```bash
python3 train.py --resume /path/to/checkpoint.safetensors
```

## 📊 Dataset

- **Training Examples**: 83,058 cybersecurity conversations
- **Validation Examples**: 1,000 examples
- **Format**: JSONL with conversational format
- **Topics**: Vulnerability analysis, security patterns, remediation guidance

## ⚙️ Training Configuration

- **Model**: PHI-3-mini-128k-instruct (3.8B parameters)
- **Trainable Parameters**: 3.146M (0.082% via LoRA)
- **Batch Size**: 2 (optimized for M4 Max 36GB)
- **Learning Rate**: 2e-4
- **Max Sequence Length**: 512 tokens
- **LoRA Configuration**: All layers (-1)

## 🔧 Using the Model

### Interactive Chat (Enhanced)

The enhanced chat interface includes MLX-LM best practices to prevent repetition and improve output quality:

```bash
python3 chat.py
```

**Enhanced Features:**
- **Temperature: 0.7** - Balanced creativity and coherence for cybersecurity responses
- **Min-P: 0.05** - Adaptive token filtering that dynamically adjusts based on model confidence
- **Top-P: 0.9** - Nucleus sampling fallback for consistent quality
- **Repetition Penalty: 1.05** - Light anti-repetition to prevent loops while preserving necessary repetition
- **Max Tokens: 512** - Extended length for comprehensive cybersecurity explanations
- **Smart Context Management** - Automatically maintains conversation history without memory overflow

**Chat Commands:**
- `quit` or `exit` - Exit the chat
- `clear` - Reset conversation history
- `settings` - Display current generation parameters

### Programming API

```python
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_repetition_penalty, apply_min_p, apply_top_p
import mlx.core as mx

# Load base model with LoRA adapters
model, tokenizer = load(
    "/Volumes/Crucial X9/ai-models/Phi-3-mini-128k-instruct-mlx", 
    adapter_path="/Volumes/Crucial X9/ai-models/PHI-3.5-cybersec-finetune/adapters"
)

# Enhanced sampler with best practices
def enhanced_sampler(logits):
    logits = apply_min_p(logits, min_p=0.05, min_tokens_to_keep=1)
    logits = apply_top_p(logits, top_p=0.9)
    temperature = 0.7
    logits = logits / temperature
    probs = mx.softmax(logits, axis=-1)
    return mx.random.categorical(mx.log(probs))

repetition_penalty = make_repetition_penalty(penalty=1.05, context_size=64)

# Generate cybersecurity responses
prompt = "What are the main types of cybersecurity attacks?"
response = generate(
    model, tokenizer, prompt, 
    max_tokens=512,
    sampler=enhanced_sampler,
    logits_processors=[repetition_penalty]
)
```

### Example Conversations

The model specializes in cybersecurity topics and provides detailed, actionable advice:

```
🔒 You: What is SQL injection?

🤖 Assistant: A SQL injection attack is a type of cybersecurity vulnerability that occurs when an attacker manipulates a web application to execute unintended SQL commands by injecting malicious SQL code into the application. This allows them to access, modify, or delete data in a database. To prevent SQL injection attacks, developers should use prepared statements and parameterized queries, which separate user input from SQL commands...
```

```
🔒 You: What are the main types of malware?

🤖 Assistant: There are several main types of malware, which include:
1. Viruses: These self-replicate by attaching themselves to files or spread through networks
2. Worms: Similar to viruses, but they do not need a host program to propagate
3. Trojans: These are disguised as legitimate software and trick users into executing them
4. Spyware: This collects information about user activities without their knowledge
5. Adware: Designed to display unwanted advertisements
6. Ransomware: Encrypts user files and demands payment for decryption...
```

## 🔧 Usage After Training

Once training is complete, use the LoRA adapters with MLX-LM:

```python
from mlx_lm import load, generate

# Load base model with LoRA adapters
model, tokenizer = load("path/to/base/model", adapter_path="./adapters")

# Generate cybersecurity responses
prompt = "Explain the OWASP Top 10 vulnerabilities"
response = generate(model, tokenizer, prompt, max_tokens=500)
```

## 📈 Training Progress

Monitor training with:
- Loss reporting every 10 iterations
- Validation every 50 iterations
- Checkpoints saved every 100 iterations
- Memory usage monitoring

## 🎯 Specialized Capabilities

This fine-tuned model excels at:
- Vulnerability analysis and explanation
- Security pattern identification
- Incident response guidance
- Compliance and remediation advice
- Cybersecurity best practices

## 📋 System Requirements

- **Memory**: 36GB+ unified memory recommended
- **Storage**: 10GB+ for model and adapters
- **OS**: macOS with Apple Silicon
- **Python**: 3.9+

## 🔍 Model Details

- **Base**: Microsoft PHI-3-mini-128k-instruct
- **Fine-tuning**: LoRA with rank optimization
- **Specialization**: Cybersecurity domain expertise
- **Context Length**: 128K tokens (base model capability)
