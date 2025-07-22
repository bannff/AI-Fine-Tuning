# Fine-Tuning Java Code Models with MLX-LM

## Datasets Used

- **saurabh5/rlvr-code-data-Java-sft** (Hugging Face):
  - Downloaded using the Hugging Face `datasets` library.
  - Format: Conversational (instruction-following) with `user` and `assistant` message pairs.
  - Path: `datasets/Java_HF_rlvr_sft/`

## Data Preparation

- The dataset is already in conversational format, suitable for MLX-LM supervised fine-tuning (SFT).
- No further conversion is needed for SFT or LoRA fine-tuning.

## Fine-Tuning with MLX-LM

### 1. Environment Setup
- Ensure you have MLX-LM and its dependencies installed (see [mlx-lm GitHub](https://github.com/ml-explore/mlx-lm)).
- Activate your Python environment if needed.

### 2. Example Fine-Tuning Command
Replace `/path/to/base-model` with your base model directory (e.g., Falcon Mamba MLX-LM format):

```
python -m mlx_lm.sft \
  --model /path/to/base-model \
  --data /Users/danielrodrigo/Workspace/datasets/Java_HF_rlvr_sft \
  --output /Users/danielrodrigo/Workspace/models/falcon-mamba-7b-instruct-mlx-java-sft \
  --epochs 3 \
  --batch-size 2 \
  --learning-rate 2e-5
```

- For LoRA/QLoRA, add the appropriate flags (see MLX-LM documentation).
- You can adjust `--epochs`, `--batch-size`, and `--learning-rate` as needed.

### 3. Monitoring and Evaluation
- Monitor training logs for loss/accuracy.
- After training, test the model using your agent wrapper or chat interface.

### 4. (Optional) Model Conversion for Inference
- To quantize or optimize the model for Apple Silicon, use `mlx_lm.convert` after fine-tuning.

## What Happens After Fine-Tuning? Model Output and Management

- **Fine-tuning creates a new model directory** at the path you specify with `--output`. This contains the updated weights, config, and any other files needed to use your fine-tuned model.
- **Your original (base) model is not changed.** You can always go back to it, re-fine-tune, or use it for other tasks.
- **Best practices:**
  - Name your model directories clearly (e.g., `-java-sft`, `-finetuned`, `-v1`, etc.).
  - Keep the original model if you want to compare, re-fine-tune, or use it for other domains.
  - Remove or archive old models only if you are sure you no longer need them, to save disk space and avoid confusion.
  - Update your inference/chat scripts to point to the new fine-tuned model directory.
- **Summary:**
  - Fine-tuning = new model files in a new directory.
  - Old model is safe unless you delete it.
  - Use clear names and update your scripts to avoid confusion.

## How Fine-Tuning Works (For Beginners)

Fine-tuning is the process of taking a pre-trained language model (like Falcon Mamba or Codestral Mamba) and training it further on a specific dataset to specialize it for a new task or domain (e.g., Java code generation, cybersecurity, agentic chat, etc.).

### Why Data Format Matters
- Modern language models (LLMs) are trained on large, structured datasets. For chatbots and agentic models, the data is often in a **conversational format**: a list of messages, each with a `role` (like `user` or `assistant`) and `content` (the text).
- This format helps the model learn how to respond in multi-turn conversations, follow instructions, and act as an agent.
- For code models, the dataset might pair a user prompt ("Write a Java function...") with the correct code as the assistant's reply.

### MLX-LM and Conversational Data
- MLX-LM and similar frameworks expect data in this conversational format for supervised fine-tuning (SFT).
- Each training example is a list of messages, e.g.:
  ```json
  {
    "messages": [
      {"role": "user", "content": "Write a Java function to reverse a string."},
      {"role": "assistant", "content": "public String reverse(String s) { ... }"}
    ]
  }
  ```
- This format is also used for instruction-following, agentic behaviors, and multi-turn chat.

### Other Data Formats
- Some older models or tasks use simpler formats (e.g., just input/output pairs, or plain text). For modern LLMs, conversational/instructional formats are preferred for best results.
- If your dataset is not in this format, you must convert it (see the provided conversion script for Juliet Java).

### Summary
- **Always check the required data format for your model/framework.**
- For MLX-LM and most modern LLMs, use a conversational format with `messages` (role/content pairs).
- Fine-tuning teaches the model to respond in the style and domain of your dataset, making it more useful for your specific needs.

## More on Data Formats and Fine-Tuning Methods

### Common Data Formats for LLM Fine-Tuning

- **Conversational Format (ChatML/OpenAI style):**
  - Used for chatbots, agentic models, and instruction-following LLMs.
  - Each example is a list of `messages` (role/content pairs), e.g.:
    ```json
    { "messages": [
        {"role": "user", "content": "Write a function..."},
        {"role": "assistant", "content": "def ..."}
      ] }
    ```
  - Required by most modern chat LLMs (OpenAI, MLX-LM, Mistral, etc.).

- **Hugging Face Transformers Format:**
  - Hugging Face datasets can be in many formats, but for code and text, common ones are:
    - **JSONL**: Each line is a JSON object with fields like `instruction`, `input`, `output`, or `text`.
    - **Parquet/Arrow**: Efficient tabular formats for large datasets.
    - **Plain text**: For language modeling, just a long text file.
  - Transformers models (like BERT, GPT-2, Llama) can be fine-tuned on any of these, but conversational/instructional formats are best for chat/instruction models.

- **BERT-style Format:**
  - For masked language modeling (MLM), data is usually plain text or tokenized text, not conversational.
  - Used for tasks like classification, NER, or QA, not chat.

### Fine-Tuning Methods

- **Supervised Fine-Tuning (SFT):**
  - Standard method: train the model to map prompts to correct responses.
  - Used for instruction-following, code generation, and chat.
  - Requires labeled data (input/output pairs or conversations).

- **LoRA (Low-Rank Adaptation):**
  - Parameter-efficient fine-tuning: only a small set of weights (adapters) are trained, not the whole model.
  - Saves memory and compute, allows fast adaptation to new tasks.
  - Supported by MLX-LM, Hugging Face, and others.

- **QLoRA:**
  - Like LoRA, but with quantized (compressed) weights for even more efficiency.
  - Useful for large models and limited hardware.

- **BERT-style Fine-Tuning:**
  - Used for tasks like classification, NER, or QA.
  - Data is usually plain text or tokenized, not conversational.
  - Not used for chat or agentic models.

### When to Use Each Format/Method
- **Conversational/instructional format:** For chatbots, agentic LLMs, code assistants, and instruction-following models.
- **Plain text/MLM:** For language modeling, BERT-style tasks, or pretraining.
- **LoRA/QLoRA:** When you want to fine-tune large models efficiently or on limited hardware.
- **SFT:** When you have high-quality labeled data and want to specialize a model for a new domain or task.

---

**This README documents the full fine-tuning workflow for Java code models using MLX-LM.**

---

# Advanced LoRA Training Configuration for MLX-LM

## 🎯 **Enhanced LoRA Setup Beyond Vanilla Training**

When fine-tuning for domain-specific tasks like cybersecurity, you can configure MLX-LM LoRA training with advanced parameters that go beyond basic settings. Here's what we implemented for our cybersecurity model training:

### **1. Enhanced Monitoring & Checkpointing**
```yaml
steps_per_report: 10      # Report loss every 10 steps (default: 10)
steps_per_eval: 100       # Validate every 100 steps (default: 200) 
save_every: 100           # Save checkpoints every 100 steps (default: 1000)
```

**What this means:**
- **Vanilla LoRA**: Saves checkpoint only at the end (or every 1000 steps)
- **Our setup**: Saves every 100 steps → 10x more frequent backups
- **Benefit**: Can resume from any 100-step interval if training crashes

### **2. Optimized LoRA Parameters**
```yaml
lora_parameters:
  rank: 16              # Higher rank (default: 8)
  alpha: 32             # 2x the rank (default: 16) 
  dropout: 0.1          # Added regularization
  scale: 2.0            # Custom scaling factor
```

**What this means:**
- **Vanilla LoRA**: rank=8, alpha=16 (basic capacity)
- **Our setup**: rank=16, alpha=32 (2x capacity for complex cybersecurity knowledge)
- **Benefit**: Can learn more complex patterns in cybersecurity data

### **3. Advanced Training Configuration**
```yaml
optimizer: adamw         # Better optimizer (default: adam)
learning_rate: 1e-4     # Optimized LR (default: 1e-5)
grad_checkpoint: true   # Memory optimization
mask_prompt: false      # Train on full sequences
max_seq_length: 512     # Explicit sequence length
seed: 42                # Reproducible training
```

**What this means:**
- **AdamW optimizer**: Better weight decay handling than Adam
- **10x higher learning rate**: Faster convergence for our dataset
- **Gradient checkpointing**: Saves memory for larger models

### **4. Comprehensive Validation Setup**
```yaml
val_batches: 25         # Validate on 25 batches (default: 10)
```

**What this means:**
- **More thorough validation**: Uses 2.5x more validation data per check
- **Better loss estimates**: More stable validation metrics

## 📊 **Comparison: Vanilla vs. Enhanced Setup**

| Feature | Vanilla LoRA | Our Enhanced Setup | Benefit |
|---------|-------------|-------------------|---------|
| **Checkpoints** | Every 1000 steps | Every 100 steps | 10x more resume points |
| **Validation** | Every 200 steps | Every 100 steps | 2x more monitoring |
| **LoRA Rank** | 8 | 16 | 2x model capacity |
| **LoRA Alpha** | 16 | 32 | Better learning scaling |
| **Optimizer** | Adam | AdamW | Better convergence |
| **Learning Rate** | 1e-5 | 1e-4 | 10x faster learning |
| **Validation Data** | 10 batches | 25 batches | More stable metrics |

## 🛠️ **Implementation Examples**

**Vanilla LoRA (basic):**
```bash
python3 -m mlx_lm lora \
  --model /path/to/model \
  --data /path/to/data \
  --train
```

**Our Enhanced Setup:**
```bash
python3 -m mlx_lm lora --config mlx_cybersec_training_config.yaml
```

**Example Enhanced Config File (mlx_cybersec_training_config.yaml):**
```yaml
# MLX-LM LoRA Training Configuration for Cybersecurity Model
model: /path/to/base/model
data: /path/to/training/data
train: true
fine_tune_type: lora
optimizer: adamw
batch_size: 4
iters: 1000
val_batches: 25
learning_rate: 1e-4
steps_per_report: 10
steps_per_eval: 100
save_every: 100
max_seq_length: 512
mask_prompt: false
grad_checkpoint: true
seed: 42

# Output path for the trained adapter
adapter_path: ./cybersecurity_finetuned_models/mlx_adapters_v3

# LoRA specific settings - improved parameters
lora_parameters:
  rank: 16
  alpha: 32
  dropout: 0.1
  scale: 2.0
```

## 🎓 **Key Learning Points for Teams:**

1. **Checkpointing Strategy**: Save frequently (every 100 steps) for long training jobs
2. **LoRA Tuning**: Higher rank/alpha for complex domains like cybersecurity
3. **Validation Frequency**: Monitor more often (every 100 vs 200 steps) for better insight
4. **Optimizer Choice**: AdamW generally outperforms Adam for transformer fine-tuning
5. **Learning Rate**: Domain-specific data may need higher LR than general datasets
6. **Reproducibility**: Always set a seed for consistent results

**The key insight**: **Domain-specific fine-tuning** (like cybersecurity) often benefits from **higher capacity LoRA settings** and **more frequent monitoring** than general-purpose training!

---

# Fine-Tuning Falcon Mamba (or Any Model) with Hugging Face Transformers

> **Note:** MLX-LM does not support LoRA or full fine-tuning for Mamba models. If you need to fine-tune Falcon Mamba or any Mamba architecture model, use Hugging Face Transformers (PyTorch) as described below.

## Step-by-Step Guide

### 1. Environment Setup
- Use a Linux machine or cloud instance with a CUDA GPU (NVIDIA recommended).
- Install Python 3.9+ and create a new virtual environment:
  ```sh
  python3 -m venv hf-mamba-venv
  source hf-mamba-venv/bin/activate
  ```
- Install the required packages:
  ```sh
  pip install torch transformers datasets accelerate peft bitsandbytes
  ```
  - `peft` and `bitsandbytes` are optional, for LoRA/QLoRA or quantized fine-tuning.

### 2. Prepare Your Data
- Your data should be in JSONL format, with each line a JSON object containing either:
  - **Chat format:**
    ```json
    { "messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
      ] }
    ```
  - **Prompt/Completion format:**
    ```json
    { "prompt": "...", "completion": "..." }
    ```
- Place your data in a directory, e.g. `datasets/java_hf_jsonl_export/train.jsonl` and `valid.jsonl`.

### 3. Example Fine-Tuning Command
- Use the Hugging Face `transformers` CLI or a Python script. Example with the CLI:
  ```sh
  accelerate launch transformers/examples/pytorch/language-modeling/run_clm.py \
    --model_name_or_path tiiuae/falcon-mamba-7b-instruct \
    --train_file datasets/java_hf_jsonl_export/train.jsonl \
    --validation_file datasets/java_hf_jsonl_export/valid.jsonl \
    --do_train --do_eval \
    --output_dir models/falcon-mamba-7b-instruct-java-sft \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --logging_steps 50 \
    --save_steps 500 \
    --fp16
  ```
- Adjust paths, batch size, and epochs as needed for your hardware and dataset size.
- For LoRA/QLoRA, see the [PEFT documentation](https://github.com/huggingface/peft).

### 4. Monitoring and Evaluation
- Training logs will show loss and evaluation metrics.
- After training, your fine-tuned model will be in the `output_dir` you specified.
- You can load and test it with the Hugging Face `transformers` library.

### 5. Best Practices
- Always keep a copy of your original model and data.
- Use clear directory names for outputs.
- Monitor GPU memory usage and adjust batch size if you run out of memory.
- For large models, consider using DeepSpeed or FSDP for distributed training.

---

**Summary:**
- MLX-LM does not support fine-tuning for Mamba models.
- Use Hugging Face Transformers for full fine-tuning on Linux/GPU.
- Prepare your data in JSONL format, follow the steps above, and monitor your training job.

For more details, see the [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers/main_classes/trainer) and [Falcon Mamba model card](https://huggingface.co/tiiuae/falcon-mamba-7b-instruct).
