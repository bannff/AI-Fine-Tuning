import os
import json
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

# Use only schema-consistent, intelligible datasets
DATASETS = [
    ("AlicanKiraz0/Cybersecurity-Dataset-Heimdall-v1.1", "train"),
    ("AlicanKiraz0/Cybersecurity-Dataset-v1", "train"),
]

# Load and concatenate datasets
all_datasets = []
for name, split in DATASETS:
    ds = load_dataset(name, split=split)
    # Filter for strict system/user/assistant schema
    ds = ds.filter(lambda x: all(k in x for k in ["system", "user", "assistant"]) and x["assistant"])
    all_datasets.append(ds)

merged_dataset = concatenate_datasets(all_datasets)

# Subsample for quick experiments
merged_dataset = merged_dataset.select(range(min(1000, len(merged_dataset))))

# Load model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")

# Apply QLoRA (LoRAConfig via peft)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Use local cleaned dataset
DATASET_PATH = "/Users/danielrodrigo/Workspace/CloScience/data/heimdall_merged_cleaned.jsonl"

def load_local_jsonl(path):
    with open(path, "r") as f:
        data = [json.loads(line) for line in f]
    return Dataset.from_list(data)

# Load cleaned, merged dataset
merged_dataset = load_local_jsonl(DATASET_PATH)

# Subsample for quick experiments
merged_dataset = merged_dataset.select(range(min(500, len(merged_dataset))))

# Preprocess: strict messages schema

def preprocess(example):
    # Flatten multi-turn messages into prompt/response pairs
    messages = example["messages"]
    prompt = ""
    response = ""
    for msg in messages:
        if msg["role"] == "user":
            prompt += msg["content"] + "\n"
        elif msg["role"] == "assistant":
            response += msg["content"] + "\n"
    prompt = prompt.strip()
    response = response.strip()
    full_text = f"User: {prompt}\nAssistant: {response}"
    input_ids = tokenizer(full_text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")["input_ids"].squeeze()
    labels = tokenizer(full_text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")["input_ids"].squeeze()
    return {"input_ids": input_ids, "labels": labels}

processed_dataset = merged_dataset.map(preprocess, batched=False, remove_columns=merged_dataset.column_names)

# Training arguments
training_args = TrainingArguments(
    output_dir="./tinyllama_heimdall_v1_merged_finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    save_steps=1000,
    logging_steps=100,
    fp16=False,  # Disable fp16 for MPS/CPU
    report_to="none",
    optim="adamw_torch"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
)

# Fine-tune
trainer.train()

# Save model
trainer.save_model("./tinyllama_heimdall_v1_merged_finetuned")
