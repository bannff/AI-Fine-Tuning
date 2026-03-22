import json

from datasets import Dataset
from transformers import TrainingArguments

from peft import LoraConfig, get_peft_model


def load_local_jsonl(path: str) -> Dataset:
    """Load JSONL file into HF Dataset"""
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return Dataset.from_list(data)


def apply_lora(model, rank: int):
    """Apply LoRA configuration to the model."""
    config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=["qkv_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    return model


def get_training_args(output_dir: str, batch_size: int, epochs: int, iters: int = None):
    """Get optimized training arguments for Mac/MPS."""
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        max_steps=iters if iters else -1,
        learning_rate=1e-4,
        save_total_limit=2,
        use_mps_device=True,
        report_to="none",
    )
