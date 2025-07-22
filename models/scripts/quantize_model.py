"""
Quantize the fine-tuned TinyLlama 1.1B-Chat model to 8-bit using HuggingFace transformers and bitsandbytes (CPU-compatible).
Saves quantized model to ./tinyllama_cybersecurity_quantized
"""
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os

# Paths
FINETUNED_MODEL_DIR = "./tinyllama_cybersecurity_finetuned"
QUANTIZED_MODEL_DIR = "./tinyllama_cybersecurity_quantized"
BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load tokenizer from base model
print("Loading tokenizer from base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

# Quantization config (8-bit, CPU-compatible)
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True
)

# Load and quantize model
print("Loading and quantizing model...")
model = AutoModelForCausalLM.from_pretrained(
    FINETUNED_MODEL_DIR,
    quantization_config=bnb_config,
    device_map="auto"
)

# Save quantized model
print(f"Saving quantized model to {QUANTIZED_MODEL_DIR} ...")
model.save_pretrained(QUANTIZED_MODEL_DIR)
tokenizer.save_pretrained(QUANTIZED_MODEL_DIR)
print("Quantization complete.")
