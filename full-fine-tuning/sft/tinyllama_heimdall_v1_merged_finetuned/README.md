---
base_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:TinyLlama/TinyLlama-1.1B-Chat-v1.0
- lora
- transformers
---

# Model Card: TinyLlama Heimdall Fine-tuned

This model is a fine-tuned version of [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) on the Heimdall dataset.

## Model Details
- **Developed by:** Daniel Rodrigo
- **Model type:** Causal Language Model
- **Language(s):** English
- **Finetuned from model:** TinyLlama/TinyLlama-1.1B-Chat-v1.0

## Uses
This model is intended for research and testing purposes in fine-tuning styles and performance evaluation.

## Training Details
The model was fine-tuned using LoRA (Low-Rank Adaptation) on the Heimdall 7-variant dataset, optimized for efficient training on Apple Silicon.

### Framework versions
- PEFT 0.16.0
- Transformers
- PyTorch
- MLX (if applicable)