# LoRA FAQ and Benchmarks

## Common Questions

### Does LoRA decrease accuracy?
**No.** In many cases, it increases accuracy on specific tasks by preserving the pre-trained knowledge of the base model while adapting it precisely to the target domain.

### Is there a limit to how many adapters I can have?
No hard limit. Since adapters are small (20-50MB), you can store thousands of them. You can also load and swap them dynamically during inference.

### Can I share adapters between different models?
No. Adapters are specific to the architecture (and often the specific weights) they were trained on. A Llama-2 adapter will not work on a Phi-3 model.

## Benchmarks

### GPT-2 (NLG Tasks)
- **Full Fine-Tuning**: 68.2 BLEU
- **LoRA**: 70.4 BLEU (+3.2%)

### DeBERTa (GLUE)
- **Full Fine-Tuning**: 91.06% Avg
- **LoRA**: 91.32% Avg (+0.26%)

### Resource Savings
- **Trainable Parameters**: 10,000x reduction.
- **GPU Memory**: 3x reduction.
- **Storage**: 99.9% reduction.

## Sources
1. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
2. [HuggingFace PEFT Documentation](https://huggingface.co/docs/peft/)
3. [Microsoft LoRA Repository](https://github.com/microsoft/LoRA)

---
*Back to [Principles](LoRA_Principles.md) or [Implementation](LoRA_Implementation.md).*
