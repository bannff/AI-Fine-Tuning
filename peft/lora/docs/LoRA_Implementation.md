# LoRA Implementation and Best Practices

## Best Practices

### Training Recommendations
1. **Rank Selection**: Start with `r=16`. It provides an excellent balance of expressiveness and efficiency.
2. **Target Modules**: Focus on attention layers (`q_proj`, `v_proj`, `o_proj`). These typically yield the highest impact.
3. **Scaling (Alpha)**: Usually set `alpha = 2 * r`. This scales the learned weights appropriately.
4. **Learning Rate**: loRA can often handle higher learning rates than full fine-tuning (e.g., 1e-4 or 2e-4).

### Resource Management
- **Mac/MPS**: Optimize batch sizes (start at 1 or 2) and use `torch_dtype="auto"`.
- **Quantization (QLoRA)**: Use 4-bit base models to fit large architectures (like Phi-3 or Llama-3) on consumer hardware.

## Technical Details

### Memory Profile (7B Model)
- **Base model**: ~14GB
- **Full Fine-Tuning**: ~28GB
- **LoRA Training**: ~16GB
- **Merged Inference**: ~14GB (Zero overhead)

### Merging Adapters
For production, always merge adapters back into the base model weights:
```python
model = model.merge_and_unload()
model.save_pretrained("./merged_model")
```

---
*Refer to [LoRA_Principles.md](LoRA_Principles.md) for core concepts and [LoRA_FAQ.md](LoRA_FAQ.md) for detailed benchmarks.*
