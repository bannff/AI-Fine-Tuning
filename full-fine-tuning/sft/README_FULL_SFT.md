# Full SFT Fine-Tuning with MLX-LM

This directory contains full Supervised Fine-Tuning (SFT) scripts for TinyLlama and Phi-3 using MLX-LM on your Mac.

## What's in Here

- **`finetune_tinyllama_full_sft.py`** - Full SFT for TinyLlama-1.1B (regular bfloat16, 2.0 GB)
- **`finetune_phi3_full_sft.py`** - Full SFT for Phi-3-mini (regular bfloat16, 7.1 GB)
- **Dataset** - Heimdall cybersecurity data split into train/valid/test at `data/heimdall_sft_data/`

## Key Facts

**Your Models (Verified)**
- TinyLlama: `tinyllama_mlx` (2.0 GB, bfloat16, **FULL PRECISION**)
- Phi-3: `Phi-3-mini-128k-instruct` (7.1 GB, bfloat16, **FULL PRECISION**)

**Memory Requirements**
- TinyLlama full training: ~8 GB (✅ Fits on Mac 20GB)
- Phi-3 full training: ~24-35 GB (❌ **Does NOT fit** - will likely OOM)

**Framework**
- Uses MLX-LM's `--fine-tune-type full` for complete parameter fine-tuning
- Not LoRA/QLoRA - updating all weights
- Gradient checkpointing enabled to reduce memory usage

## Usage

### TinyLlama

```bash
# Full training (600 iterations)
python3 finetune_tinyllama_full_sft.py

# Test mode (50 iterations, diagnostic)
python3 finetune_tinyllama_full_sft.py --test
```

### Phi-3

```bash
# Full training (600 iterations)  
python3 finetune_phi3_full_sft.py

# Test mode (50 iterations, diagnostic)
python3 finetune_phi3_full_sft.py --test
```

## Current Issues & Tuning

### Loss Becoming NaN
If training loss becomes NaN during validation, try:

1. **Lower learning rate further** (edit script):
   ```python
   LEARNING_RATE = 1e-6  # Even smaller
   ```

2. **Add learning rate warmup** (needs MLX-LM config file):
   ```yaml
   # config.yaml
   learning_rate: 2e-6
   lr_schedule: 
     type: warmup_linear
     warmup_steps: 100
     total_steps: 600
   ```

3. **Use gradient accumulation**:
   ```python
   # Add --grad-accumulate 2  (needs MLX-LM version check)
   ```

4. **Reduce batch size**:
   ```python
   BATCH_SIZE = 2  # Down from 4
   ```

### Data Format
- Each record is a conversation with system/user/assistant messages
- Format: `{"text": "system: ... user: ... assistant: ..."}`
- 119,022 training records, 14,877 valid, 14,879 test

## Output Structure

```
outputs/
├── tinyllama_heimdall_full_sft/
│   ├── adapters.safetensors       # Fine-tuned weights
│   ├── adapter_config.json         # Training config
│   └── checkpoint-*/               # Intermediate checkpoints
│
├── phi3_heimdall_full_sft/
│   ├── adapters.safetensors
│   ├── adapter_config.json
│   └── checkpoint-*/
│
└── heimdall_sft_data/
    ├── train.jsonl                 # Training data
    ├── valid.jsonl                 # Validation data
    └── test.jsonl                  # Test data
```

## Next Steps

1. **Run test mode first** - validates setup without long training
2. **Adjust hyperparameters** if needed (see tuning section)
3. **Run full training** when confident
4. **Phi-3 might fail** - Consider QLoRA as fallback if OOM

## Comparison to QLoRA

| Method | Phi-3 Memory | TinyLlama Memory | Quality |
|--------|--------------|-----------------|---------|
| Full SFT | 24-35 GB ❌ | 8 GB ✅ | 100% (all params) |
| QLoRA | 5.65 GB ✅ | 3.3 GB ✅ | ~99% (adapters only) |
| LoRA | (doesn't matter) | 8 GB | ~95% |

If Phi-3 full training fails, your existing QLoRA scripts work great on ~6 GB.

## References

- MLX-LM Docs: https://github.com/ml-explore/mlx-lm
- Full fine-tuning flag: `--fine-tune-type full`
- Data format: Each line must be valid JSON with "text" key
