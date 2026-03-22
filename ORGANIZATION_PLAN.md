# Fine Tuning Project Organization Plan

## Current Inventory Analysis

I have analyzed your `/workspace/models` and `/workspace/datasets` folders and found the following resources scattered across your workspace:

### 1. Models & Adapters
*   **`models/cybersecurity_finetuned_models/mlx_adapters_superior_v1`**: A LoRA adapter (Rank 16, Alpha 128).
*   **`models/PyScience/adapters`**: Likely another LoRA adapter.
*   **`models/PyScience/phi-3-cybersec-lora`**: A Phi-3 specific LoRA project.
*   **`models/PyScience/tinyllama_heimdall_v1_merged_finetuned`**: A merged model (likely SFT or merged LoRA).

### 2. Training Scripts
*   **QLoRA**:
    *   `models/PyScience/scripts/finetune_phi3_qlora_mlx.py`
    *   `models/PyScience/scripts/finetune_tinyllama_qlora_mlx.py`
*   **SFT (Supervised Fine-Tuning)**:
    *   `models/PyScience/scripts/finetune_tinyllama_full_sft.py`
    *   `models/PyScience/scripts/finetune_phi3_sft_mlx.py`
    *   `models/PyScience/scripts/README_FULL_SFT.md`
*   **LoRA (General)**:
    *   `datasets/scripts/train_tinyllama_lora.py`
    *   `datasets/scripts/train_gold_lora.sh`
    *   `datasets/scripts/train_cybersec_lora.sh`

### 3. Configs
*   `datasets/scripts/lora_config.yaml`
*   `datasets/scripts/tinyllama_lora_config.yaml`
*   `models/PyScience/mamba/configs/` (Mamba specific)

## Proposed Organization

I have created the following structure in `models/Fine-Tuning/`:

### `full-fine-tuning/`
*   **`cpt/`**: (Empty) Place Continued Pre-Training scripts here.
*   **`sft/`**: Move `finetune_*_sft_*.py` scripts here.
*   **`dpo/`**: (Empty) Place DPO scripts here.

### `peft/`
*   **`qlora/`**: Move `finetune_*_qlora_*.py` scripts here.
*   **`lora/`**: Move `mlx_adapters_superior_v1`, `phi-3-cybersec-lora`, and `train_*_lora.py` scripts here.

## Next Steps
To complete the organization, I recommend moving the files. **Shall I proceed with moving these files?**
(Note: Moving files might break hardcoded paths in your existing scripts, so you may need to update paths in `config.yaml` files).
