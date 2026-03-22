# Fine-Tuning Project Overview

This repository is a consolidated stable for fine-tuning LLMs using various styles and techniques. It is designed to be highly professional and agent-ready.

## Project Structure

```
fine-tuning/
├── .agents/        # Agent steering, skills, and workflows
├── .beads/         # AI-native issue tracking
├── .kiro/          # Kiro agent role definitions
├── .github/        # GitHub governance (templates, labels)
├── full-fine-tuning/
│   ├── cpt/        # Continued Pre-Training
│   ├── dpo/        # Direct Preference Optimization
│   └── sft/        # Supervised Fine-Tuning
├── peft/           # Parameter-Efficient Fine-Tuning
│   ├── lora/       # LoRA (Low-Rank Adaptation)
│   └── qlora/      # Quantized LoRA
├── scripts/        # Utility and verification scripts
└── README.md
```

## Core Vision

1.  **Consolidation**: A single source of truth for fine-tuning across different frameworks (MLX, PyTorch, etc.).
2.  **Agent-Friendliness**: Designed from the ground up to be explored and operated by AI agents.
3.  **Professionalism**: Strict adherence to engineering best practices (SRP, <200 LOC, Clean Architecture).

## Fine-Tuning Styles

### 1. Full Fine-Tuning (SFT/CPT)
Complete model training pipelines for deep adaptation.
- **SFT**: Supervised Fine-Tuning for instruction following.
- **CPT**: Continued Pre-Training for domain adaptation.

### 2. PEFT (LoRA/QLoRA)
Efficient adaptation techniques that reduce memory footprint and training time.
- **LoRA**: Lightweight adapters for specific skills or knowledge.
- **QLoRA**: Quantized training for even greater efficiency on consumer hardware.

---

*Harnessing the power of specialized models through professional fine-tuning.*
