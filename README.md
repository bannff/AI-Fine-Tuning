# 🎯 Fine-Tuning Stable

> **Consolidated, AI-Native, and Professional Fine-Tuning Resources**

Managed by **Daniel Rodrigo**, this repository is the definitive source for fine-tuning Large Language Models (LLMs) across diverse styles and frameworks. Built with a focus on **Agent-Readiness** and **Clean Architecture**, it follows the high engineering standards of the `python-factory` ecosystem.

---

## 🏗 Architecture

The stable is organized into three primary layers, ensuring a separation of concerns and a polymorphic design.

### 1. Full Fine-Tuning (`full-fine-tuning/`)
- **CPT**: Continued Pre-Training for domain adaptation.
- **SFT**: Supervised Fine-Tuning for instruction-following.
- **DPO**: Direct Preference Optimization for human-aligned outputs.

### 2. PEFT & Quantization (`peft/`)
- **LoRA**: Parameter-efficient adaptation via low-rank matrices.
- **MLX QLoRA**: 4-bit quantized matrices *hyper-optimized* for Apple Silicon (M-Series).
- **Runtime Adapters**: Highly modular `mlx_lm` abstractions enabling instant swap-in capabilities.

### 3. Agent Infrastructure (`.agents/`)
- **AI-Native Steering**: Optimized for operation by autonomous programming agents.
- **Beads**: Local-first, git-integrated issue tracking.
- **Roles**: 6 specialized cognitive definitions (Meta-Architect to Doc-Writer) mapped in `.agents/roles/`.

### 4. Quality Assurance (`tests/`)
- **Hypothesis**: Property-based randomized data generation.
- **PyTest**: Ensures pure functions and dynamic MLX JSONL chunking adhere to mathematical bounds.

---

## 📜 Engineering Tenets

This repository adheres to a strict "Behavioral Contract":

1.  **MCP-First**: Tools are exposed via MCP primitives for seamless agent interaction.
2.  **Clean Architecture**: Business logic in `runtime/`, MCP surface in `mcp/`, API in `interface.py`.
3.  **< 200 LOC**: Every file is kept under 200 lines to ensure modularity and scan-ability.
4.  **SRP**: The Single Responsibility Principle governs every module and function.
5.  **Polymorphic**: Model and dataset interactions use standardized Protocol adapters.

---

## 🚀 Quick Start

### 1. Setup
```bash
git clone https://github.com/bannff/fine-tuning.git
cd fine-tuning
bd init  # Initialize Beads issue tracking
```

### 2. Operation
Check the steering documentation for detailed workflows:
- [Development Principles](.agents/steering/dev-principles.md)
- [Project Overview](.agents/steering/project-overview.md)
- [Workflow Guide](.agents/steering/workflow.md)

### 3. Verification
```bash
bash scripts/verify_repo.sh
```

---

## ⚖️ License

This repository is licensed under the **Business Source License 1.1 (BSL 1.1)**. See [LICENSE](LICENSE) for details. It will convert to **Apache 2.0** on 2030-03-03.

---

*Precision Fine-Tuning. Agent-Ready. Professional.* ⚡
