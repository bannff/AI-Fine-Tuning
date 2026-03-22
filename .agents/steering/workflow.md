# Development Workflow

This document outlines the professional workflow for contributing to the Fine-Tuning repository.

## 1. Issue Tracking (Beads)

We use **Beads** for local-first issue tracking. Before starting any work, ensure there is an issue:

```bash
bd list
bd create "Implement DPO pipeline for Mistral"
bd update <id> --claim
```

## 2. Implementation Standards

- **File Limits**: Keep files under 200 lines. If a training script gets too large, move the training loop or data processing to `runtime/`.
- **Principles**: Follow the `dev-principles.md` (SRP, Clean Architecture).
- **License**: Ensure all new files carry the appropriate header (see `LICENSE`).

## 3. Training Workflow

1.  **Prepare Dataset**: Use standardized formats (JSONL preferred).
2.  **Configure**: Create or update a `.yaml` config in the relevant folder.
3.  **Run**: Execute the training script via CLI/MCP.
4.  **Validate**: Run evaluation scripts to verify performance gains.
5.  **Clean up**: Remove temporary checkpoints or artifacts before committing.

## 4. GitHub Process

- **Branching**: Use descriptive names (`feature/`, `bugfix/`, `chore/`).
- **PRs**: Fill out the `pull_request_template.md` completely.
- **Approval**: All PRs must pass the `scripts/verify_repo.sh` check.

## 5. Agent Interaction

- Always use the **Implementer** role for coding tasks.
- Use the **QA-Tester** role for benchmarking and verification.
- Update **steering docs** if the architecture changes.
