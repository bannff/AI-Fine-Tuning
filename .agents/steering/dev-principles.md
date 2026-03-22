# Dev Principles

These are the engineering principles that govern this workspace. Every agent must audit their work against these before finishing. These principles are the "behavioral contract" for this repo.

## Architectural Principles

- **Clean Architecture**: Business logic (training loops, data processing) in `runtime/`, MCP surface in `mcp/`, and public API in `interface.py`.
- **Polymorphic/Agnostic**: Use adapter patterns for datasets and models. Avoid hardcoded paths or backend-specific logic where possible.
- **MCP-First**: When building tools, focus on exposing them via MCP for agent-driven execution.

## Code Quality Principles

- **<200 LOC**: Every file stays under 200 lines. Split large scripts into modular components.
- **SRP (Single Responsibility Principle)**: Each module, class, and function has a single, clear responsibility.
- **No Cross-Imports**: Maintain strict boundaries between different fine-tuning styles (SFT vs. LoRA) unless sharing common utilities.
- **DRY**: Factor out common training boilerplate into shared utilities.

## Fine-Tuning Principles

- **Reproducibility**: Always log configurations, hyper-parameters, and dataset versions.
- **Efficiency**: Prioritize PEFT (LoRA, QLoRA) for rapid iteration and resource management.
- **Validation**: Every training run must include a validation step and performance logging.

## Documentation Principles

- **Docs Follow Code**: Update steering files and role definitions when the repo structure changes.
- **WHY Over WHAT**: Explain the rationale behind specific hyper-parameters or architectural choices.
- **Interactive**: Ensure examples and quickstarts are runnable and up-to-date.

## Verification

- **Automated Checks**: Run verification scripts (e.g., `scripts/verify_repo.sh`) to ensure compliance.
- **Audit**: Periodically audit scripts for sensitive information or broken paths.
