## Description

<!-- Provide a brief description of the changes in this PR -->

## Related Issue

<!-- Link to the related issue (use "Fixes #123" to auto-close the issue when merged) -->

Fixes #

## Changes Made

<!-- List the key changes in this PR -->

- 
- 

## Checklist

<!-- Check all that apply -->

- [ ] Follows the conventions in `.agents/steering/dev-principles.md`
- [ ] `uv run ruff check .` and `uv run ruff format --check .` pass
- [ ] `uv run mypy` passes
- [ ] `uv run pytest` passes
- [ ] `bash scripts/verify_repo.sh` passes
- [ ] Every file in `src/` is still under 200 lines
- [ ] No new module is named after a package it imports
- [ ] Core stays import-light: nothing heavy imported at module scope
- [ ] If a method or backend changed: `capabilities.py` and the README support
      table agree
- [ ] If a default changed: `docs/lora-practice.md` explains why, with a source
- [ ] No weights, adapters, checkpoints or logs added to the repository

## Type of Change

<!-- Check all that apply -->

- [ ] Feature (new functionality)
- [ ] Bug fix
- [ ] Chore (maintenance, refactoring, tooling)
- [ ] Documentation
- [ ] Security fix

## Testing

<!-- Describe how you tested these changes (e.g., validation runs, benchmarking results) -->

## Additional Notes

<!-- Any additional context, screenshots, or notes for reviewers -->
