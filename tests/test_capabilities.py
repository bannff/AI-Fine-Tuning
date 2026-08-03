"""Capability matrix tests.

The matrix is the repository's support claim. These tests assert that it is
internally consistent and that unsupported combinations raise rather than
silently degrading -- the failure mode the factory sibling has, where a `lora`
request routes to an in-memory stub because the real backend was never selected.
"""

from __future__ import annotations

import pytest

from finetune.capabilities import (
    BACKEND_EXTRA,
    BACKEND_PROBE,
    CAPABILITIES,
    NOT_IMPLEMENTED,
    Backend,
    Method,
    installed_backends,
    is_installed,
    resolve,
    supported_methods,
    supporting_backends,
)
from finetune.errors import UnsupportedCombinationError


def test_matrix_has_no_duplicate_entries() -> None:
    pairs = [(c.method, c.backend) for c in CAPABILITIES]
    assert len(pairs) == len(set(pairs))


def test_every_capability_is_documented() -> None:
    for capability in CAPABILITIES:
        assert capability.note.strip(), f"{capability.method}/{capability.backend} has no note"


def test_every_backend_has_an_extra_and_a_probe() -> None:
    for backend in Backend:
        assert backend in BACKEND_EXTRA
        assert backend in BACKEND_PROBE


def test_every_method_is_supported_by_at_least_one_backend() -> None:
    """A method in the enum with no implementation would be a stub."""
    for method in Method:
        assert supporting_backends(method), f"{method} has no backend"


def test_resolve_returns_the_matching_capability() -> None:
    capability = resolve(Method.LORA, Backend.MLX)
    assert capability.method is Method.LORA
    assert capability.backend is Backend.MLX


def test_unsupported_pair_raises_and_names_alternatives() -> None:
    """torch full fine-tuning is not implemented; the error must say so plainly."""
    with pytest.raises(UnsupportedCombinationError) as excinfo:
        resolve(Method.FULL, Backend.TORCH)

    error = excinfo.value
    assert error.method == "full"
    assert error.backend == "torch"
    assert error.supported_by == ("mlx",)
    assert "Supported by: mlx" in str(error)


def test_mlx_supports_every_method() -> None:
    assert set(supported_methods(Backend.MLX)) == set(Method)


def test_torch_does_not_claim_full_finetuning() -> None:
    assert Method.FULL not in supported_methods(Backend.TORCH)


def test_qlora_on_torch_declares_its_extra_requirement() -> None:
    """bitsandbytes is CUDA-only; the constraint belongs in the matrix, not a README."""
    capability = resolve(Method.QLORA, Backend.TORCH)
    assert "bitsandbytes" in capability.extra_requirements
    assert "bitsandbytes" in capability.note


def test_not_implemented_entries_explain_where_to_get_them() -> None:
    assert NOT_IMPLEMENTED
    for name, reason in NOT_IMPLEMENTED.items():
        assert len(reason) > 40, f"{name} needs a real explanation"


def test_is_installed_is_a_boolean_for_every_backend() -> None:
    for backend in Backend:
        assert isinstance(is_installed(backend), bool)


def test_installed_backends_is_a_subset_of_all_backends() -> None:
    assert set(installed_backends()) <= set(Backend)


def test_enum_values_are_stable_strings() -> None:
    """These appear in YAML configs, so renaming them is a breaking change."""
    assert {str(m) for m in Method} == {"lora", "dora", "qlora", "full"}
    assert {str(b) for b in Backend} == {"mlx", "torch"}
