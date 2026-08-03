"""Load and serialise :class:`~finetune.config.RunConfig` as YAML.

Two properties matter here and neither is free:

Unknown keys are an error. A config file that silently ignores a misspelled or
unsupported key is worse than no config file, because it reports success while
doing something else. The previous version of this repository shipped a
``lora_config.yaml`` setting ``lora_alpha: 64`` that nothing read.

Relative paths resolve against the config file, not the working directory, so a
config can be committed and still work when run from anywhere.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, TypeVar

import yaml

from .capabilities import Backend, Method
from .config import AdapterConfig, QuantizationConfig, TrainingConfig
from .errors import ConfigError
from .run import RunConfig

T = TypeVar("T")

_NESTED: dict[str, type] = {
    "adapter": AdapterConfig,
    "training": TrainingConfig,
    "quantization": QuantizationConfig,
}
_TOP_LEVEL = {"model", "method", "backend", "data_dir", "output_dir", *_NESTED}


def _field_names(cls: type) -> frozenset[str]:
    return frozenset(f.name for f in dataclasses.fields(cls))


def _suggest(unknown: str, valid: frozenset[str] | set[str]) -> str:
    """Return a ' (did you mean ...)' hint for a likely typo, else ''."""
    import difflib

    close = difflib.get_close_matches(unknown, sorted(valid), n=1, cutoff=0.7)
    return f" (did you mean {close[0]!r}?)" if close else ""


def _reject_unknown(section: str, given: dict[str, Any], valid: frozenset[str] | set[str]) -> None:
    unknown = set(given) - set(valid)
    if not unknown:
        return
    name = sorted(unknown)[0]
    where = f"{section}.{name}" if section else name
    raise ConfigError(
        f"unknown config key {where!r}{_suggest(name, valid)}. "
        f"Valid keys in {section or 'the top level'}: {', '.join(sorted(valid))}"
    )


def _build(cls: type[T], section: str, raw: object) -> T:
    if not isinstance(raw, dict):
        raise ConfigError(f"config section {section!r} must be a mapping, got {type(raw).__name__}")
    _reject_unknown(section, raw, _field_names(cls))
    return cls(**raw)


def _coerce_target_modules(adapter: dict[str, Any]) -> None:
    """Normalise ``target_modules`` from YAML into what AdapterConfig expects."""
    value = adapter.get("target_modules")
    if isinstance(value, list):
        if not all(isinstance(v, str) for v in value):
            raise ConfigError("adapter.target_modules list must contain only strings")
        adapter["target_modules"] = tuple(value)


def _enum(cls: type[T], section: str, value: object) -> T:
    try:
        return cls(value)  # type: ignore[call-arg]
    except ValueError:
        options = ", ".join(sorted(str(m.value) for m in cls))  # type: ignore[attr-defined]
        raise ConfigError(f"{section} must be one of: {options}; got {value!r}") from None


def parse_run_config(raw: dict[str, Any], base_dir: Path) -> RunConfig:
    """Build a :class:`RunConfig` from an already-parsed mapping.

    Args:
        raw: The decoded YAML mapping.
        base_dir: Directory that relative ``data_dir`` / ``output_dir`` resolve against.
    """
    _reject_unknown("", raw, _TOP_LEVEL)

    for required in ("model", "data_dir", "output_dir"):
        if required not in raw:
            raise ConfigError(f"config is missing required key {required!r}")

    adapter_raw = dict(raw.get("adapter") or {})
    _coerce_target_modules(adapter_raw)

    quantization = None
    if raw.get("quantization") is not None:
        quantization = _build(QuantizationConfig, "quantization", raw["quantization"])

    return RunConfig(
        model=str(raw["model"]),
        data_dir=_resolve(base_dir, raw["data_dir"]),
        output_dir=_resolve(base_dir, raw["output_dir"]),
        method=_enum(Method, "method", raw.get("method", Method.LORA.value)),
        backend=_enum(Backend, "backend", raw.get("backend", Backend.MLX.value)),
        adapter=_build(AdapterConfig, "adapter", adapter_raw),
        training=_build(TrainingConfig, "training", dict(raw.get("training") or {})),
        quantization=quantization,
    )


def _resolve(base_dir: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def load_run_config(path: str | Path) -> RunConfig:
    """Read and validate a YAML run config.

    Raises:
        ConfigError: if the file is missing, is not a mapping, contains an
            unknown key, or fails any field-level validation.
    """
    config_path = Path(path).expanduser()
    try:
        text = config_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {config_path}") from None
    except OSError as exc:
        raise ConfigError(f"could not read config {config_path}: {exc}") from exc

    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc

    if raw is None:
        raise ConfigError(f"config file is empty: {config_path}")
    if not isinstance(raw, dict):
        raise ConfigError(f"config must be a mapping at the top level, got {type(raw).__name__}")

    return parse_run_config(raw, base_dir=config_path.parent)


def to_dict(config: RunConfig) -> dict[str, Any]:
    """Serialise a config to plain types, with the learning rate resolved.

    Written alongside training output so a run's provenance records the values
    actually used, not the ones the file happened to omit.
    """
    training = dataclasses.asdict(config.training)
    training["learning_rate"] = config.learning_rate
    adapter = dataclasses.asdict(config.adapter)
    if isinstance(adapter["target_modules"], tuple):
        adapter["target_modules"] = list(adapter["target_modules"])

    payload: dict[str, Any] = {
        "model": config.model,
        "method": str(config.method),
        "backend": str(config.backend),
        "data_dir": str(config.data_dir),
        "output_dir": str(config.output_dir),
        "training": training,
    }
    if config.uses_adapter:
        payload["adapter"] = adapter
    if config.quantization is not None:
        payload["quantization"] = dataclasses.asdict(config.quantization)
    return payload


def dump_run_config(config: RunConfig) -> str:
    """Render a config as YAML."""
    return yaml.safe_dump(to_dict(config), sort_keys=False, default_flow_style=False)


__all__ = ["dump_run_config", "load_run_config", "parse_run_config", "to_dict"]
