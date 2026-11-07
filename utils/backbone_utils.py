from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import AutoConfig

from utils.project_paths import resolve_existing_path


DEFAULT_BACKBONE_FAMILY = "auto"
LLADA_BACKBONE_FAMILY = "llada"
DREAM_BACKBONE_FAMILY = "dream"
SUPPORTED_BACKBONE_FAMILIES = (
    DEFAULT_BACKBONE_FAMILY,
    LLADA_BACKBONE_FAMILY,
    DREAM_BACKBONE_FAMILY,
)


@dataclass(frozen=True)
class BackboneSpec:
    family: str
    path: str
    model_type: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    mask_token_id: int | None
    pad_token_id: int | None


def _read_local_config_dict(path: str) -> dict[str, Any] | None:
    resolved = resolve_existing_path(path)
    if not resolved:
        return None

    resolved_path = Path(resolved)
    if resolved_path.is_file():
        config_path = resolved_path
    else:
        config_path = resolved_path / "config.json"

    if not config_path.exists():
        return None

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict config in {config_path}, got {type(data)!r}")
    return data


def _read_config_dict(path: str) -> dict[str, Any]:
    data = _read_local_config_dict(path)
    if data is not None:
        return data

    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    if hasattr(config, "to_dict"):
        return dict(config.to_dict())
    return dict(vars(config))


def _infer_family_from_model_type(model_type: str) -> str:
    normalized = str(model_type or "").strip().lower()
    if "llada" in normalized:
        return LLADA_BACKBONE_FAMILY
    if "dream" in normalized:
        return DREAM_BACKBONE_FAMILY
    raise ValueError(f"Unsupported backbone model_type: {model_type!r}")


def _require_int(config_dict: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = config_dict.get(key)
        if value is not None:
            return int(value)
    raise ValueError(f"Could not find any of keys {keys!r} in config.")


def inspect_pretrained_backbone(
    path: str,
    *,
    explicit_family: str = DEFAULT_BACKBONE_FAMILY,
) -> BackboneSpec:
    if explicit_family not in SUPPORTED_BACKBONE_FAMILIES:
        raise ValueError(
            f"Unsupported backbone family: {explicit_family!r}. "
            f"Expected one of {SUPPORTED_BACKBONE_FAMILIES!r}."
        )

    config_dict = _read_config_dict(path)
    model_type = str(config_dict.get("model_type", "")).strip()
    detected_family = _infer_family_from_model_type(model_type)
    family = detected_family if explicit_family == DEFAULT_BACKBONE_FAMILY else explicit_family
    if explicit_family != DEFAULT_BACKBONE_FAMILY and explicit_family != detected_family:
        raise ValueError(
            f"Backbone family mismatch for {path}: explicit={explicit_family!r}, "
            f"config model_type={model_type!r} -> detected={detected_family!r}."
        )

    vocab_size = _require_int(config_dict, "vocab_size", "embedding_size")
    hidden_size = _require_int(config_dict, "d_model", "hidden_size")
    num_hidden_layers = _require_int(config_dict, "n_layers", "num_hidden_layers")
    mask_token_id = config_dict.get("mask_token_id")
    pad_token_id = config_dict.get("pad_token_id")
    return BackboneSpec(
        family=family,
        path=path,
        model_type=model_type,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        mask_token_id=None if mask_token_id is None else int(mask_token_id),
        pad_token_id=None if pad_token_id is None else int(pad_token_id),
    )


def default_training_n_layers(spec: BackboneSpec) -> int:
    if spec.family == LLADA_BACKBONE_FAMILY:
        return min(27, spec.num_hidden_layers)
    if spec.family == DREAM_BACKBONE_FAMILY:
        return spec.num_hidden_layers
    raise ValueError(f"Unsupported backbone family: {spec.family!r}")
