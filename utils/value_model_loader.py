from __future__ import annotations

import json
import os
import sys
from typing import Any

import torch

from models.value_model import FULL_CLS_ARCH, READOUT_CLS_ARCH, ValueModel
from utils.project_paths import resolve_project_path


_VALUE_MODEL_ARG_SPECS: tuple[tuple[str, str, Any], ...] = (
    ("n_layers", "--n_layers", 27),
    ("mask_id", "--mask_id", 126336),
    ("value_model_arch", "--value_model_arch", READOUT_CLS_ARCH),
    ("pooling_mode", "--pooling_mode", "auto"),
    ("readout_heads", "--readout_heads", 8),
    ("readout_num_queries", "--readout_num_queries", 1),
    ("readout_aggregation", "--readout_aggregation", "mean"),
)


def add_value_model_args(parser: Any) -> Any:
    existing = set(getattr(parser, "_option_string_actions", {}).keys())

    def add(flag: str, *args: Any, **kwargs: Any) -> None:
        if flag not in existing:
            parser.add_argument(flag, *args, **kwargs)
            existing.add(flag)

    add("--n_layers", type=int, default=None)
    add("--mask_id", type=int, default=None)
    add("--value_model_arch", type=str, default=None, choices=[FULL_CLS_ARCH, READOUT_CLS_ARCH])
    add(
        "--pooling_mode",
        type=str,
        default=None,
        choices=["auto", FULL_CLS_ARCH, READOUT_CLS_ARCH, "mean_visible", "max_visible", "last_visible"],
    )
    add("--readout_heads", type=int, default=None)
    add("--readout_num_queries", type=int, default=None)
    add("--readout_aggregation", type=str, default=None, choices=["mean", "max", "concat"])
    return parser


def _load_train_config_args(value_model_path: str) -> dict[str, Any]:
    checkpoint_dir = os.path.dirname(resolve_project_path(value_model_path))
    train_config_path = os.path.join(checkpoint_dir, "train_config.json")
    if not os.path.exists(train_config_path):
        return {}
    with open(train_config_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        args = payload.get("args")
        if isinstance(args, dict):
            return args
    return {}


def _normalize_architecture(value: Any) -> str:
    arch = str(value or READOUT_CLS_ARCH)
    if arch == "cls_token":
        return FULL_CLS_ARCH
    return arch


def _cli_flag_present(flag: str) -> bool:
    return flag in sys.argv


def resolve_value_model_init_kwargs(
    args: Any,
    *,
    value_model_path: str,
    mask_id: int | None = None,
) -> dict[str, Any]:
    saved_args = _load_train_config_args(value_model_path)
    resolved: dict[str, Any] = {}
    for key, flag, default in _VALUE_MODEL_ARG_SPECS:
        cli_value = getattr(args, key, None)
        saved_value = saved_args.get(key)
        if _cli_flag_present(flag):
            value = cli_value
        elif saved_value is not None:
            value = saved_value
        elif cli_value is not None:
            value = cli_value
        else:
            value = default
        resolved[key] = value

    if mask_id is not None and not _cli_flag_present("--mask_id"):
        resolved["mask_id"] = int(mask_id)
    resolved["value_model_arch"] = _normalize_architecture(resolved["value_model_arch"])
    resolved["n_layers"] = int(resolved["n_layers"])
    resolved["mask_id"] = int(resolved["mask_id"])
    resolved["readout_heads"] = int(resolved["readout_heads"])
    resolved["readout_num_queries"] = max(int(resolved["readout_num_queries"]), 1)
    resolved["pooling_mode"] = str(resolved["pooling_mode"])
    resolved["readout_aggregation"] = str(resolved["readout_aggregation"])
    return resolved


def upgrade_value_model_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    readout_num_queries: int,
) -> dict[str, torch.Tensor]:
    state = dict(state_dict)
    if "cls_query" in state and "cls_queries" not in state:
        cls_query = state.pop("cls_query")
        if cls_query.dim() == 2:
            cls_query = cls_query.unsqueeze(1)
        if cls_query.dim() != 3:
            raise ValueError(f"Unexpected cls_query shape: {tuple(cls_query.shape)}")
        if cls_query.shape[1] == 1 and int(readout_num_queries) > 1:
            cls_query = cls_query.expand(-1, int(readout_num_queries), -1).clone()
        state["cls_queries"] = cls_query
    return state


def load_value_model_from_checkpoint(
    *,
    llada_path: str,
    value_model_path: str,
    device: str,
    torch_dtype: torch.dtype,
    args: Any,
    mask_id: int | None = None,
    strict: bool = True,
) -> tuple[ValueModel, dict[str, Any]]:
    resolved = resolve_value_model_init_kwargs(
        args,
        value_model_path=value_model_path,
        mask_id=mask_id,
    )
    model = ValueModel.from_pretrained(
        llada_path,
        n_layers=resolved["n_layers"],
        device=device,
        torch_dtype=torch_dtype,
        mask_id=resolved["mask_id"],
        architecture=resolved["value_model_arch"],
        readout_heads=resolved["readout_heads"],
        pooling_mode=resolved["pooling_mode"],
        readout_num_queries=resolved["readout_num_queries"],
        readout_aggregation=resolved["readout_aggregation"],
    )
    state = torch.load(resolve_project_path(value_model_path), map_location=device)
    state = upgrade_value_model_state_dict(
        state,
        readout_num_queries=resolved["readout_num_queries"],
    )
    model.load_state_dict(state, strict=strict)
    model.eval()
    return model, resolved


def apply_value_model_runtime_args(args: Any, resolved: dict[str, Any]) -> Any:
    for key, value in resolved.items():
        setattr(args, key, value)
    return args
