"""Lower-cost counterfactual-safe decoding for readout-CLS value models.

This module keeps the public decode interface close to
``counterfactual_safe_decode`` while making the cached readout path explicit:

1. Let the generator propose the next partially denoised state.
2. For readout-CLS value models, encode all proposed rows once as a batch.
3. Score Base/Drop/Hold views by reusing those encoded token features.

The checkpoint remains the regular ``readout_cls`` checkpoint.  This mode does
not reuse generator hidden states; doing that would change the feature
distribution unless the value readout were trained for those features.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict

import torch

from utils.counterfactual_safe_steering import (
    CounterfactualSafeSteeringConfig,
    _counterfactual_safe_remask_row,
    _propose_new_tokens,
)


def _empty_meta() -> Dict[str, Any]:
    return {
        "base_safe_score": None,
        "current_remask_threshold": None,
        "necessity_unsafe": [],
        "sufficiency_unsafe": [],
        "unsafe_score": [],
        "uncertainty_score": [],
        "remask_score": [],
        "remasked_positions": [],
    }


def _current_threshold(
    cfg: CounterfactualSafeSteeringConfig,
    *,
    step_idx: int,
    total_steps: int,
) -> float:
    if bool(getattr(cfg, "fixed_remask_threshold", False)) or not bool(
        getattr(cfg, "use_linear_threshold_decay", True)
    ):
        return float(cfg.remask_threshold)
    progression = float(step_idx) / float(max(total_steps, 1))
    return max(
        float(cfg.remask_threshold) * (1.0 - progression),
        float(getattr(cfg, "remask_threshold_floor", 0.0)),
    )


def _compute_t_batch_per_row(
    states: torch.Tensor,
    total_initial_masks: torch.Tensor,
    mask_id: int,
    device: str,
    time_feature_mode: str,
) -> torch.Tensor:
    denom = total_initial_masks.to(device=states.device, dtype=torch.float32).clamp_min(1.0)
    mask_counts = states.eq(int(mask_id)).sum(dim=1).to(torch.float32)
    remaining_mask_ratio = mask_counts / denom
    mode = str(time_feature_mode).strip().lower()
    if mode == "remaining_mask_ratio":
        t_batch = remaining_mask_ratio.unsqueeze(-1)
    elif mode == "progress_ratio":
        t_batch = (1.0 - remaining_mask_ratio).unsqueeze(-1)
    else:
        raise ValueError(
            "Unknown time_feature_mode: "
            f"{time_feature_mode!r}. Expected 'remaining_mask_ratio' or 'progress_ratio'."
        )
    return t_batch.to(device)


def _build_cls_views(
    proposed_row: torch.Tensor,
    attention_mask_row: torch.Tensor,
    audit_positions: torch.Tensor,
    mask_id: int,
) -> torch.Tensor:
    visible_content = proposed_row.ne(int(mask_id)) & attention_mask_row.bool()
    cls_views = [visible_content.clone()]
    for pos in audit_positions.tolist():
        dropped = visible_content.clone()
        dropped[0, int(pos)] = False
        cls_views.append(dropped)
    for pos in audit_positions.tolist():
        held = torch.zeros_like(visible_content, dtype=torch.bool)
        held[0, int(pos)] = True
        cls_views.append(held)
    return torch.cat(cls_views, dim=0)


def _remask_from_scores(
    proposed_row: torch.Tensor,
    audit_positions: torch.Tensor,
    token_probs_row: torch.Tensor,
    scores: torch.Tensor,
    cfg: CounterfactualSafeSteeringConfig,
    *,
    current_remask_threshold: float,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    n_audit = int(audit_positions.numel())
    base_safe = scores[0]
    base_unsafe = 1.0 - base_safe
    drop_unsafe = 1.0 - scores[1 : 1 + n_audit]
    hold_unsafe = 1.0 - scores[1 + n_audit :]

    necessity_unsafe = torch.clamp(base_unsafe - drop_unsafe, min=0.0, max=1.0)
    sufficiency_unsafe = torch.clamp(hold_unsafe, min=0.0, max=1.0)
    unsafe_score = (
        float(cfg.necessity_weight) * necessity_unsafe
        + float(cfg.sufficiency_weight) * sufficiency_unsafe
    )

    audit_positions = audit_positions.to(device=unsafe_score.device)
    token_probs_row = token_probs_row.to(device=unsafe_score.device, dtype=unsafe_score.dtype)
    uncertainty_score = torch.clamp(
        1.0 - token_probs_row.index_select(0, audit_positions),
        min=0.0,
        max=1.0,
    )
    remask_score = (
        float(cfg.unsafe_weight) * unsafe_score
        + float(cfg.uncertainty_weight) * uncertainty_score
    )

    candidate_order = torch.argsort(remask_score, descending=True)
    candidate_scores = remask_score.index_select(0, candidate_order)
    eligible = candidate_order[candidate_scores >= float(current_remask_threshold)]

    max_remask = int(math.ceil(float(n_audit) * max(float(cfg.max_remask_ratio), 0.0)))
    max_remask = min(max_remask, max(n_audit - int(cfg.min_keep_per_step), 0))
    if max_remask <= 0 or eligible.numel() == 0:
        return proposed_row, {
            "base_safe_score": float(base_safe.item()),
            "current_remask_threshold": float(current_remask_threshold),
            "necessity_unsafe": necessity_unsafe.detach().cpu().tolist(),
            "sufficiency_unsafe": sufficiency_unsafe.detach().cpu().tolist(),
            "unsafe_score": unsafe_score.detach().cpu().tolist(),
            "uncertainty_score": uncertainty_score.detach().cpu().tolist(),
            "remask_score": remask_score.detach().cpu().tolist(),
            "remasked_positions": [],
        }

    chosen = eligible[:max_remask]
    remask_positions = audit_positions.index_select(0, chosen)
    out = proposed_row.clone()
    out[0, remask_positions.to(device=out.device)] = int(cfg.mask_id)
    return out, {
        "base_safe_score": float(base_safe.item()),
        "current_remask_threshold": float(current_remask_threshold),
        "necessity_unsafe": necessity_unsafe.detach().cpu().tolist(),
        "sufficiency_unsafe": sufficiency_unsafe.detach().cpu().tolist(),
        "unsafe_score": unsafe_score.detach().cpu().tolist(),
        "uncertainty_score": uncertainty_score.detach().cpu().tolist(),
        "remask_score": remask_score.detach().cpu().tolist(),
        "remasked_positions": remask_positions.detach().cpu().tolist(),
    }


@torch.no_grad()
def _counterfactual_safe_remask_batch_low_cost(
    val_model,
    proposed_x: torch.Tensor,
    attention_mask: torch.Tensor,
    audit_indices_by_row: list[torch.Tensor],
    token_probs: torch.Tensor,
    total_initial_masks: torch.Tensor,
    device_val: str,
    cfg: CounterfactualSafeSteeringConfig,
    remaining_steps_after: int,
    step_idx: int,
    total_steps: int,
) -> tuple[torch.Tensor, list[Dict[str, Any]]]:
    """Batch-cache value features, then score each row's counterfactual views."""
    safe_rows = [proposed_x[row : row + 1] for row in range(proposed_x.shape[0])]
    metas = [_empty_meta() for _ in range(proposed_x.shape[0])]

    active_rows: list[int] = []
    if remaining_steps_after > 0:
        for row, audit_positions in enumerate(audit_indices_by_row):
            if audit_positions.numel() > int(cfg.min_keep_per_step):
                active_rows.append(row)

    if not active_rows:
        return torch.cat(safe_rows, dim=0), metas

    supports_cached = (
        hasattr(val_model, "encode_tokens")
        and hasattr(val_model, "score_encoded")
        and bool(getattr(val_model, "supports_cached_counterfactual", False))
    )
    if not supports_cached:
        for row in active_rows:
            safe_rows[row], metas[row] = _counterfactual_safe_remask_row(
                val_model=val_model,
                proposed_row=proposed_x[row : row + 1],
                attention_mask_row=attention_mask[row : row + 1],
                new_positions=audit_indices_by_row[row],
                token_probs_row=token_probs[row],
                total_initial_masks=int(total_initial_masks[row].item()),
                device_val=device_val,
                cfg=cfg,
                remaining_steps_after=remaining_steps_after,
                step_idx=step_idx,
                total_steps=total_steps,
            )
        return torch.cat(safe_rows, dim=0), metas

    row_idx = torch.tensor(active_rows, dtype=torch.long, device=proposed_x.device)
    states = proposed_x.index_select(0, row_idx).to(device_val)
    att = attention_mask.index_select(0, row_idx).to(device_val)
    initial_masks = total_initial_masks.index_select(0, row_idx).to(device_val)
    t_batch = _compute_t_batch_per_row(
        states=states,
        total_initial_masks=initial_masks,
        mask_id=cfg.mask_id,
        device=device_val,
        time_feature_mode=cfg.time_feature_mode,
    )
    encoded = val_model.encode_tokens(input_ids=states, t=t_batch, attention_mask=att)
    threshold = _current_threshold(cfg, step_idx=step_idx, total_steps=total_steps)

    for local_idx, row in enumerate(active_rows):
        cls_attend_to = _build_cls_views(
            proposed_row=proposed_x[row : row + 1],
            attention_mask_row=attention_mask[row : row + 1],
            audit_positions=audit_indices_by_row[row],
            mask_id=cfg.mask_id,
        ).to(device_val)
        view_count = int(cls_attend_to.shape[0])
        row_att = att[local_idx : local_idx + 1].expand(view_count, -1)
        row_input = states[local_idx : local_idx + 1]
        scores = val_model.score_encoded(
            encoded_tokens=encoded[local_idx : local_idx + 1],
            t=t_batch[local_idx : local_idx + 1],
            attention_mask=row_att,
            cls_attend_to=cls_attend_to,
            input_ids=row_input,
        ).squeeze(-1)
        safe_rows[row], metas[row] = _remask_from_scores(
            proposed_row=proposed_x[row : row + 1],
            audit_positions=audit_indices_by_row[row],
            token_probs_row=token_probs[row],
            scores=scores.to(dtype=torch.float32, device=proposed_x.device),
            cfg=cfg,
            current_remask_threshold=threshold,
        )

    return torch.cat(safe_rows, dim=0), metas


@torch.no_grad()
def low_cost_counterfactual_safe_decode(
    model,
    val_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device_gen: str,
    device_val: str,
    cfg: CounterfactualSafeSteeringConfig,
    prompt_length: int | None = None,
    editable_mask: torch.Tensor | None = None,
    block_length: int | None = None,
    logits_eos_inf: bool = False,
    confidence_eos_eot_inf: bool = False,
    eos_token_id: int = 126081,
    eot_token_id: int = 126348,
    stop_checker: Callable[[torch.Tensor], bool] | None = None,
    trace_callback: Callable[[Dict[str, Any]], None] | None = None,
) -> torch.Tensor:
    """Decode with batched cached-readout value scoring.

    This is a drop-in alternative to ``counterfactual_safe_decode`` for
    ``readout_cls`` checkpoints.  For non-cached value models it falls back to
    the row-wise remask implementation.
    """
    def emit_trace(**event: Any) -> None:
        if trace_callback is not None:
            trace_callback(event)

    x = input_ids.clone().to(device_gen)
    att = attention_mask.to(device_gen)
    total_initial_masks = (x == int(cfg.mask_id)).sum(dim=1).clamp_min(1)
    suppressed_token_ids = torch.full_like(x, fill_value=-1, dtype=torch.long, device=device_gen)
    suppressed_active = torch.zeros_like(x, dtype=torch.bool, device=device_gen)

    if editable_mask is not None:
        if editable_mask.shape != x.shape:
            raise ValueError(
                "editable_mask must match input_ids shape. "
                f"Got {tuple(editable_mask.shape)} vs {tuple(x.shape)}."
            )
        editable_mask = editable_mask.to(device=device_gen, dtype=torch.bool)
    elif prompt_length is not None:
        editable_mask = torch.zeros_like(x, dtype=torch.bool)
        editable_mask[:, prompt_length:] = att[:, prompt_length:].bool()
    else:
        editable_mask = (input_ids == int(cfg.mask_id)).to(device_gen)

    emit_trace(
        event_type="init",
        step_index=None,
        prompt_length=prompt_length,
        total_length=int(x.shape[1]),
        masks_remaining=[int(v) for v in (x == int(cfg.mask_id)).sum(dim=1).detach().cpu().tolist()],
        state_ids=x.detach().clone().cpu(),
        low_cost_mode=True,
    )

    response_positions = torch.nonzero(editable_mask[0], as_tuple=False).squeeze(-1)
    if response_positions.numel() == 0:
        block_ranges = [(0, x.shape[1])]
    else:
        response_start = int(response_positions.min().item())
        response_len = int(response_positions.numel())
        block_len = int(block_length or response_len)
        if block_len <= 0:
            raise ValueError(f"block_length must be positive, got {block_length!r}")
        if response_len % block_len != 0:
            raise ValueError(f"editable response length {response_len} must be divisible by block_length {block_len}")
        block_ranges = [
            (response_start + i * block_len, response_start + (i + 1) * block_len)
            for i in range(response_len // block_len)
        ]

    if int(cfg.steps) % len(block_ranges) != 0:
        raise ValueError(f"steps {cfg.steps} must be divisible by number of blocks {len(block_ranges)}")
    steps_per_block = max(int(cfg.steps) // len(block_ranges), 1)

    for block_idx, (block_start, block_end) in enumerate(block_ranges):
        block_mask = torch.zeros_like(x, dtype=torch.bool)
        block_mask[:, block_start:block_end] = True
        for inner_step in range(steps_per_block):
            global_step = block_idx * steps_per_block + inner_step
            mask_index = (x == int(cfg.mask_id)) & block_mask
            if not mask_index.any():
                break
            suppressed_active &= x == int(cfg.mask_id)

            out = model(x, attention_mask=att)
            if float(cfg.suppression_value) > 0.0:
                rows, cols = torch.nonzero(suppressed_active, as_tuple=True)
                if rows.numel() > 0:
                    tok_ids = suppressed_token_ids[rows, cols]
                    valid = tok_ids >= 0
                    if valid.any():
                        out.logits[rows[valid], cols[valid], tok_ids[valid]] -= float(cfg.suppression_value)

            proposed_x, transfer_index, token_probs = _propose_new_tokens(
                x=x,
                logits=out.logits,
                mask_index=mask_index,
                step_idx=inner_step,
                total_steps=steps_per_block,
                temperature=cfg.proposal_temperature,
                logits_eos_inf=logits_eos_inf,
                confidence_eos_eot_inf=confidence_eos_eot_inf,
                eos_token_id=eos_token_id,
                eot_token_id=eot_token_id,
            )
            if not transfer_index.any():
                break

            active_editable_mask = editable_mask & block_mask
            audit_indices_by_row: list[torch.Tensor] = []
            fresh_indices_by_row: list[torch.Tensor] = []
            for row in range(proposed_x.shape[0]):
                proposed_row = proposed_x[row : row + 1]
                editable_row = active_editable_mask[row]
                fresh_positions = torch.nonzero(
                    transfer_index[row] & editable_row,
                    as_tuple=False,
                ).squeeze(-1)
                is_filled = (proposed_row[0] != int(cfg.mask_id)) & editable_row
                historical_mask = is_filled.clone()
                if fresh_positions.numel() > 0:
                    historical_mask[fresh_positions] = False
                historical_indices = torch.nonzero(historical_mask, as_tuple=False).squeeze(-1)

                audit_indices = fresh_positions
                if historical_indices.numel() > 0 and int(cfg.historical_audit_topk) > 0:
                    hist_uncertainty = 1.0 - token_probs[row].index_select(0, historical_indices)
                    n_audit_hist = min(int(cfg.historical_audit_topk), int(historical_indices.numel()))
                    _, top_hist_idx = torch.topk(hist_uncertainty, k=n_audit_hist)
                    audit_indices = torch.cat([fresh_positions, historical_indices.index_select(0, top_hist_idx)])

                fresh_indices_by_row.append(fresh_positions)
                audit_indices_by_row.append(audit_indices)

            remaining_steps_after = int(cfg.steps) - global_step - 1
            x, audit_metas = _counterfactual_safe_remask_batch_low_cost(
                val_model=val_model,
                proposed_x=proposed_x,
                attention_mask=att,
                audit_indices_by_row=audit_indices_by_row,
                token_probs=token_probs,
                total_initial_masks=total_initial_masks,
                device_val=device_val,
                cfg=cfg,
                remaining_steps_after=remaining_steps_after,
                step_idx=global_step,
                total_steps=int(cfg.steps),
            )

            for row, audit_meta in enumerate(audit_metas):
                emit_trace(
                    event_type="step",
                    step_index=int(global_step),
                    batch_index=int(row),
                    prompt_length=prompt_length,
                    proposal_state_ids=proposed_x[row].detach().clone().cpu(),
                    steered_state_ids=x[row].detach().clone().cpu(),
                    fresh_positions=fresh_indices_by_row[row].detach().cpu().tolist(),
                    audit_positions=audit_indices_by_row[row].detach().cpu().tolist(),
                    active_suppressed_positions=torch.nonzero(
                        suppressed_active[row],
                        as_tuple=False,
                    ).squeeze(-1).detach().cpu().tolist(),
                    active_suppressed_token_ids=suppressed_token_ids[
                        row,
                        suppressed_active[row],
                    ].detach().cpu().tolist(),
                    proposal_masks_remaining=int((proposed_x[row] == int(cfg.mask_id)).sum().item()),
                    steered_masks_remaining=int((x[row] == int(cfg.mask_id)).sum().item()),
                    low_cost_mode=True,
                    **audit_meta,
                )

                if audit_meta.get("remasked_positions"):
                    remask_pos = torch.tensor(
                        audit_meta["remasked_positions"],
                        dtype=torch.long,
                        device=device_gen,
                    )
                    suppressed_token_ids[row, remask_pos] = proposed_x[row, remask_pos].to(torch.long)
                    suppressed_active[row, remask_pos] = True

            if stop_checker is not None and stop_checker(x):
                break

    emit_trace(
        event_type="final",
        step_index=None,
        prompt_length=prompt_length,
        total_length=int(x.shape[1]),
        masks_remaining=[int(v) for v in (x == int(cfg.mask_id)).sum(dim=1).detach().cpu().tolist()],
        state_ids=x.detach().clone().cpu(),
        low_cost_mode=True,
    )
    return x


counterfactual_safe_decode_low_cost = low_cost_counterfactual_safe_decode


def _infer_value_feature_layer(val_model, feature_layer: int | None) -> int:
    if feature_layer is not None:
        return int(feature_layer)
    transformer = getattr(getattr(val_model, "llada", None), "transformer", None)
    blocks = getattr(transformer, "blocks", None)
    if blocks is not None:
        return int(len(blocks))
    block_groups = getattr(transformer, "block_groups", None)
    if block_groups is not None:
        return int(sum(len(group) for group in block_groups))
    raise ValueError("generator_feature_layer is required when the value model layer count cannot be inferred")


def _get_hidden_state_at_layer(model_output, layer_idx: int) -> torch.Tensor:
    hidden_states = getattr(model_output, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError(
            "Generator did not return hidden_states. "
            "Call with output_hidden_states=True, return_dict=True."
        )
    if layer_idx < 0 or layer_idx >= len(hidden_states):
        raise ValueError(
            f"generator_feature_layer={layer_idx} is out of range for "
            f"{len(hidden_states)} returned hidden-state tensors."
        )
    return hidden_states[layer_idx]


def _generator_hidden_to_value_features(
    val_model,
    raw_hidden: torch.Tensor,
    t_batch: torch.Tensor,
    *,
    device_val: str,
    time_injection: str,
) -> torch.Tensor:
    hidden = raw_hidden.to(device_val)
    mode = str(time_injection or "add_then_ln").strip().lower()
    if mode in {"add_then_ln", "pre_ln", "preln"}:
        time_query = val_model._compute_time_query(  # noqa: SLF001 - experimental inference path.
            t_batch.to(device_val),
            hidden.shape[0],
            hidden.dtype,
            hidden.device,
        )
        return val_model.llada.transformer.ln_f(hidden + time_query)
    if mode in {"ln_then_add", "post_ln", "postln"}:
        hidden = val_model.llada.transformer.ln_f(hidden)
        time_query = val_model._compute_time_query(  # noqa: SLF001 - experimental inference path.
            t_batch.to(device_val),
            hidden.shape[0],
            hidden.dtype,
            hidden.device,
        )
        return hidden + time_query
    if mode in {"none", "no_time", "off"}:
        return val_model.llada.transformer.ln_f(hidden)
    raise ValueError(
        f"Unknown generator_feature_time_injection={time_injection!r}. "
        "Expected add_then_ln, ln_then_add, or none."
    )


@torch.no_grad()
def _counterfactual_safe_remask_batch_from_encoded(
    val_model,
    proposed_x: torch.Tensor,
    attention_mask: torch.Tensor,
    audit_indices_by_row: list[torch.Tensor],
    token_probs: torch.Tensor,
    encoded: torch.Tensor,
    t_batch: torch.Tensor,
    active_rows: list[int],
    device_val: str,
    cfg: CounterfactualSafeSteeringConfig,
    *,
    step_idx: int,
    total_steps: int,
) -> tuple[torch.Tensor, list[Dict[str, Any]]]:
    safe_rows = [proposed_x[row : row + 1] for row in range(proposed_x.shape[0])]
    metas = [_empty_meta() for _ in range(proposed_x.shape[0])]
    if not active_rows:
        return torch.cat(safe_rows, dim=0), metas

    row_idx = torch.tensor(active_rows, dtype=torch.long, device=proposed_x.device)
    states = proposed_x.index_select(0, row_idx).to(device_val)
    att = attention_mask.index_select(0, row_idx).to(device_val)
    threshold = _current_threshold(cfg, step_idx=step_idx, total_steps=total_steps)

    for local_idx, row in enumerate(active_rows):
        cls_attend_to = _build_cls_views(
            proposed_row=proposed_x[row : row + 1],
            attention_mask_row=attention_mask[row : row + 1],
            audit_positions=audit_indices_by_row[row],
            mask_id=cfg.mask_id,
        ).to(device_val)
        view_count = int(cls_attend_to.shape[0])
        scores = val_model.score_encoded(
            encoded_tokens=encoded[local_idx : local_idx + 1],
            t=t_batch[local_idx : local_idx + 1],
            attention_mask=att[local_idx : local_idx + 1].expand(view_count, -1),
            cls_attend_to=cls_attend_to,
            input_ids=states[local_idx : local_idx + 1],
        ).squeeze(-1)
        safe_rows[row], metas[row] = _remask_from_scores(
            proposed_row=proposed_x[row : row + 1],
            audit_positions=audit_indices_by_row[row],
            token_probs_row=token_probs[row],
            scores=scores.to(dtype=torch.float32, device=proposed_x.device),
            cfg=cfg,
            current_remask_threshold=threshold,
        )

    return torch.cat(safe_rows, dim=0), metas


@torch.no_grad()
def generator_feature_counterfactual_safe_decode(
    model,
    val_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device_gen: str,
    device_val: str,
    cfg: CounterfactualSafeSteeringConfig,
    prompt_length: int | None = None,
    editable_mask: torch.Tensor | None = None,
    block_length: int | None = None,
    logits_eos_inf: bool = False,
    confidence_eos_eot_inf: bool = False,
    eos_token_id: int = 126081,
    eot_token_id: int = 126348,
    stop_checker: Callable[[Dict[str, Any]], bool] | None = None,
    trace_callback: Callable[[Dict[str, Any]], None] | None = None,
    generator_feature_layer: int | None = None,
    generator_feature_time_injection: str = "add_then_ln",
    generator_feature_stale: bool = False,
) -> torch.Tensor:
    """Experimental decode path that scores readout-CLS views from generator hidden states.

    By default this takes hidden states from an extra generator forward on
    ``proposed_x`` so the features correspond to the state being audited.  Set
    ``generator_feature_stale=True`` to reuse the pre-proposal hidden states
    from the logits forward; that is cheaper but intentionally approximate.
    """
    def emit_trace(**event: Any) -> None:
        if trace_callback is not None:
            trace_callback(event)

    if not hasattr(val_model, "score_encoded"):
        raise RuntimeError("generator_feature_counterfactual_safe_decode requires a readout-style value model")

    feature_layer = _infer_value_feature_layer(val_model, generator_feature_layer)
    x = input_ids.clone().to(device_gen)
    att = attention_mask.to(device_gen)
    total_initial_masks = (x == int(cfg.mask_id)).sum(dim=1).clamp_min(1)
    suppressed_token_ids = torch.full_like(x, fill_value=-1, dtype=torch.long, device=device_gen)
    suppressed_active = torch.zeros_like(x, dtype=torch.bool, device=device_gen)

    if editable_mask is not None:
        if editable_mask.shape != x.shape:
            raise ValueError(
                "editable_mask must match input_ids shape. "
                f"Got {tuple(editable_mask.shape)} vs {tuple(x.shape)}."
            )
        editable_mask = editable_mask.to(device=device_gen, dtype=torch.bool)
    elif prompt_length is not None:
        editable_mask = torch.zeros_like(x, dtype=torch.bool)
        editable_mask[:, prompt_length:] = att[:, prompt_length:].bool()
    else:
        editable_mask = (input_ids == int(cfg.mask_id)).to(device_gen)

    emit_trace(
        event_type="init",
        step_index=None,
        prompt_length=prompt_length,
        total_length=int(x.shape[1]),
        masks_remaining=[int(v) for v in (x == int(cfg.mask_id)).sum(dim=1).detach().cpu().tolist()],
        state_ids=x.detach().clone().cpu(),
        generator_feature_mode=True,
        generator_feature_layer=int(feature_layer),
        generator_feature_time_injection=generator_feature_time_injection,
        generator_feature_stale=bool(generator_feature_stale),
    )

    response_positions = torch.nonzero(editable_mask[0], as_tuple=False).squeeze(-1)
    if response_positions.numel() == 0:
        block_ranges = [(0, x.shape[1])]
    else:
        response_start = int(response_positions.min().item())
        response_len = int(response_positions.numel())
        block_len = int(block_length or response_len)
        if block_len <= 0:
            raise ValueError(f"block_length must be positive, got {block_length!r}")
        if response_len % block_len != 0:
            raise ValueError(f"editable response length {response_len} must be divisible by block_length {block_len}")
        block_ranges = [
            (response_start + i * block_len, response_start + (i + 1) * block_len)
            for i in range(response_len // block_len)
        ]

    if int(cfg.steps) % len(block_ranges) != 0:
        raise ValueError(f"steps {cfg.steps} must be divisible by number of blocks {len(block_ranges)}")
    steps_per_block = max(int(cfg.steps) // len(block_ranges), 1)

    for block_idx, (block_start, block_end) in enumerate(block_ranges):
        block_mask = torch.zeros_like(x, dtype=torch.bool)
        block_mask[:, block_start:block_end] = True
        for inner_step in range(steps_per_block):
            global_step = block_idx * steps_per_block + inner_step
            mask_index = (x == int(cfg.mask_id)) & block_mask
            if not mask_index.any():
                break
            suppressed_active &= x == int(cfg.mask_id)

            out = model(
                x,
                attention_mask=att,
                output_hidden_states=bool(generator_feature_stale),
                return_dict=True,
            )
            if float(cfg.suppression_value) > 0.0:
                rows, cols = torch.nonzero(suppressed_active, as_tuple=True)
                if rows.numel() > 0:
                    tok_ids = suppressed_token_ids[rows, cols]
                    valid = tok_ids >= 0
                    if valid.any():
                        out.logits[rows[valid], cols[valid], tok_ids[valid]] -= float(cfg.suppression_value)

            proposed_x, transfer_index, token_probs = _propose_new_tokens(
                x=x,
                logits=out.logits,
                mask_index=mask_index,
                step_idx=inner_step,
                total_steps=steps_per_block,
                temperature=cfg.proposal_temperature,
                logits_eos_inf=logits_eos_inf,
                confidence_eos_eot_inf=confidence_eos_eot_inf,
                eos_token_id=eos_token_id,
                eot_token_id=eot_token_id,
            )
            if not transfer_index.any():
                break

            active_editable_mask = editable_mask & block_mask
            audit_indices_by_row: list[torch.Tensor] = []
            fresh_indices_by_row: list[torch.Tensor] = []
            active_rows: list[int] = []
            if int(cfg.steps) - global_step - 1 > 0:
                for row in range(proposed_x.shape[0]):
                    proposed_row = proposed_x[row : row + 1]
                    editable_row = active_editable_mask[row]
                    fresh_positions = torch.nonzero(
                        transfer_index[row] & editable_row,
                        as_tuple=False,
                    ).squeeze(-1)
                    is_filled = (proposed_row[0] != int(cfg.mask_id)) & editable_row
                    historical_mask = is_filled.clone()
                    if fresh_positions.numel() > 0:
                        historical_mask[fresh_positions] = False
                    historical_indices = torch.nonzero(historical_mask, as_tuple=False).squeeze(-1)

                    audit_indices = fresh_positions
                    if historical_indices.numel() > 0 and int(cfg.historical_audit_topk) > 0:
                        hist_uncertainty = 1.0 - token_probs[row].index_select(0, historical_indices)
                        n_audit_hist = min(int(cfg.historical_audit_topk), int(historical_indices.numel()))
                        _, top_hist_idx = torch.topk(hist_uncertainty, k=n_audit_hist)
                        audit_indices = torch.cat([fresh_positions, historical_indices.index_select(0, top_hist_idx)])

                    fresh_indices_by_row.append(fresh_positions)
                    audit_indices_by_row.append(audit_indices)
                    if audit_indices.numel() > int(cfg.min_keep_per_step):
                        active_rows.append(row)
            else:
                for row in range(proposed_x.shape[0]):
                    fresh = torch.nonzero(transfer_index[row] & active_editable_mask[row], as_tuple=False).squeeze(-1)
                    fresh_indices_by_row.append(fresh)
                    audit_indices_by_row.append(fresh)

            if active_rows:
                row_idx = torch.tensor(active_rows, dtype=torch.long, device=proposed_x.device)
                feature_states = x.index_select(0, row_idx) if generator_feature_stale else proposed_x.index_select(0, row_idx)
                feature_att = att.index_select(0, row_idx)
                feature_initial_masks = total_initial_masks.index_select(0, row_idx).to(device_val)
                t_batch = _compute_t_batch_per_row(
                    states=proposed_x.index_select(0, row_idx),
                    total_initial_masks=feature_initial_masks,
                    mask_id=cfg.mask_id,
                    device=device_val,
                    time_feature_mode=cfg.time_feature_mode,
                )
                if generator_feature_stale:
                    feature_output = out
                    raw_hidden = _get_hidden_state_at_layer(feature_output, feature_layer).index_select(0, row_idx)
                else:
                    feature_output = model(
                        feature_states,
                        attention_mask=feature_att,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    raw_hidden = _get_hidden_state_at_layer(feature_output, feature_layer)

                encoded = _generator_hidden_to_value_features(
                    val_model,
                    raw_hidden,
                    t_batch,
                    device_val=device_val,
                    time_injection=generator_feature_time_injection,
                )
                x, audit_metas = _counterfactual_safe_remask_batch_from_encoded(
                    val_model=val_model,
                    proposed_x=proposed_x,
                    attention_mask=att,
                    audit_indices_by_row=audit_indices_by_row,
                    token_probs=token_probs,
                    encoded=encoded,
                    t_batch=t_batch,
                    active_rows=active_rows,
                    device_val=device_val,
                    cfg=cfg,
                    step_idx=global_step,
                    total_steps=int(cfg.steps),
                )
            else:
                x = proposed_x
                audit_metas = [_empty_meta() for _ in range(proposed_x.shape[0])]

            for row, audit_meta in enumerate(audit_metas):
                emit_trace(
                    event_type="step",
                    step_index=int(global_step),
                    batch_index=int(row),
                    prompt_length=prompt_length,
                    proposal_state_ids=proposed_x[row].detach().clone().cpu(),
                    steered_state_ids=x[row].detach().clone().cpu(),
                    fresh_positions=fresh_indices_by_row[row].detach().cpu().tolist(),
                    audit_positions=audit_indices_by_row[row].detach().cpu().tolist(),
                    active_suppressed_positions=torch.nonzero(
                        suppressed_active[row],
                        as_tuple=False,
                    ).squeeze(-1).detach().cpu().tolist(),
                    active_suppressed_token_ids=suppressed_token_ids[
                        row,
                        suppressed_active[row],
                    ].detach().cpu().tolist(),
                    proposal_masks_remaining=int((proposed_x[row] == int(cfg.mask_id)).sum().item()),
                    steered_masks_remaining=int((x[row] == int(cfg.mask_id)).sum().item()),
                    generator_feature_mode=True,
                    generator_feature_layer=int(feature_layer),
                    generator_feature_time_injection=generator_feature_time_injection,
                    generator_feature_stale=bool(generator_feature_stale),
                    **audit_meta,
                )

                if audit_meta.get("remasked_positions"):
                    remask_pos = torch.tensor(
                        audit_meta["remasked_positions"],
                        dtype=torch.long,
                        device=device_gen,
                    )
                    suppressed_token_ids[row, remask_pos] = proposed_x[row, remask_pos].to(torch.long)
                    suppressed_active[row, remask_pos] = True

            if stop_checker is not None and stop_checker(x):
                break

    emit_trace(
        event_type="final",
        step_index=None,
        prompt_length=prompt_length,
        total_length=int(x.shape[1]),
        masks_remaining=[int(v) for v in (x == int(cfg.mask_id)).sum(dim=1).detach().cpu().tolist()],
        state_ids=x.detach().clone().cpu(),
        generator_feature_mode=True,
        generator_feature_layer=int(feature_layer),
        generator_feature_time_injection=generator_feature_time_injection,
        generator_feature_stale=bool(generator_feature_stale),
    )
    return x


counterfactual_safe_decode_generator_feature = generator_feature_counterfactual_safe_decode
