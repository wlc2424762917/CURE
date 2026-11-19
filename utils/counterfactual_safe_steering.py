"""Counterfactual-safe decoding for masked diffusion-style generation.

The high-level loop is:
1. Let the base model propose a few new tokens for the current masked state.
2. Ask the value model whether those newly committed tokens look unsafe.
3. Build simple counterfactual variants by dropping or isolating candidate tokens.
4. Re-mask only the suspicious positions so later denoising steps can rewrite them.
"""

import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict

import torch
import torch.nn.functional as F

from utils.generate_function_llada import add_gumbel_noise, get_num_transfer_tokens


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(value, minimum)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class CounterfactualSafeSteeringConfig:
    steps: int = 64
    gen_length: int = 128
    mask_id: int = 126336
    time_feature_mode: str = "progress_ratio"
    proposal_temperature: float = 0.8
    unsafe_weight: float = 0.7
    uncertainty_weight: float = 0.3
    necessity_weight: float = 0.5
    sufficiency_weight: float = 0.5
    remask_threshold: float = 0.55
    max_remask_ratio: float = 0.5
    min_keep_per_step: int = 1
    suppression_value: float = 0.0
    historical_audit_topk: int = 8
    use_linear_threshold_decay: bool = True
    remask_threshold_floor: float = 0.0
    fixed_remask_threshold: bool = False
    block_length: int = 0
    logits_eos_inf: bool = False
    confidence_eos_eot_inf: bool = False
    eos_token_id: int = 126081
    eot_token_id: int = 126348

    @classmethod
    def from_env(cls, steps: int = 64, gen_length: int = 128, mask_id: int = 126336):
        return cls(
            steps=_env_int("CF_SAFE_STEPS", steps, minimum=1),
            gen_length=_env_int("CF_SAFE_GEN_LENGTH", gen_length, minimum=1),
            mask_id=_env_int("CF_SAFE_MASK_ID", mask_id, minimum=1),
            time_feature_mode=(os.getenv("CF_SAFE_TIME_FEATURE_MODE", "progress_ratio").strip() or "progress_ratio"),
            proposal_temperature=_env_float("CF_SAFE_PROPOSAL_TEMPERATURE", 0.8),
            unsafe_weight=_env_float("CF_SAFE_UNSAFE_WEIGHT", 0.7),
            uncertainty_weight=_env_float("CF_SAFE_UNCERTAINTY_WEIGHT", 0.3),
            necessity_weight=_env_float("CF_SAFE_NECESSITY_WEIGHT", 0.5),
            sufficiency_weight=_env_float("CF_SAFE_SUFFICIENCY_WEIGHT", 0.5),
            remask_threshold=_env_float("CF_SAFE_REMASK_THRESHOLD", 0.55),
            max_remask_ratio=_env_float("CF_SAFE_MAX_REMASK_RATIO", 0.5),
            min_keep_per_step=_env_int("CF_SAFE_MIN_KEEP_PER_STEP", 1, minimum=0),
            suppression_value=_env_float("CF_SAFE_SUPPRESSION_VALUE", 0.0),
            historical_audit_topk=_env_int("CF_SAFE_HISTORICAL_AUDIT_TOPK", 8, minimum=0),
            use_linear_threshold_decay=_env_bool("CF_SAFE_USE_LINEAR_THRESHOLD_DECAY", True),
            remask_threshold_floor=_env_float("CF_SAFE_REMASK_THRESHOLD_FLOOR", 0.0),
            fixed_remask_threshold=_env_bool("CF_SAFE_FIXED_REMASK_THRESHOLD", False),
            block_length=_env_int("CF_SAFE_BLOCK_LENGTH", 0, minimum=0),
            logits_eos_inf=_env_bool("CF_SAFE_LOGITS_EOS_INF", False),
            confidence_eos_eot_inf=_env_bool("CF_SAFE_CONFIDENCE_EOS_EOT_INF", False),
            eos_token_id=_env_int("CF_SAFE_EOS_TOKEN_ID", 126081, minimum=0),
            eot_token_id=_env_int("CF_SAFE_EOT_TOKEN_ID", 126348, minimum=0),
        )


def prepare_masked_chat_inputs(tokenizer, behavior: str, device: str, cfg: CounterfactualSafeSteeringConfig):
    """Build a chat prompt followed by a fully masked response suffix."""
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": behavior}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_with_masks = prompt + (tokenizer.decode([cfg.mask_id]) * cfg.gen_length)
    inputs = tokenizer(prompt_with_masks, return_tensors="pt")
    match_idx = len(tokenizer.encode(prompt, add_special_tokens=False))
    return (
        inputs["input_ids"].to(device),
        inputs["attention_mask"].to(device),
        match_idx,
    )


def _repeat_attention_mask(attention_mask: torch.Tensor, repeat_count: int, device: str) -> torch.Tensor:
    if attention_mask.shape[0] == repeat_count:
        return attention_mask.to(device)
    if attention_mask.shape[0] != 1:
        raise ValueError(
            "Counterfactual steering expects either a single-row attention mask or a "
            f"mask batch that already matches the repeated states. Got {attention_mask.shape}."
        )
    return attention_mask.repeat(repeat_count, 1).to(device)


def _compute_t_batch(
    states: torch.Tensor,
    total_initial_masks: int,
    mask_id: int,
    device: str,
    time_feature_mode: str = "progress_ratio",
) -> torch.Tensor:
    denom = max(int(total_initial_masks), 1)
    mask_counts = (states == mask_id).sum(dim=1).to(torch.float32)
    remaining_mask_ratio = mask_counts / float(denom)
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


@torch.no_grad()
def _value_scores(
    val_model,
    states: torch.Tensor,
    attention_mask: torch.Tensor,
    total_initial_masks: int,
    mask_id: int,
    device_val: str,
    cls_attend_to: torch.Tensor | None = None,
    time_feature_mode: str = "progress_ratio",
) -> torch.Tensor:
    """Evaluate one or more candidate states with the time-aware value model."""
    states_val = states.to(device_val)
    att_val = _repeat_attention_mask(attention_mask, states.shape[0], device_val)
    t_batch = _compute_t_batch(
        states,
        total_initial_masks,
        mask_id,
        device_val,
        time_feature_mode=time_feature_mode,
    )
    cls_attend_val = None
    if cls_attend_to is not None:
        if cls_attend_to.shape != states.shape:
            raise ValueError(
                "cls_attend_to must match states shape. "
                f"Got {tuple(cls_attend_to.shape)} vs {tuple(states.shape)}."
            )
        cls_attend_val = cls_attend_to.to(device_val)
    values = val_model(
        states_val,
        t_batch,
        attention_mask=att_val,
        cls_attend_to=cls_attend_val,
    )
    return values.squeeze(-1).to(torch.float32)


def _propose_new_tokens(
    x: torch.Tensor,
    logits: torch.Tensor,
    mask_index: torch.Tensor,
    step_idx: int,
    total_steps: int,
    temperature: float,
    logits_eos_inf: bool = False,
    confidence_eos_eot_inf: bool = False,
    eos_token_id: int = 126081,
    eot_token_id: int = 126348,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Commit the most confident masked positions for the current denoising step.

    This is the "proposal" phase of the defense:
    - sample a token proposal at every position
    - measure confidence only on currently masked positions
    - choose how many positions to fill based on how many denoising steps remain
    - write only those selected positions into the next state

    The output is intentionally provisional. Later in the same outer loop,
    counterfactual remasking may roll back some of these newly written tokens.
    """
    if logits_eos_inf:
        logits = logits.clone()
        logits[:, :, eos_token_id] = -torch.inf

    # Add stochasticity to avoid deterministically following a single trajectory.
    # This keeps the proposal stage compatible with diffusion-style decoding.
    logits_noisy = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_noisy, dim=-1)
    probs = F.softmax(logits_noisy.to(torch.float64), dim=-1)
    token_probs = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1).to(torch.float32)
    if confidence_eos_eot_inf:
        special = (x0 == eos_token_id) | (x0 == eot_token_id)
        token_probs = torch.where(special, torch.full_like(token_probs, -float("inf")), token_probs)
    # Confidence is only meaningful on positions that are still masked and need filling.
    confidence = torch.where(mask_index, token_probs, torch.full_like(token_probs, -float("inf")))

    remaining_steps = max(total_steps - step_idx, 1)
    # Recompute how many positions to fill given how many denoising steps remain.
    # Early in decoding we usually fill fewer tokens per step; later we can be more aggressive.
    dynamic_schedule = get_num_transfer_tokens(mask_index, remaining_steps)
    k_vec = dynamic_schedule[:, 0]

    transfer_index = torch.zeros_like(x, dtype=torch.bool)
    for row in range(x.shape[0]):
        # Clamp by the number of masks still available in this row.
        row_k = min(int(k_vec[row].item()), int(mask_index[row].sum().item()))
        if row_k <= 0:
            continue
        # Only the top-k most confident masked positions are committed this round.
        _, select_index = torch.topk(confidence[row], k=row_k)
        transfer_index[row, select_index] = True

    x_next = x.clone()
    # Write the proposal into the selected positions, leaving everything else unchanged.
    x_next[transfer_index] = x0[transfer_index]
    return x_next, transfer_index, token_probs


@torch.no_grad()
def _counterfactual_safe_remask_row(
    val_model,
    proposed_row: torch.Tensor,
    attention_mask_row: torch.Tensor,
    new_positions: torch.Tensor,
    token_probs_row: torch.Tensor,
    total_initial_masks: int,
    device_val: str,
    cfg: CounterfactualSafeSteeringConfig,
    remaining_steps_after: int,
    step_idx: int,
    total_steps: int,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Re-mask suspicious positions after a local counterfactual audit.

    For each audited position we score:
    - the full proposal
    - a "drop-one" state where that token is removed
    - a "hold-one" state where only that token is kept among audited positions

    This gives a cheap approximation of whether a token is necessary or sufficient
    for the unsafe trajectory according to the value model.
    """
    if new_positions.numel() == 0:
        return proposed_row, {
            "base_safe_score": None,
            "current_remask_threshold": None,
            "necessity_unsafe": [],
            "sufficiency_unsafe": [],
            "unsafe_score": [],
            "uncertainty_score": [],
            "remask_score": [],
            "remasked_positions": [],
        }

    if remaining_steps_after <= 0:
        return proposed_row, {
            "base_safe_score": None,
            "current_remask_threshold": None,
            "necessity_unsafe": [],
            "sufficiency_unsafe": [],
            "unsafe_score": [],
            "uncertainty_score": [],
            "remask_score": [],
            "remasked_positions": [],
        }

    if new_positions.numel() <= cfg.min_keep_per_step:
        return proposed_row, {
            "base_safe_score": None,
            "current_remask_threshold": None,
            "necessity_unsafe": [],
            "sufficiency_unsafe": [],
            "unsafe_score": [],
            "uncertainty_score": [],
            "remask_score": [],
            "remasked_positions": [],
        }

    if bool(getattr(cfg, "fixed_remask_threshold", False)) or not bool(
        getattr(cfg, "use_linear_threshold_decay", True)
    ):
        current_remask_threshold = float(cfg.remask_threshold)
    else:
        # Dynamic threshold: starts at cfg.remask_threshold, decreases to 0 near the end.
        # This makes the steering stricter as generation progresses.
        progression = float(step_idx) / float(max(total_steps, 1))
        current_remask_threshold = max(
            cfg.remask_threshold * (1.0 - progression),
            float(getattr(cfg, "remask_threshold_floor", 0.0)),
        )

    visible_content = (proposed_row != cfg.mask_id) & attention_mask_row.bool()

    # Counterfactual bundle:
    # - state[0]: base visibility (all non-mask content)
    # - next n states: drop each audited token from CLS visibility
    # - final n states: hold only one audited token in CLS visibility
    cls_views = [visible_content.clone()]
    for pos in new_positions.tolist():
        dropped = visible_content.clone()
        dropped[0, pos] = False
        cls_views.append(dropped)

    for keep_pos in new_positions.tolist():
        held = torch.zeros_like(visible_content, dtype=torch.bool)
        held[0, keep_pos] = True
        cls_views.append(held)

    cls_attend_batch = torch.cat(cls_views, dim=0)
    t_row = _compute_t_batch(
        states=proposed_row,
        total_initial_masks=total_initial_masks,
        mask_id=cfg.mask_id,
        device=device_val,
        time_feature_mode=cfg.time_feature_mode,
    )
    if hasattr(val_model, "score_masked_views"):
        scores = val_model.score_masked_views(
            input_ids=proposed_row.to(device_val),
            t=t_row,
            attention_mask=attention_mask_row.to(device_val),
            cls_attend_to=cls_attend_batch.to(device_val),
        ).to(dtype=torch.float32, device=proposed_row.device)
    else:
        state_batch = proposed_row.repeat(len(cls_views), 1)
        scores = _value_scores(
            val_model=val_model,
            states=state_batch,
            attention_mask=attention_mask_row,
            total_initial_masks=total_initial_masks,
            mask_id=cfg.mask_id,
            device_val=device_val,
            cls_attend_to=cls_attend_batch,
            time_feature_mode=cfg.time_feature_mode,
        ).to(device=proposed_row.device)

    n_new = int(new_positions.numel())
    base_safe = scores[0]
    base_unsafe = 1.0 - base_safe
    drop_unsafe = 1.0 - scores[1 : 1 + n_new]
    hold_unsafe = 1.0 - scores[1 + n_new :]

    # If dropping a token lowers unsafe score, that token was helping cause risk.
    necessity_unsafe = torch.clamp(base_unsafe - drop_unsafe, min=0.0, max=1.0)
    # If keeping only a token still leaves the state unsafe, that token alone is risky.
    sufficiency_unsafe = torch.clamp(hold_unsafe, min=0.0, max=1.0)
    unsafe_score = (
        cfg.necessity_weight * necessity_unsafe
        + cfg.sufficiency_weight * sufficiency_unsafe
    )

    token_probs_row = token_probs_row.to(device=unsafe_score.device, dtype=unsafe_score.dtype)
    new_positions = new_positions.to(device=unsafe_score.device)
    uncertainty_score = torch.clamp(1.0 - token_probs_row.index_select(0, new_positions), min=0.0, max=1.0)
    remask_score = (
        cfg.unsafe_weight * unsafe_score
        + cfg.uncertainty_weight * uncertainty_score
    )

    # Re-mask only the highest-risk positions, but never all of them at once.
    candidate_order = torch.argsort(remask_score, descending=True)
    candidate_scores = remask_score.index_select(0, candidate_order)
    eligible = candidate_order[candidate_scores >= current_remask_threshold]

    max_remask = int(math.ceil(float(n_new) * max(cfg.max_remask_ratio, 0.0)))
    max_remask = min(max_remask, max(n_new - cfg.min_keep_per_step, 0))
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
    remask_positions = new_positions.index_select(0, chosen)
    out = proposed_row.clone()
    out[0, remask_positions.to(device=out.device)] = cfg.mask_id
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
def counterfactual_safe_decode(
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
    """Decode by alternating between proposal and surgical counterfactual rollback."""
    def emit_trace(**event: Any) -> None:
        if trace_callback is None:
            return
        trace_callback(event)

    x = input_ids.clone().to(device_gen)
    att = attention_mask.to(device_gen)
    total_initial_masks = (x == cfg.mask_id).sum(dim=1).clamp_min(1)
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
        # Backward-compatible fallback: treat the suffix after prompt_length
        # as the editable response region.
        editable_mask = torch.zeros_like(x, dtype=torch.bool)
        editable_mask[:, prompt_length:] = att[:, prompt_length:].bool()
    else:
        # Default to the positions that were masked at initialization.
        # This lets already-generated response tokens be re-masked later,
        # while still protecting the fixed prompt/template tokens.
        editable_mask = (input_ids == cfg.mask_id).to(device_gen)

    emit_trace(
        event_type="init",
        step_index=None,
        prompt_length=prompt_length,
        total_length=int(x.shape[1]),
        masks_remaining=[int(v) for v in (x == cfg.mask_id).sum(dim=1).detach().cpu().tolist()],
        state_ids=x.detach().clone().cpu(),
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
            mask_index = (x == cfg.mask_id) & block_mask
            if not mask_index.any():
                break
            suppressed_active &= x == cfg.mask_id

            out = model(x, attention_mask=att)
            if cfg.suppression_value > 0.0:
                rows, cols = torch.nonzero(suppressed_active, as_tuple=True)
                if rows.numel() > 0:
                    tok_ids = suppressed_token_ids[rows, cols]
                    valid = tok_ids >= 0
                    if valid.any():
                        rows = rows[valid]
                        cols = cols[valid]
                        tok_ids = tok_ids[valid]
                        out.logits[rows, cols, tok_ids] -= float(cfg.suppression_value)
            # First let the base generator make a normal denoising move.
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

            # Nothing new was filled, so further iterations would be a no-op.
            if not transfer_index.any():
                break

            next_rows = []
            remaining_steps_after = int(cfg.steps) - global_step - 1
            active_editable_mask = editable_mask & block_mask
            for row in range(proposed_x.shape[0]):
                proposed_row = proposed_x[row : row + 1]
                attention_mask_row = att[row : row + 1]
                editable_row = active_editable_mask[row]

                # 1. Identify tokens to audit inside the editable response region.
                # Always audit newly written editable tokens.
                new_positions = torch.nonzero(
                    transfer_index[row] & editable_row,
                    as_tuple=False,
                ).squeeze(-1)

                # Also audit some historical tokens that are "uncertain"
                is_filled = (proposed_row[0] != cfg.mask_id) & editable_row
                # Exclude newly added positions to get purely historical filled ones
                historical_mask = is_filled.clone()
                if new_positions.numel() > 0:
                    historical_mask[new_positions] = False
                historical_indices = torch.nonzero(historical_mask, as_tuple=False).squeeze(-1)

                audit_indices = new_positions
                if historical_indices.numel() > 0 and int(cfg.historical_audit_topk) > 0:
                    # A small historical audit catches risky tokens that slipped through
                    # earlier steps but still look unstable under the current logits.
                    hist_uncertainty = 1.0 - token_probs[row].index_select(0, historical_indices)
                    n_audit_hist = min(int(cfg.historical_audit_topk), int(historical_indices.numel()))
                    _, top_hist_idx = torch.topk(hist_uncertainty, k=n_audit_hist)
                    audit_indices = torch.cat([new_positions, historical_indices.index_select(0, top_hist_idx)])

                # 2. Surgical Counterfactual Remasking
                # We now audit BOTH new and suspicious historical tokens.
                safe_row, audit_meta = _counterfactual_safe_remask_row(
                    val_model=val_model,
                    proposed_row=proposed_row,
                    attention_mask_row=attention_mask_row,
                    new_positions=audit_indices,
                    token_probs_row=token_probs[row],
                    total_initial_masks=int(total_initial_masks[row].item()),
                    device_val=device_val,
                    cfg=cfg,
                    remaining_steps_after=remaining_steps_after,
                    step_idx=global_step,
                    total_steps=int(cfg.steps),
                )

                # NOTE: Emergency Panic is disabled (was a second value evaluation that added ~50% overhead).
                # With prompt protection, surgical remasking is sufficient.

                emit_trace(
                    event_type="step",
                    step_index=int(global_step),
                    batch_index=int(row),
                    prompt_length=prompt_length,
                    proposal_state_ids=proposed_row[0].detach().clone().cpu(),
                    steered_state_ids=safe_row[0].detach().clone().cpu(),
                    fresh_positions=new_positions.detach().cpu().tolist(),
                    audit_positions=audit_indices.detach().cpu().tolist(),
                    active_suppressed_positions=torch.nonzero(suppressed_active[row], as_tuple=False).squeeze(-1).detach().cpu().tolist(),
                    active_suppressed_token_ids=suppressed_token_ids[row, suppressed_active[row]].detach().cpu().tolist(),
                    proposal_masks_remaining=int((proposed_row[0] == cfg.mask_id).sum().item()),
                    steered_masks_remaining=int((safe_row[0] == cfg.mask_id).sum().item()),
                    **audit_meta,
                )

                if audit_meta.get("remasked_positions"):
                    remask_pos = torch.tensor(
                        audit_meta["remasked_positions"],
                        dtype=torch.long,
                        device=device_gen,
                    )
                    suppressed_token_ids[row, remask_pos] = proposed_row[0, remask_pos].to(torch.long)
                    suppressed_active[row, remask_pos] = True

                next_rows.append(safe_row)

            # The next denoising step continues from the repaired sequence, not the raw proposal.
            x = torch.cat(next_rows, dim=0)
            if stop_checker is not None and stop_checker(x):
                break

    emit_trace(
        event_type="final",
        step_index=None,
        prompt_length=prompt_length,
        total_length=int(x.shape[1]),
        masks_remaining=[int(v) for v in (x == cfg.mask_id).sum(dim=1).detach().cpu().tolist()],
        state_ids=x.detach().clone().cpu(),
    )
    return x


@torch.no_grad()
def counterfactual_safe_generate(
    model,
    val_model,
    tokenizer,
    behavior: str,
    device_gen: str,
    device_val: str,
    cfg: CounterfactualSafeSteeringConfig,
) -> str:
    """Convenience wrapper that decodes only the response suffix as text."""
    input_ids, attention_mask, match_idx = prepare_masked_chat_inputs(
        tokenizer=tokenizer,
        behavior=behavior,
        device=device_gen,
        cfg=cfg,
    )
    output_ids = counterfactual_safe_decode(
        model=model,
        val_model=val_model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        device_gen=device_gen,
        device_val=device_val,
        cfg=cfg,
    )
    return tokenizer.decode(output_ids[0, match_idx:], skip_special_tokens=True).strip()
