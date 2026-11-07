from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.value_model_loader import (
    add_value_model_args,
    apply_value_model_runtime_args,
    load_value_model_from_checkpoint,
)
from utils import dija
from utils.counterfactual_safe_steering import (
    CounterfactualSafeSteeringConfig,
    counterfactual_safe_decode,
)
from utils.low_cost_counterfactual_safe_steering import (
    generator_feature_counterfactual_safe_decode,
    low_cost_counterfactual_safe_decode,
)
from utils.modeling_llada import LLaDAModelLM
from utils.project_paths import (
    DEFAULT_LLADA_SOURCE,
    resolve_model_source,
    resolve_project_path,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_device(x: str) -> str:
    if x == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return x


def load_single_config(path: str) -> CounterfactualSafeSteeringConfig:
    resolved = resolve_project_path(path)
    with open(resolved, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        cfg_obj = data
    elif isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
        cfg_obj = data[0]
    else:
        raise ValueError("config_json must contain exactly one config object or a one-item list")

    use_linear_threshold_decay = cfg_obj.get("use_linear_threshold_decay", True)
    if isinstance(use_linear_threshold_decay, str):
        use_linear_threshold_decay = use_linear_threshold_decay.strip().lower() in {"1", "true", "yes", "y", "on"}
    fixed_remask_threshold = cfg_obj.get("fixed_remask_threshold", False)
    if isinstance(fixed_remask_threshold, str):
        fixed_remask_threshold = fixed_remask_threshold.strip().lower() in {"1", "true", "yes", "y", "on"}

    return CounterfactualSafeSteeringConfig(
        steps=int(cfg_obj.get("steps", 64)),
        gen_length=int(cfg_obj.get("gen_length", 128)),
        mask_id=int(cfg_obj.get("mask_id", 126336)),
        time_feature_mode=str(cfg_obj.get("time_feature_mode", "progress_ratio")),
        proposal_temperature=float(cfg_obj.get("proposal_temperature", 0.8)),
        unsafe_weight=float(cfg_obj.get("unsafe_weight", 0.7)),
        uncertainty_weight=float(cfg_obj.get("uncertainty_weight", 0.3)),
        necessity_weight=float(cfg_obj.get("necessity_weight", 0.5)),
        sufficiency_weight=float(cfg_obj.get("sufficiency_weight", 0.5)),
        remask_threshold=float(cfg_obj.get("remask_threshold", 0.55)),
        max_remask_ratio=float(cfg_obj.get("max_remask_ratio", 0.5)),
        min_keep_per_step=int(cfg_obj.get("min_keep_per_step", 1)),
        suppression_value=float(cfg_obj.get("suppression_value", 0.0)),
        historical_audit_topk=int(cfg_obj.get("historical_audit_topk", 8)),
        use_linear_threshold_decay=bool(use_linear_threshold_decay),
        remask_threshold_floor=float(cfg_obj.get("remask_threshold_floor", 0.0)),
        fixed_remask_threshold=bool(fixed_remask_threshold),
    )


def save_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def token_common_prefix_len(left: torch.Tensor, right: torch.Tensor) -> int:
    """Return the contiguous token prefix length shared by two 1-D token tensors."""
    n = 0
    for left_tok, right_tok in zip(left, right):
        if left_tok.item() != right_tok.item():
            break
        n += 1
    return n


def decode_dija_response(
    tokenizer,
    output_ids: torch.Tensor,
    vanilla_prompt: str,
    prompt: str,
    device: torch.device | str,
    keep_special_tokens: bool = False,
) -> str:
    vanilla_input_ids, _ = dija.get_tokenized_input(vanilla_prompt, tokenizer, device)
    input_ids, _ = dija.get_tokenized_input(prompt, tokenizer, device)
    matching_count = token_common_prefix_len(vanilla_input_ids[0], input_ids[0])
    response = tokenizer.decode(
        output_ids[0].long().cpu()[matching_count:],
        skip_special_tokens=not keep_special_tokens,
    )
    if keep_special_tokens:
        return response.strip()
    if response.startswith("assistant\n"):
        response = response.split("assistant\n", 1)[-1]
    response = response.replace(dija.START_TOKEN, "").replace(dija.END_TOKEN, "").strip()
    if "assistant\n" in response:
        response = response.split("assistant\n", 1)[-1]
    return response.replace(dija.START_TOKEN, "").replace(dija.END_TOKEN, "").strip()


def build_editable_mask(
    vanilla_prompt: str,
    refined_prompt: str,
    tokenizer,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Make only the refined response scaffold editable; protect the shared prefix."""
    van_ids, _ = dija.get_tokenized_input(vanilla_prompt, tokenizer, device)
    ref_ids, _ = dija.get_tokenized_input(refined_prompt, tokenizer, device)
    matching_count = token_common_prefix_len(van_ids[0], ref_ids[0])
    editable_mask = torch.zeros_like(ref_ids, dtype=torch.bool, device=device)
    editable_mask[:, matching_count:] = True
    return editable_mask, matching_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate CURE-only AdvBench DIJA responses while "
            "allowing steering to rewrite only the refined response scaffold."
        )
    )
    parser.add_argument("--llada_path", type=str, default=os.getenv("LLADA_PATH"))
    parser.add_argument(
        "--value_model_path",
        type=str,
        default=os.getenv("VALUE_MODEL_PATH", "models/value_model_checkpoints/value_model_best.pt"),
    )
    parser.add_argument(
        "--attack_json_path",
        type=str,
        default=str(PROJECT_ROOT / "benchmarks" / "AdvBench" / "data" / "prompts_dija.json"),
    )
    parser.add_argument("--config_json", type=str, required=True)
    parser.add_argument(
        "--output_file",
        type=str,
        default="results/advbench_diffuguard_dija_editable_response.json",
    )
    parser.add_argument("--device_gen", type=str, default=os.getenv("DEVICE_GEN", "cuda:0"))
    parser.add_argument("--device_val", type=str, default=os.getenv("DEVICE_VAL", "cuda:0"))
    parser.add_argument(
        "--value_model_arch",
        type=str,
        default=os.getenv("VALUE_MODEL_ARCH", "readout_cls"),
        choices=["full_cls", "readout_cls"],
    )
    parser.add_argument(
        "--pooling_mode",
        type=str,
        default=os.getenv("VALUE_MODEL_POOLING_MODE", "auto"),
        choices=["auto", "full_cls", "readout_cls", "mean_visible", "max_visible", "last_visible"],
    )
    parser.add_argument("--n_layers", type=int, default=27)
    parser.add_argument("--readout_heads", type=int, default=8)
    parser.add_argument("--mask_id", type=int, default=126336)
    parser.add_argument("--mask_counts", type=int, default=36)
    parser.add_argument(
        "--low_cost_steering",
        action="store_true",
        help="Use the batched cached-readout steering path for readout_cls value models.",
    )
    parser.add_argument(
        "--generator_feature_steering",
        action="store_true",
        help=(
            "Experimental: score readout_cls views from generator hidden states "
            "instead of running the value-model backbone."
        ),
    )
    parser.add_argument(
        "--generator_feature_layer",
        type=int,
        default=None,
        help="Generator hidden-state index to use. Defaults to the value model layer count.",
    )
    parser.add_argument(
        "--generator_feature_time_injection",
        type=str,
        default="add_then_ln",
        choices=["add_then_ln", "ln_then_add", "none"],
        help="How to inject the value model time embedding into generator hidden features.",
    )
    parser.add_argument(
        "--generator_feature_stale",
        action="store_true",
        help=(
            "Reuse the pre-proposal generator hidden states from the logits forward. "
            "Cheaper but features do not correspond to proposed_x."
        ),
    )
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--seed", type=int, default=456,
        help="Generation base seed; CURE sample idx uses seed + 2*idx + 1.",
    )
    parser.add_argument(
        "--keep_special_tokens",
        action="store_true",
        help="Decode steering responses with special tokens preserved.",
    )
    add_value_model_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    args.llada_path = resolve_model_source(
        args.llada_path,
        env_name="LLADA_PATH",
        model_subdir="LLaDA-8B-Instruct",
        hf_fallback=DEFAULT_LLADA_SOURCE,
    )
    args.value_model_path = resolve_project_path(args.value_model_path)
    args.attack_json_path = resolve_project_path(args.attack_json_path)
    args.output_file = resolve_project_path(args.output_file)
    args.device_gen = ensure_device(args.device_gen)
    args.device_val = ensure_device(args.device_val)

    if args.low_cost_steering and args.generator_feature_steering:
        raise ValueError("Select either low_cost_steering or generator_feature_steering")
    if args.generator_feature_stale and not args.generator_feature_steering:
        raise ValueError("generator_feature_stale requires generator_feature_steering")

    for label, path in [("Attack JSON", args.attack_json_path), ("Value checkpoint", args.value_model_path)]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    safe_cfg = load_single_config(args.config_json)

    print(f"Resolved LLaDA source: {args.llada_path}")
    print(f"Resolved value checkpoint: {args.value_model_path}")
    print(f"Resolved attack JSON: {args.attack_json_path}")
    print(f"Resolved output file: {args.output_file}")
    print(f"DIJA input scaffold: mask_counts={args.mask_counts}")
    print(f"Counterfactual-safe config: {safe_cfg}")

    tokenizer = AutoTokenizer.from_pretrained(args.llada_path, trust_remote_code=True)
    model = LLaDAModelLM.from_pretrained(
        args.llada_path,
        torch_dtype=torch.float32 if args.device_gen == "cpu" else torch.bfloat16,
    ).to(args.device_gen).eval()

    val_model, value_model_kwargs = load_value_model_from_checkpoint(
        llada_path=args.llada_path,
        value_model_path=args.value_model_path,
        device=args.device_val,
        torch_dtype=torch.float32 if args.device_val == "cpu" else torch.float16,
        args=args,
        mask_id=locals().get('mask_id', getattr(safe_cfg, 'mask_id', getattr(args, 'mask_id', None))),
    )
    apply_value_model_runtime_args(args, value_model_kwargs)

    with open(args.attack_json_path, "r", encoding="utf-8") as f:
        attacks = json.load(f)
    if args.max_samples >= 0:
        attacks = attacks[: args.max_samples]

    results: List[Dict[str, Any]] = []
    if args.resume and os.path.exists(args.output_file):
        try:
            old = json.load(open(args.output_file, "r", encoding="utf-8"))
            if isinstance(old, dict) and isinstance(old.get("results"), list):
                results = old["results"]
        except Exception:
            results = []

    # A saved CURE response is complete, including an empty generated response.
    results = [
        r for r in results
        if isinstance(r, dict) and "idx" in r and "steering_response" in r
    ]
    processed = {r["idx"] for r in results}
    is_instruct = ("instruct" in str(args.llada_path).lower()) or ("1.5" in str(args.llada_path))
    device = torch.device(args.device_gen)

    pbar = tqdm(list(enumerate(attacks)), desc="Generate CURE DIJA Responses")
    with torch.no_grad():
        for idx, item in pbar:
            if idx in processed:
                continue

            behavior, refined = dija.pick_two_prompt_fields(item)
            if not behavior.strip():
                continue

            chosen_user = dija.process_user_text(refined, args.mask_counts)
            vanilla_user = dija.process_user_text(behavior, args.mask_counts)
            prompt = dija.build_chat_prompt(tokenizer, chosen_user, is_instruct, None)
            vanilla_prompt = dija.build_chat_prompt(tokenizer, vanilla_user, is_instruct, None)

            editable_mask, shared_prefix_tokens = build_editable_mask(
                vanilla_prompt=vanilla_prompt,
                refined_prompt=prompt,
                tokenizer=tokenizer,
                device=device,
            )

            # Keep the original CURE seed schedule when skipping baseline generation.
            set_seed(args.seed + idx * 2 + 1)
            input_ids, attention_mask = dija.get_tokenized_input(prompt, tokenizer, device)
            if args.generator_feature_steering:
                out_safe = generator_feature_counterfactual_safe_decode(
                    model=model,
                    val_model=val_model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    device_gen=args.device_gen,
                    device_val=args.device_val,
                    cfg=safe_cfg,
                    editable_mask=editable_mask,
                    generator_feature_layer=args.generator_feature_layer,
                    generator_feature_time_injection=args.generator_feature_time_injection,
                    generator_feature_stale=args.generator_feature_stale,
                )
            else:
                decode_fn = low_cost_counterfactual_safe_decode if args.low_cost_steering else counterfactual_safe_decode
                out_safe = decode_fn(
                    model=model,
                    val_model=val_model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    device_gen=args.device_gen,
                    device_val=args.device_val,
                    cfg=safe_cfg,
                    editable_mask=editable_mask,
                )
            steering_response = decode_dija_response(
                tokenizer=tokenizer,
                output_ids=out_safe,
                vanilla_prompt=vanilla_prompt,
                prompt=prompt,
                device=device,
                keep_special_tokens=args.keep_special_tokens,
            )

            rec = {
                "idx": idx,
                "behavior": behavior,
                "refined_behavior": refined,
                "editable_prefix_token_count": shared_prefix_tokens,
                "steering_response": steering_response,
            }
            results.append(rec)
            pbar.set_postfix(done=len(results))

            if len(results) % max(args.save_every, 1) == 0:
                payload = {
                    "config": {
                        "args": vars(args),
                        "steering_input": {
                            "mask_counts": args.mask_counts,
                        },
                        "counterfactual_safe": safe_cfg.__dict__,
                    },
                    "summary": {"n": len(results)},
                    "results": results,
                }
                save_json(args.output_file, payload)

    final = {
        "config": {
            "args": vars(args),
            "steering_input": {
                "mask_counts": args.mask_counts,
            },
            "counterfactual_safe": safe_cfg.__dict__,
        },
        "summary": {"n": len(results)},
        "results": results,
    }
    save_json(args.output_file, final)

    print("\n=== Generation Summary ===")
    print(f"Samples : {len(results)}")
    print(f"Saved   : {args.output_file}")


if __name__ == "__main__":
    main()
