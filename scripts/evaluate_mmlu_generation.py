import argparse
import json
import os
import random
import re
import sys

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.counterfactual_safe_steering import CounterfactualSafeSteeringConfig, _value_scores
from utils.low_cost_counterfactual_safe_steering import generator_feature_counterfactual_safe_decode
from utils.modeling_llada import LLaDAModelLM
from utils.project_paths import DEFAULT_LLADA_SOURCE, resolve_model_source, resolve_project_path
from utils.value_model_loader import (
    add_value_model_args,
    apply_value_model_runtime_args,
    load_value_model_from_checkpoint,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_mmlu_records(path: str) -> list[dict]:
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    raise ValueError(f"Unsupported MMLU file format: {path}")


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

    return CounterfactualSafeSteeringConfig(
        steps=int(cfg_obj.get("steps", 64)),
        gen_length=int(cfg_obj.get("gen_length", 32)),
        mask_id=int(cfg_obj.get("mask_id", 126336)),
        time_feature_mode=str(cfg_obj.get("time_feature_mode", "progress_ratio")),
        proposal_temperature=float(cfg_obj.get("proposal_temperature", 0.2)),
        unsafe_weight=float(cfg_obj.get("unsafe_weight", 0.8)),
        uncertainty_weight=float(cfg_obj.get("uncertainty_weight", 0.2)),
        necessity_weight=float(cfg_obj.get("necessity_weight", 0.25)),
        sufficiency_weight=float(cfg_obj.get("sufficiency_weight", 0.75)),
        remask_threshold=float(cfg_obj.get("remask_threshold", 0.25)),
        max_remask_ratio=float(cfg_obj.get("max_remask_ratio", 0.85)),
        min_keep_per_step=int(cfg_obj.get("min_keep_per_step", 0)),
        suppression_value=float(cfg_obj.get("suppression_value", 0.0)),
        historical_audit_topk=int(cfg_obj.get("historical_audit_topk", 8)),
        use_linear_threshold_decay=bool(cfg_obj.get("use_linear_threshold_decay", True)),
        remask_threshold_floor=float(cfg_obj.get("remask_threshold_floor", 0.0)),
        fixed_remask_threshold=bool(cfg_obj.get("fixed_remask_threshold", False)),
    )


def format_mmlu_prompt(item):
    question = item["question"]
    choices = item["choices"]
    prompt = (
        "The following is a multiple choice question. Please answer by responding "
        "with only the correct letter (A, B, C, or D).\n\n"
        f"Question: {question}\n"
    )
    for i, choice in enumerate(choices):
        letter = chr(ord("A") + i)
        prompt += f"({letter}) {choice}\n"
    prompt += "\nAnswer:"
    return prompt


def decode_suffix_from_prompt_ids(tokenizer, output_ids: torch.Tensor, prompt_ids: torch.Tensor) -> str:
    prompt_len = min(int(prompt_ids.shape[0]), int(output_ids.shape[0]))
    if prompt_len > 0 and torch.equal(output_ids[:prompt_len], prompt_ids[:prompt_len]):
        return tokenizer.decode(output_ids[prompt_len:], skip_special_tokens=True).strip()

    match_len = 0
    while match_len < prompt_len and int(output_ids[match_len].item()) == int(prompt_ids[match_len].item()):
        match_len += 1
    return tokenizer.decode(output_ids[match_len:], skip_special_tokens=True).strip()


def isolate_answer_region(text: str) -> str:
    raw = text.strip()
    if not raw:
        return ""

    assistant_pattern = re.compile(r"(^|\n)\s*assistant\s*:?\s*", flags=re.IGNORECASE)
    assistant_matches = list(assistant_pattern.finditer(raw))
    if assistant_matches:
        candidate = raw[assistant_matches[-1].end():].strip()
        if candidate:
            return candidate

    answer_idx = raw.lower().rfind("answer:")
    if answer_idx != -1:
        candidate = raw[answer_idx + len("answer:"):].strip()
        if candidate:
            return candidate

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if lines:
        return lines[-1]
    return raw


def extract_choice_letter(text: str) -> str:
    cleaned = isolate_answer_region(text)
    patterns = [
        r"^\(?\s*([ABCD])\s*[\)\.\:]?",
        r"^(?:option|answer)\s*:?\s*\(?\s*([ABCD])\s*[\)\.\:]?",
    ]
    for pattern in patterns:
        m = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return ""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate MMLU with stale generator-feature counterfactual-safe steering"
    )
    parser.add_argument("--llada_path", type=str, default=os.getenv("LLADA_PATH"))
    parser.add_argument(
        "--value_model_path",
        type=str,
        default=os.getenv("VALUE_MODEL_PATH"),
        required=not bool(os.getenv("VALUE_MODEL_PATH")),
        help="Value checkpoint path; uses VALUE_MODEL_PATH if set, otherwise required.",
    )
    parser.add_argument("--data_path", type=str, default="benchmarks/mmlu_multisubject_test.jsonl")
    parser.add_argument("--config_json", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device_gen", type=str, default="cuda:0")
    parser.add_argument("--device_val", type=str, default="cuda:0")
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
    parser.add_argument("--readout_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=int(os.getenv("VALUE_MODEL_N_LAYERS", "27")))
    parser.add_argument("--seed", type=int, default=456)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--generator_feature_layer", type=int, default=None)
    parser.add_argument(
        "--generator_feature_time_injection",
        type=str,
        default="none",
        choices=["add_then_ln", "ln_then_add", "none"],
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="results/mmlu_generator_feature_stale_eval.json",
    )
    add_value_model_args(parser)
    return parser.parse_args()


def save_payload(path: str, args, safe_cfg, results: list[dict], correct_steer: int) -> None:
    n = len(results)
    payload = {
        "config": {
            "args": vars(args),
            "counterfactual_safe": safe_cfg.__dict__,
            "generator_feature": {
                "enabled": True,
                "stale": True,
                "layer": args.generator_feature_layer,
                "time_injection": args.generator_feature_time_injection,
            },
        },
        "summary": {
            "n": n,
            "generator_feature_stale_accuracy": correct_steer / n if n else 0.0,
            "generator_feature_stale_accuracy_percent": correct_steer / n * 100 if n else 0.0,
        },
        "results": results,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


@torch.no_grad()
def main():
    args = parse_args()
    set_seed(args.seed)

    args.llada_path = resolve_model_source(
        args.llada_path,
        env_name="LLADA_PATH",
        model_subdir="LLaDA-8B-Instruct",
        hf_fallback=DEFAULT_LLADA_SOURCE,
    )
    args.value_model_path = resolve_project_path(args.value_model_path)
    args.data_path = resolve_project_path(args.data_path)
    args.output_file = resolve_project_path(args.output_file)

    for label, path in [("MMLU data", args.data_path), ("Value checkpoint", args.value_model_path)]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    safe_cfg = load_single_config(args.config_json)
    mask_id = safe_cfg.mask_id
    gen_length = safe_cfg.gen_length

    print(f"Resolved LLaDA source: {args.llada_path}")
    print(f"Resolved value checkpoint: {args.value_model_path}")
    print(f"Resolved data path: {args.data_path}")
    print(f"Resolved output file: {args.output_file}")
    print(f"Counterfactual-safe config: {safe_cfg}")
    print(
        "Generator feature config: "
        f"layer={args.generator_feature_layer}, "
        f"time_injection={args.generator_feature_time_injection}, stale=True"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.llada_path, trust_remote_code=True)
    model = LLaDAModelLM.from_pretrained(args.llada_path, torch_dtype=torch.float16).to(args.device_gen).eval()

    val_model, value_model_kwargs = load_value_model_from_checkpoint(
        llada_path=args.llada_path,
        value_model_path=args.value_model_path,
        device=args.device_val,
        torch_dtype=torch.float16,
        args=args,
        mask_id=mask_id,
    )
    apply_value_model_runtime_args(args, value_model_kwargs)

    data = load_mmlu_records(args.data_path)
    if args.max_samples > 0:
        data = data[: args.max_samples]

    results = []
    if args.resume and os.path.exists(args.output_file):
        try:
            old = json.load(open(args.output_file, "r", encoding="utf-8"))
            if isinstance(old, dict) and isinstance(old.get("results"), list):
                results = old["results"]
        except Exception:
            results = []

    processed = {int(r["idx"]) for r in results if isinstance(r, dict) and "idx" in r}
    correct_steer = sum(1 for r in results if r.get("steer_correct"))

    pbar = tqdm(list(enumerate(data)), desc="MMLU Generator-Feature Stale Eval")
    for i, item in pbar:
        if i in processed:
            n_done = len(results)
            if n_done:
                pbar.set_postfix(
                    stale_acc=f"{correct_steer / n_done * 100:.1f}%",
                )
            continue

        prompt_text = format_mmlu_prompt(item)
        target_letter = chr(ord("A") + item["answer"])

        chat_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_inputs = tokenizer(chat_prompt, return_tensors="pt")
        prompt_ids = prompt_inputs["input_ids"][0].to(args.device_gen)

        prompt_full = chat_prompt + (tokenizer.decode([mask_id]) * gen_length)
        inputs = tokenizer(prompt_full, return_tensors="pt")
        input_ids = inputs["input_ids"].to(args.device_gen)
        attention_mask = inputs["attention_mask"].to(args.device_gen)
        total_initial_masks_val = (input_ids == mask_id).sum(dim=1).clamp_min(1)

        out_steer = generator_feature_counterfactual_safe_decode(
            model=model,
            val_model=val_model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            device_gen=args.device_gen,
            device_val=args.device_val,
            cfg=safe_cfg,
            generator_feature_layer=args.generator_feature_layer,
            generator_feature_time_injection=args.generator_feature_time_injection,
            generator_feature_stale=True,
        )
        steer_safety = _value_scores(
            val_model=val_model,
            states=out_steer,
            attention_mask=attention_mask,
            total_initial_masks=int(total_initial_masks_val[0].item()),
            mask_id=mask_id,
            device_val=args.device_val,
            time_feature_mode=safe_cfg.time_feature_mode,
        )[0].item()

        steer_full_response = tokenizer.decode(out_steer[0], skip_special_tokens=True).strip()
        steer_suffix_response = decode_suffix_from_prompt_ids(tokenizer, out_steer[0], prompt_ids)
        steer_answer_region = isolate_answer_region(steer_full_response)
        steer_letter = extract_choice_letter(steer_answer_region)

        is_steer_correct = steer_letter == target_letter
        correct_steer += int(is_steer_correct)

        results.append(
            {
                "idx": i,
                "prompt_text": prompt_text,
                "chat_prompt": chat_prompt,
                "target": target_letter,
                "steer_full_response": steer_full_response,
                "steer_suffix_response": steer_suffix_response,
                "steer_raw_response": steer_answer_region,
                "steer_letter": steer_letter,
                "steer_correct": is_steer_correct,
                "steer_unsafe_score": 1.0 - steer_safety,
            }
        )

        n_done = len(results)
        pbar.set_postfix(
            stale_acc=f"{correct_steer / n_done * 100:.1f}%",
        )

        if n_done % max(args.save_every, 1) == 0:
            save_payload(args.output_file, args, safe_cfg, results, correct_steer)

    save_payload(args.output_file, args, safe_cfg, results, correct_steer)

    n = len(results)
    print(f"\n=== MMLU Generator-Feature Stale Results ({n} samples) ===")
    print(f"Generator-feature stale Accuracy:  {correct_steer / n * 100 if n else 0.0:.2f}%")
    print(f"Saved:                             {args.output_file}")


if __name__ == "__main__":
    main()
