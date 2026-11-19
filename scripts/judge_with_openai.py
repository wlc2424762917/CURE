from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import httpx
import yaml
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.project_paths import resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Judge steering responses with an OpenAI-compatible safety judge."
    )
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--model", type=str, default=os.getenv("SAFETY_MODEL_NAME", "gpt-4o-mini"))
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--request_timeout", type=float, default=30.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--prompt_config",
        type=str,
        default=os.getenv("OPENAI_JUDGE_PROMPT_CONFIG", "configs/openai_judge_prompt.yaml"),
    )
    return parser.parse_args()


def load_prompt_config(path: str) -> tuple[str, str]:
    resolved = resolve_project_path(path)
    with open(resolved, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Prompt config must contain a YAML mapping: {resolved}")
    system_prompt = str(payload.get("system_prompt", "")).strip()
    user_prompt_template = str(payload.get("user_prompt_template", "")).strip()
    if not system_prompt or not user_prompt_template:
        raise ValueError(
            f"Prompt config must define non-empty system_prompt and user_prompt_template: {resolved}"
        )
    return system_prompt, user_prompt_template


def build_client() -> OpenAI:
    api_key = (
        os.getenv("SAFETY_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("openai_api_key")
    )
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY / SAFETY_API_KEY / openai_api_key")

    base_url = (
        os.getenv("SAFETY_API_BASE")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("openai_api_base")
    )
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    proxy = (
        os.getenv("SAFETY_API_PROXY")
        or os.getenv("HTTPS_PROXY")
        or os.getenv("https_proxy")
        or os.getenv("HTTP_PROXY")
        or os.getenv("http_proxy")
    )
    if proxy:
        kwargs["http_client"] = httpx.Client(proxy=proxy)
    return OpenAI(**kwargs)


def judge_one(
    client: OpenAI,
    model: str,
    behavior: str,
    response: str,
    request_timeout: float,
    system_prompt: str,
    user_prompt_template: str,
    max_retries: int = 3,
) -> Optional[Dict[str, Any]]:
    prompt = user_prompt_template.format(behavior=behavior, response=response)

    for attempt in range(max_retries):
        try:
            out = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
                max_tokens=300,
                timeout=float(request_timeout),
            )
            data = json.loads(out.choices[0].message.content)
            cat = data.get("response_safety_category")
            if cat in {"SAFE_REFUSAL", "UNSAFE_COMPLIANCE"}:
                return data
        except Exception as exc:
            print(
                f"[WARN] judge_one failed attempt={attempt + 1}/{max_retries} "
                f"model={model} error={type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
        time.sleep(2 + attempt)
    return None


def asr_of(results: List[Dict[str, Any]], key: str) -> float:
    judged = [
        r for r in results
        if isinstance(r.get(key), dict)
        and r[key].get("response_safety_category") in {"SAFE_REFUSAL", "UNSAFE_COMPLIANCE"}
    ]
    if not judged:
        return 0.0
    unsafe = sum(1 for r in judged if r[key]["response_safety_category"] == "UNSAFE_COMPLIANCE")
    return unsafe / len(judged)


def save_payload(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_summary(results):
    steering_asr = asr_of(results, "steering_judge")
    summary = {"n": len(results), "steering_judge_asr": steering_asr,
               "steering_judge_asr_percent": steering_asr * 100}
    keys = ["steering_judge"]
    for key in keys:
        judged = sum(isinstance(r.get(key), dict) and r[key].get("response_safety_category")
                     in {"SAFE_REFUSAL", "UNSAFE_COMPLIANCE"} for r in results)
        summary[key + "_judged"] = judged
        summary[key + "_missing"] = len(results) - judged
    return summary


def main() -> None:
    args = parse_args()
    args.input_file = resolve_project_path(args.input_file)
    args.output_file = resolve_project_path(args.output_file)
    args.prompt_config = resolve_project_path(args.prompt_config)
    system_prompt, user_prompt_template = load_prompt_config(args.prompt_config)
    client = build_client()

    with open(args.input_file, "r", encoding="utf-8") as f:
        source = json.load(f)
    results = source.get("results", [])

    judged_results = results
    if args.resume and os.path.exists(args.output_file):
        try:
            old = json.load(open(args.output_file, "r", encoding="utf-8"))
            if isinstance(old, dict) and isinstance(old.get("results"), list):
                old_results = old["results"]
                old_by_idx = {
                    item.get("idx"): item
                    for item in old_results
                    if isinstance(item, dict) and item.get("idx") is not None
                }
                merged_results = []
                seen = set()
                for item in results:
                    if not isinstance(item, dict):
                        merged_results.append(item)
                        continue
                    idx = item.get("idx")
                    if idx is not None and idx in old_by_idx:
                        merged_results.append(old_by_idx[idx])
                        seen.add(idx)
                    else:
                        merged_results.append(item)
                for item in old_results:
                    if not isinstance(item, dict):
                        continue
                    idx = item.get("idx")
                    if idx is not None and idx not in seen and idx not in {
                        src.get("idx") for src in results if isinstance(src, dict)
                    }:
                        merged_results.append(item)
                judged_results = merged_results
        except Exception:
            judged_results = results

    for idx, item in enumerate(judged_results):
        behavior = str(item.get("refined_behavior") or item.get("behavior") or "").strip()
        steering_response = str(item.get("steering_response", "")).strip()
        if not behavior:
            continue

        if not isinstance(item.get("steering_judge"), dict) and steering_response:
            item["steering_judge"] = judge_one(
                client,
                args.model,
                behavior,
                steering_response,
                request_timeout=args.request_timeout,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template,
            )

        steering_done = isinstance(item.get("steering_judge"), dict)
        print(
            f"[{idx + 1}/{len(judged_results)}] idx={item.get('idx')} "
            f"steering={'ok' if steering_done else 'none'}",
            flush=True,
        )

        if (idx + 1) % max(args.save_every, 1) == 0:
            payload = {
                "config": {
                    "source_file": args.input_file,
                    "judge_model": args.model,
                    "prompt_config": args.prompt_config,
                },
                "summary": build_summary(judged_results),
                "results": judged_results,
            }
            save_payload(args.output_file, payload)

    steering_asr = asr_of(judged_results, "steering_judge")
    final = {
        "config": {
            "source_file": args.input_file,
            "judge_model": args.model,
            "prompt_config": args.prompt_config,
        },
        "summary": build_summary(judged_results),
        "results": judged_results,
    }
    save_payload(args.output_file, final)

    print("=== OpenAI Judge Summary ===")
    print(f"Model                : {args.model}")
    print(f"Steering judge ASR   : {steering_asr * 100:.2f}%")
    print(f"Saved                : {args.output_file}")


if __name__ == "__main__":
    main()
