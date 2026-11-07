import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LLADA_SOURCE = "GSAI-ML/LLaDA-8B-Instruct"
DEFAULT_HARMBENCH_SOURCE = "cais/HarmBench-Llama-2-13b-cls"


def project_path(*parts: str) -> str:
    return str(PROJECT_ROOT.joinpath(*parts).resolve())


def get_llada_path(explicit: str | None = None) -> str:
    """Resolve LLaDA model path, checking env var LLADA_PATH first."""
    return resolve_model_source(
        explicit=explicit,
        env_name="LLADA_PATH",
        model_subdir="LLaDA-8B-Instruct",
        hf_fallback=DEFAULT_LLADA_SOURCE,
    )


def get_classifier_path(explicit: str | None = None) -> str:
    """Resolve HarmBench classifier path, checking env var CLASSIFIER_PATH first."""
    return resolve_model_source(
        explicit=explicit,
        env_name="CLASSIFIER_PATH",
        model_subdir="HarmBench-Llama-2-13b-cls",
        hf_fallback=DEFAULT_HARMBENCH_SOURCE,
    )


def resolve_project_path(path: str | os.PathLike[str]) -> str:
    p = Path(path).expanduser()
    if p.is_absolute():
        return str(p.resolve())
    return str((PROJECT_ROOT / p).resolve())


def resolve_existing_path(raw: str) -> str | None:
    p = Path(raw).expanduser()
    if p.is_absolute() and p.exists():
        return str(p.resolve())
    if p.exists():
        return str(p.resolve())

    project_candidate = PROJECT_ROOT / p
    if project_candidate.exists():
        return str(project_candidate.resolve())
    return None


def resolve_model_source(
    explicit: str | None,
    env_name: str,
    model_subdir: str,
    hf_fallback: str,
) -> str:
    # Let an explicit CLI argument override the environment variable.
    raw = explicit or os.getenv(env_name)
    if raw:
        # Try local path first
        local = resolve_existing_path(raw)
        if local:
            return local
        # For HF identifiers (contain "/" but not absolute path), check HF cache
        if "/" in raw and not raw.startswith("/"):
            hf_cache = Path(os.getenv("HF_HOME", "~/.cache/huggingface"))
            cache_key = f"models--{raw.replace('/', '--')}"
            cache_path = hf_cache / "hub" / cache_key
            if cache_path.exists():
                # Find snapshot directory (actual model files)
                snapshots = cache_path / "snapshots"
                if snapshots.exists():
                    for snap in snapshots.iterdir():
                        if snap.is_dir():
                            return str(snap)
            return raw
        return raw

    candidate_roots = [PROJECT_ROOT, *PROJECT_ROOT.parents]
    seen = set()
    for root in candidate_roots:
        candidate = root / "hf_models" / model_subdir
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return str(candidate.resolve())
    return hf_fallback
