from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.value_model import ValueModel
from utils.project_paths import (
    DEFAULT_LLADA_SOURCE,
    resolve_model_source,
    resolve_project_path,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_world_info() -> tuple[int, int, int]:
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def is_distributed() -> bool:
    return int(os.getenv("WORLD_SIZE", "1")) > 1


def is_main_process() -> bool:
    return int(os.getenv("RANK", "0")) == 0


def maybe_init_distributed(device_arg: str) -> tuple[str, int, int, int]:
    rank, world_size, local_rank = get_world_info()
    if world_size <= 1:
        device = device_arg
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return device, rank, world_size, local_rank

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        backend = "nccl"
    else:
        device = "cpu"
        backend = "gloo"

    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return device, rank, world_size, local_rank


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def sync_workers() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def distributed_mean(value: float, device: str) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return float(value)
    tensor = torch.tensor([float(value)], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= float(dist.get_world_size())
    return float(tensor.item())


class DistributedWeightedSampler(Sampler[int]):
    """Weighted sampler compatible with DDP.

    `num_samples` is interpreted as the desired global sample count per epoch.
    Each rank deterministically samples the same global draw and then takes its shard.
    """

    def __init__(
        self,
        weights: torch.Tensor,
        num_samples: int,
        *,
        replacement: bool = True,
        rank: int | None = None,
        num_replicas: int | None = None,
        seed: int = 42,
    ) -> None:
        if weights.numel() == 0:
            raise ValueError("weights must be non-empty")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")

        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0

        self.weights = weights.to(dtype=torch.double)
        self.global_num_samples = int(num_samples)
        self.replacement = bool(replacement)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = int(math.ceil(float(self.global_num_samples) / float(self.num_replicas)))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        sampled = torch.multinomial(
            self.weights,
            self.total_size,
            self.replacement,
            generator=generator,
        ).tolist()
        indices = sampled[self.rank : self.total_size : self.num_replicas]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def load_records(data_path: str) -> list[dict[str, Any]]:
    with open(data_path, "rb") as f:
        if f.read(80).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(
                f"{data_path} is a Git LFS pointer. Install Git LFS and run "
                "'git lfs install && git lfs pull' from the repository root."
            )
    if data_path.endswith(".jsonl"):
        rows = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    with open(data_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    raise ValueError(f"Unsupported data format in {data_path}. Expected JSON list or JSONL.")


def resolve_data_file(data_dir: str, data_file: str | None) -> str:
    data_dir = resolve_project_path(data_dir)
    if data_file:
        if not os.path.isabs(data_file):
            data_file = os.path.join(data_dir, data_file)
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        return data_file

    preferred = os.path.join(data_dir, "advanced_tree_data.json")
    if os.path.exists(preferred):
        return preferred

    candidates = [
        os.path.join(data_dir, x)
        for x in os.listdir(data_dir)
        if x.endswith(".json") or x.endswith(".jsonl")
    ]
    if not candidates:
        raise FileNotFoundError(f"No JSON/JSONL files found under {data_dir}")
    candidates.sort()
    return candidates[-1]


def parse_source_specs(source_specs: list[str], data_dir: str, data_file: str | None) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    if source_specs:
        for idx, spec in enumerate(source_specs):
            if "::" in spec:
                name, raw_path = spec.split("::", 1)
            else:
                name, raw_path = f"source_{idx}", spec

            if os.path.isdir(resolve_project_path(raw_path)):
                path = resolve_data_file(resolve_project_path(raw_path), data_file)
            else:
                path = resolve_project_path(raw_path)
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Data source not found: {path}")
            specs.append((name.strip() or f"source_{idx}", path))
        return specs

    resolved = resolve_data_file(data_dir, data_file)
    return [(os.path.basename(resolve_project_path(data_dir.rstrip("/"))) or "default", resolved)]


def stable_hash(text: str, seed: int) -> int:
    payload = f"{seed}:{text}".encode("utf-8")
    return int(hashlib.sha1(payload).hexdigest(), 16)


def normalize_behavior(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def value_bin(x: float, n_bins: int) -> int:
    x = clamp01(x)
    if n_bins <= 1:
        return 0
    if x >= 1.0:
        return n_bins - 1
    return int(x * n_bins)


@dataclass
class TreeSample:
    input_ids: list[int]
    cls_attend_to: list[int] | None
    t: float
    y: float
    behavior: str
    mode: str
    source: str
    y_bin: int
    t_bin: int


class ValueTreeDataset(Dataset):
    def __init__(
        self,
        source_files: list[tuple[str, str]],
        *,
        y_bins: int,
        t_bins: int,
        max_records_per_source: int = -1,
    ):
        self.samples: list[TreeSample] = []
        self.source_paths = {name: path for name, path in source_files}

        for source_name, path in source_files:
            rows = load_records(path)
            added = 0
            for item in rows:
                x_t = item.get("x_t", item.get("token_ids"))
                t = item.get("t", item.get("time_step"))
                y = item.get("y", item.get("value", item.get("value_target")))
                behavior = item.get("behavior", item.get("Behavior", ""))
                mode_raw = item.get("mode", "unknown")
                mode = str(mode_raw).strip() if mode_raw is not None else ""
                if not mode:
                    mode = "unknown"
                if x_t is None or t is None or y is None or not behavior:
                    continue
                if not isinstance(x_t, list) or len(x_t) == 0:
                    continue
                cls_attend_to = item.get("cls_attend_to")
                if cls_attend_to is not None:
                    if not isinstance(cls_attend_to, list) or len(cls_attend_to) != len(x_t):
                        continue
                    cls_attend_to = [1 if bool(v) else 0 for v in cls_attend_to]
                t_f = clamp01(float(t))
                y_f = clamp01(float(y))
                self.samples.append(
                    TreeSample(
                        input_ids=x_t,
                        cls_attend_to=cls_attend_to,
                        t=t_f,
                        y=y_f,
                        behavior=str(behavior),
                        mode=mode,
                        source=source_name,
                        y_bin=value_bin(y_f, y_bins),
                        t_bin=value_bin(t_f, t_bins),
                    )
                )
                added += 1
                if max_records_per_source > 0 and added >= max_records_per_source:
                    break

        if not self.samples:
            raise ValueError("No valid samples found in provided sources.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.samples[idx]
        input_ids = torch.tensor(item.input_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        if item.cls_attend_to is None:
            cls_attend_to = input_ids.ne(126336)
        else:
            cls_attend_to = torch.tensor(item.cls_attend_to, dtype=torch.bool)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "cls_attend_to": cls_attend_to,
            "t": torch.tensor(item.t, dtype=torch.float32),
            "y": torch.tensor(item.y, dtype=torch.float32),
            "mode": item.mode,
            "source": item.source,
            "behavior": item.behavior,
            "y_bin": item.y_bin,
            "t_bin": item.t_bin,
        }


class WeightedSubset(Dataset):
    def __init__(self, dataset: ValueTreeDataset, indices: list[int], sample_weights: list[float] | None = None):
        self.dataset = dataset
        self.indices = list(indices)
        self.sample_weights = list(sample_weights) if sample_weights is not None else [1.0] * len(self.indices)
        if len(self.sample_weights) != len(self.indices):
            raise ValueError("sample_weights length must match indices length")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = dict(self.dataset[self.indices[idx]])
        item["sample_weight"] = float(self.sample_weights[idx])
        return item


def split_indices_by_behavior(
    dataset: ValueTreeDataset,
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")

    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, sample in enumerate(dataset.samples):
        grouped[normalize_behavior(sample.behavior)].append(idx)

    groups = [(stable_hash(key, seed), key, indices) for key, indices in grouped.items()]
    groups.sort(key=lambda x: x[0])

    total = len(dataset)
    target_val = max(1, min(total - 1, int(round(total * val_ratio))))
    val_indices: list[int] = []
    train_indices: list[int] = []

    running_val = 0
    for _, _, indices in groups:
        if running_val < target_val:
            val_indices.extend(indices)
            running_val += len(indices)
        else:
            train_indices.extend(indices)

    if not train_indices or not val_indices:
        raise RuntimeError(
            "Behavior-based split produced an empty train or validation split. "
            f"train={len(train_indices)} val={len(val_indices)}"
        )

    return train_indices, val_indices


def summary_for_indices(dataset: ValueTreeDataset, indices: Iterable[int]) -> dict[str, Any]:
    indices = list(indices)
    if not indices:
        return {"count": 0}

    mode_counts = Counter()
    source_counts = Counter()
    behavior_counts = Counter()
    y_bins = Counter()
    t_bins = Counter()
    ys: list[float] = []
    ts: list[float] = []

    for idx in indices:
        sample = dataset.samples[idx]
        mode_counts[sample.mode] += 1
        source_counts[sample.source] += 1
        behavior_counts[normalize_behavior(sample.behavior)] += 1
        y_bins[str(sample.y_bin)] += 1
        t_bins[str(sample.t_bin)] += 1
        ys.append(sample.y)
        ts.append(sample.t)

    ys_sorted = sorted(ys)
    ts_sorted = sorted(ts)

    def q(xs: list[float], p: float) -> float:
        pos = min(len(xs) - 1, max(0, int(round((len(xs) - 1) * p))))
        return float(xs[pos])

    return {
        "count": len(indices),
        "unique_behaviors": len(behavior_counts),
        "mode_counts": dict(mode_counts),
        "source_counts": dict(source_counts),
        "y_bins": dict(sorted(y_bins.items())),
        "t_bins": dict(sorted(t_bins.items())),
        "y_summary": {
            "mean": float(sum(ys) / len(ys)),
            "p10": q(ys_sorted, 0.1),
            "p50": q(ys_sorted, 0.5),
            "p90": q(ys_sorted, 0.9),
        },
        "t_summary": {
            "mean": float(sum(ts) / len(ts)),
            "p10": q(ts_sorted, 0.1),
            "p50": q(ts_sorted, 0.5),
            "p90": q(ts_sorted, 0.9),
        },
    }


def compute_sampling_weights(
    dataset: ValueTreeDataset,
    indices: list[int],
    *,
    source_alpha: float,
    mode_alpha: float,
    y_alpha: float,
    t_alpha: float,
    extreme_boost: float,
    extreme_low: float,
    extreme_high: float,
    max_weight_ratio: float,
) -> list[float]:
    if not indices:
        return []

    source_counts = Counter()
    mode_counts = Counter()
    mode_y_counts = Counter()
    mode_t_counts = Counter()

    for idx in indices:
        sample = dataset.samples[idx]
        source_counts[sample.source] += 1
        mode_counts[sample.mode] += 1
        mode_y_counts[(sample.mode, sample.y_bin)] += 1
        mode_t_counts[(sample.mode, sample.t_bin)] += 1

    weights: list[float] = []
    total = float(len(indices))
    for idx in indices:
        sample = dataset.samples[idx]
        w = 1.0
        if source_alpha > 0:
            w *= (total / float(source_counts[sample.source])) ** source_alpha
        if mode_alpha > 0:
            w *= (total / float(mode_counts[sample.mode])) ** mode_alpha
        if y_alpha > 0:
            w *= (total / float(mode_y_counts[(sample.mode, sample.y_bin)])) ** y_alpha
        if t_alpha > 0:
            w *= (total / float(mode_t_counts[(sample.mode, sample.t_bin)])) ** t_alpha
        if sample.y <= extreme_low or sample.y >= extreme_high:
            w *= max(extreme_boost, 1.0)
        weights.append(float(w))

    mean_w = sum(weights) / max(len(weights), 1)
    if mean_w <= 0:
        weights = [1.0 for _ in weights]
    else:
        weights = [w / mean_w for w in weights]

    if max_weight_ratio > 0:
        lo = 1.0 / max_weight_ratio
        hi = max_weight_ratio
        weights = [max(lo, min(hi, w)) for w in weights]

    return weights


def make_collate_fn(pad_token_id: int):
    mode_to_id = {"attack": 0, "vanilla": 1}

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        max_len = max(x["input_ids"].shape[0] for x in batch)
        bsz = len(batch)
        input_ids = torch.full((bsz, max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
        cls_attend_to = torch.zeros((bsz, max_len), dtype=torch.bool)
        t = torch.zeros((bsz, 1), dtype=torch.float32)
        y = torch.zeros((bsz, 1), dtype=torch.float32)
        sample_weight = torch.ones((bsz, 1), dtype=torch.float32)
        mode_id = torch.full((bsz,), -1, dtype=torch.long)

        modes: list[str] = []
        sources: list[str] = []
        behaviors: list[str] = []

        for i, item in enumerate(batch):
            cur_len = item["input_ids"].shape[0]
            input_ids[i, :cur_len] = item["input_ids"]
            attention_mask[i, :cur_len] = item["attention_mask"]
            cls_attend_to[i, :cur_len] = item["cls_attend_to"]
            t[i, 0] = item["t"]
            y[i, 0] = item["y"]
            sample_weight[i, 0] = item.get("sample_weight", 1.0)
            mode_id[i] = mode_to_id.get(item["mode"], -1)
            modes.append(item["mode"])
            sources.append(item["source"])
            behaviors.append(item["behavior"])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "cls_attend_to": cls_attend_to,
            "t": t,
            "y": y,
            "sample_weight": sample_weight,
            "mode_id": mode_id,
            "mode": modes,
            "source": sources,
            "behavior": behaviors,
        }

    return collate


def smooth_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    beta: float,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none", beta=beta)
    if sample_weight is not None:
        denom = sample_weight.sum().clamp_min(1e-6)
        return (loss * sample_weight).sum() / denom
    return loss.mean()


def batch_pairwise_ranking_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    t: torch.Tensor,
    mode_id: torch.Tensor,
    *,
    y_margin: float,
    t_window: float,
    hinge_margin: float,
    max_pairs: int,
) -> tuple[torch.Tensor, int, float]:
    pred = pred.squeeze(-1)
    target = target.squeeze(-1)
    t = t.squeeze(-1)

    pairs: list[tuple[int, int, float]] = []
    batch_size = int(pred.shape[0])
    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            if mode_id[i] != mode_id[j]:
                continue
            t_gap = abs(float(t[i].item()) - float(t[j].item()))
            if t_gap > t_window:
                continue
            y_diff = float(target[i].item() - target[j].item())
            abs_y_diff = abs(y_diff)
            if abs_y_diff < y_margin:
                continue
            pairs.append((i, j, abs_y_diff))

    if not pairs:
        zero = pred.new_zeros(())
        return zero, 0, 0.0

    pairs.sort(key=lambda x: x[2], reverse=True)
    if max_pairs > 0:
        pairs = pairs[:max_pairs]

    losses = []
    correct = 0
    for i, j, pair_weight in pairs:
        if target[i] >= target[j]:
            high, low = i, j
        else:
            high, low = j, i
        margin_term = hinge_margin - (pred[high] - pred[low])
        losses.append(torch.relu(margin_term) * pair_weight)
        if float(pred[high].item()) > float(pred[low].item()):
            correct += 1

    loss = torch.stack(losses).mean() if losses else pred.new_zeros(())
    acc = float(correct) / float(len(pairs))
    return loss, len(pairs), acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: str,
    huber_beta: float,
    ranking_lambda: float,
    ranking_y_margin: float,
    ranking_t_window: float,
    ranking_hinge_margin: float,
    ranking_max_pairs: int,
) -> dict[str, Any]:
    model.eval()
    reg_total = 0.0
    mse_total = 0.0
    mae_total = 0.0
    batches = 0
    num_examples = 0
    pair_loss_total = 0.0
    pair_count_total = 0
    pair_acc_weighted = 0.0
    per_mode: dict[str, dict[str, float]] = defaultdict(lambda: {"n": 0.0, "mse": 0.0, "mae": 0.0})

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            t = batch["t"].to(device)
            y = batch["y"].to(device)
            cls_attend_to = batch["cls_attend_to"].to(device)
            pred = model(input_ids, t, attention_mask=attention_mask, cls_attend_to=cls_attend_to)
            batch_examples = int(y.shape[0])

            reg_loss = smooth_l1_loss(pred, y, beta=huber_beta, sample_weight=None)
            reg_total += float(reg_loss.item()) * batch_examples
            mse_total += float(F.mse_loss(pred, y).item()) * batch_examples
            mae_total += float(F.l1_loss(pred, y).item()) * batch_examples
            batches += 1
            num_examples += batch_examples

            pair_loss, pair_count, pair_acc = batch_pairwise_ranking_loss(
                pred,
                y,
                t,
                batch["mode_id"].to(device),
                y_margin=ranking_y_margin,
                t_window=ranking_t_window,
                hinge_margin=ranking_hinge_margin,
                max_pairs=ranking_max_pairs,
            )
            pair_loss_total += float(pair_loss.item()) * batch_examples
            pair_count_total += int(pair_count)
            pair_acc_weighted += pair_acc * float(pair_count)

            pred_cpu = pred.squeeze(-1).detach().cpu()
            y_cpu = y.squeeze(-1).detach().cpu()
            for idx, mode in enumerate(batch["mode"]):
                err = float(pred_cpu[idx].item() - y_cpu[idx].item())
                slot = per_mode[mode]
                slot["n"] += 1
                slot["mse"] += err * err
                slot["mae"] += abs(err)

    if dist.is_available() and dist.is_initialized():
        stats = torch.tensor(
            [
                reg_total,
                mse_total,
                mae_total,
                float(batches),
                float(num_examples),
                pair_loss_total,
                float(pair_count_total),
                pair_acc_weighted,
            ],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        reg_total = float(stats[0].item())
        mse_total = float(stats[1].item())
        mae_total = float(stats[2].item())
        batches = int(round(float(stats[3].item())))
        num_examples = int(round(float(stats[4].item())))
        pair_loss_total = float(stats[5].item())
        pair_count_total = int(round(float(stats[6].item())))
        pair_acc_weighted = float(stats[7].item())

        gathered_per_mode: list[dict[str, dict[str, float]]] = [None for _ in range(dist.get_world_size())]  # type: ignore
        dist.all_gather_object(gathered_per_mode, dict(per_mode))
        merged_per_mode: dict[str, dict[str, float]] = defaultdict(lambda: {"n": 0.0, "mse": 0.0, "mae": 0.0})
        for shard in gathered_per_mode:
            for mode, stats_dict in shard.items():
                slot = merged_per_mode[mode]
                slot["n"] += float(stats_dict["n"])
                slot["mse"] += float(stats_dict["mse"])
                slot["mae"] += float(stats_dict["mae"])
        per_mode = merged_per_mode

    if batches == 0 or num_examples == 0:
        return {
            "loss": float("inf"),
            "regression_loss": float("inf"),
            "mse": float("inf"),
            "mae": float("inf"),
            "pair_loss": 0.0,
            "pair_count": 0,
            "pair_accuracy": 0.0,
            "per_mode": {},
        }

    out = {
        "loss": reg_total / num_examples + (ranking_lambda * (pair_loss_total / max(num_examples, 1))),
        "regression_loss": reg_total / num_examples,
        "mse": mse_total / num_examples,
        "mae": mae_total / num_examples,
        "pair_loss": pair_loss_total / num_examples,
        "pair_count": pair_count_total,
        "pair_accuracy": pair_acc_weighted / max(pair_count_total, 1),
        "per_mode": {},
    }
    for mode, stats in per_mode.items():
        n = max(stats["n"], 1.0)
        out["per_mode"][mode] = {
            "count": int(stats["n"]),
            "mse": stats["mse"] / n,
            "mae": stats["mae"] / n,
        }
    return out


def save_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def maybe_unfreeze_top_blocks(model: ValueModel, n_top_blocks: int) -> None:
    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model, "cls_embedding"):
        model.cls_embedding.requires_grad = True
    if hasattr(model, "cls_query"):
        model.cls_query.requires_grad = True
    if hasattr(model, "cls_queries"):
        model.cls_queries.requires_grad = True
    if hasattr(model, "readout_attn"):
        for param in model.readout_attn.parameters():
            param.requires_grad = True
    if hasattr(model, "readout_norm"):
        for param in model.readout_norm.parameters():
            param.requires_grad = True
    if hasattr(model, "readout_ff"):
        for param in model.readout_ff.parameters():
            param.requires_grad = True
    if hasattr(model, "readout_ff_norm"):
        for param in model.readout_ff_norm.parameters():
            param.requires_grad = True
    for param in model.time_embed.parameters():
        param.requires_grad = True
    for param in model.regression_head.parameters():
        param.requires_grad = True

    if n_top_blocks <= 0:
        return

    blocks = None
    if hasattr(model.llada.transformer, "blocks"):
        blocks = model.llada.transformer.blocks
    elif hasattr(model.llada.transformer, "block_groups"):
        blocks = model.llada.transformer.block_groups
    if blocks is None:
        return

    n_top_blocks = min(int(n_top_blocks), len(blocks))
    for block in blocks[-n_top_blocks:]:
        for param in block.parameters():
            param.requires_grad = True

    if hasattr(model.llada.transformer, "ln_f"):
        for param in model.llada.transformer.ln_f.parameters():
            param.requires_grad = True


def train(args: argparse.Namespace) -> None:
    device, rank, world_size, local_rank = maybe_init_distributed(args.device)
    set_seed(args.seed + rank)

    args.llada_path = resolve_model_source(
        args.llada_path,
        env_name="LLADA_PATH",
        model_subdir="LLaDA-8B-Instruct",
        hf_fallback=DEFAULT_LLADA_SOURCE,
    )
    args.output_dir = resolve_project_path(args.output_dir)
    args.data_dir = resolve_project_path(args.data_dir)
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
    sync_workers()

    source_files = parse_source_specs(args.source, args.data_dir, args.data_file)
    dataset = ValueTreeDataset(
        source_files=source_files,
        y_bins=args.y_bins,
        t_bins=args.t_bins,
        max_records_per_source=args.max_records_per_source,
    )

    train_indices, val_indices = split_indices_by_behavior(
        dataset,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    split_summary = {
        "sources": [{"name": name, "path": path} for name, path in source_files],
        "train": summary_for_indices(dataset, train_indices),
        "val": summary_for_indices(dataset, val_indices),
        "behavior_overlap": len(
            {
                normalize_behavior(dataset.samples[idx].behavior)
                for idx in train_indices
            }
            & {
                normalize_behavior(dataset.samples[idx].behavior)
                for idx in val_indices
            }
        ),
    }
    split_summary_path = os.path.join(args.output_dir, "split_summary.json")
    if is_main_process():
        save_json(split_summary_path, split_summary)

    train_weights = compute_sampling_weights(
        dataset,
        train_indices,
        source_alpha=args.source_balance_alpha,
        mode_alpha=args.mode_balance_alpha,
        y_alpha=args.y_balance_alpha,
        t_alpha=args.t_balance_alpha,
        extreme_boost=args.extreme_boost,
        extreme_low=args.extreme_low,
        extreme_high=args.extreme_high,
        max_weight_ratio=args.max_weight_ratio,
    )
    loss_weights = [float(w ** args.loss_weight_power) for w in train_weights]
    train_set = WeightedSubset(dataset, train_indices, sample_weights=loss_weights)
    val_set = WeightedSubset(dataset, val_indices, sample_weights=None)
    train_subset_weights = list(train_weights)

    collate_fn = make_collate_fn(args.pad_token_id)
    sampler_num_samples = len(train_set) if args.samples_per_epoch <= 0 else int(args.samples_per_epoch)
    if world_size > 1:
        sampler = DistributedWeightedSampler(
            weights=torch.as_tensor(train_subset_weights, dtype=torch.double),
            num_samples=sampler_num_samples,
            replacement=True,
            rank=rank,
            num_replicas=world_size,
            seed=args.seed,
        )
        val_sampler = DistributedSampler(
            val_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
    else:
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(train_subset_weights, dtype=torch.double),
            num_samples=sampler_num_samples,
            replacement=True,
        )
        val_sampler = None
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False if val_sampler is None else None,
        sampler=val_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    if is_main_process():
        print(f"Training device: {device}")
        print(f"Distributed: {'on' if world_size > 1 else 'off'} | rank={rank} world_size={world_size} local_rank={local_rank}")
        print(f"Resolved LLaDA source: {args.llada_path}")
        print(f"Resolved output dir: {args.output_dir}")
        print(f"Training samples: {len(train_indices)} | Validation samples: {len(val_indices)}")
        print(f"Split summary written to: {split_summary_path}")
        print("Training sources:")
        for source_name, path in source_files:
            print(f"  - {source_name}: {path}")

    train_dtype = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }[args.dtype]
    model = ValueModel.from_pretrained(
        args.llada_path,
        n_layers=args.n_layers,
        device=device,
        torch_dtype=train_dtype,
        mask_id=args.mask_id,
        architecture=args.value_model_arch,
        readout_heads=args.readout_heads,
        pooling_mode=args.pooling_mode,
        readout_num_queries=args.readout_num_queries,
        readout_aggregation=args.readout_aggregation,
    )

    if args.freeze_backbone:
        maybe_unfreeze_top_blocks(model, args.unfreeze_top_blocks)

    if world_size > 1:
        ddp_kwargs = {}
        if device.startswith("cuda"):
            ddp_kwargs["device_ids"] = [local_rank]
            ddp_kwargs["output_device"] = local_rank
        model = DDP(model, find_unused_parameters=False, **ddp_kwargs)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history_path = os.path.join(args.output_dir, "train_history.jsonl")
    if is_main_process():
        save_json(
            os.path.join(args.output_dir, "train_config.json"),
            {
                "args": vars(args),
                "sources": [{"name": name, "path": path} for name, path in source_files],
                "value_model_arch": args.value_model_arch,
                "distributed": world_size > 1,
                "world_size": world_size,
            },
        )

    best_val = float("inf")
    global_step = 0
    for epoch in range(args.epochs):
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        reg_total = 0.0
        pair_total = 0.0
        loss_total = 0.0
        sample_total = 0
        pair_count_total = 0
        pair_acc_weighted = 0.0
        finite_steps = 0
        skipped_steps = 0
        accum_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", disable=not is_main_process())
        for step_idx, batch in enumerate(pbar, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            t = batch["t"].to(device)
            y = batch["y"].to(device)
            sample_weight = batch["sample_weight"].to(device)
            mode_id = batch["mode_id"].to(device)
            cls_attend_to = batch["cls_attend_to"].to(device)

            pred = model(input_ids, t, attention_mask=attention_mask, cls_attend_to=cls_attend_to)
            reg_loss = smooth_l1_loss(
                pred,
                y,
                beta=args.huber_beta,
                sample_weight=sample_weight,
            )
            pair_loss, pair_count, pair_acc = batch_pairwise_ranking_loss(
                pred,
                y,
                t,
                mode_id,
                y_margin=args.ranking_y_margin,
                t_window=args.ranking_t_window,
                hinge_margin=args.ranking_hinge_margin,
                max_pairs=args.ranking_max_pairs,
            )
            loss = reg_loss + (args.ranking_lambda * pair_loss)
            loss_is_finite = bool(torch.isfinite(loss))
            if world_size > 1:
                finite_flag = torch.tensor([1 if loss_is_finite else 0], device=device, dtype=torch.int32)
                dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
                loss_is_finite = bool(int(finite_flag.item()) == 1)
            if not loss_is_finite:
                skipped_steps += 1
                optimizer.zero_grad(set_to_none=True)
                if is_main_process():
                    pbar.set_postfix(loss="nonfinite", skipped=skipped_steps)
                continue

            scaled_loss = loss / max(args.grad_accum_steps, 1)
            scaled_loss.backward()
            accum_batches += 1

            if accum_batches >= max(args.grad_accum_steps, 1):
                if args.grad_clip > 0:
                    grad_norm = clip_grad_norm_(model.parameters(), args.grad_clip)
                    if not torch.isfinite(torch.as_tensor(grad_norm)):
                        skipped_steps += 1
                        optimizer.zero_grad(set_to_none=True)
                        accum_batches = 0
                        if is_main_process():
                            pbar.set_postfix(loss="bad_grad", skipped=skipped_steps)
                        continue
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accum_batches = 0
                global_step += 1

            batch_examples = int(y.shape[0])
            reg_total += float(reg_loss.item()) * batch_examples
            pair_total += float(pair_loss.item()) * batch_examples
            loss_total += float(loss.item()) * batch_examples
            sample_total += batch_examples
            finite_steps += 1
            pair_count_total += int(pair_count)
            pair_acc_weighted += pair_acc * float(pair_count)
            if is_main_process():
                pbar.set_postfix(
                    loss=f"{loss.item():.6f}",
                    reg=f"{reg_loss.item():.6f}",
                    pair=f"{pair_loss.item():.6f}",
                )

        if accum_batches > 0:
            if args.grad_clip > 0:
                grad_norm = clip_grad_norm_(model.parameters(), args.grad_clip)
                if not torch.isfinite(torch.as_tensor(grad_norm)):
                    optimizer.zero_grad(set_to_none=True)
                    raise RuntimeError("Gradient norm became non-finite on the final accumulated step.")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        if finite_steps == 0:
            raise RuntimeError("All training steps were skipped due to non-finite loss/gradients.")

        if world_size > 1:
            train_stats = torch.tensor(
                [
                    loss_total,
                    reg_total,
                    pair_total,
                    float(sample_total),
                    float(pair_count_total),
                    pair_acc_weighted,
                    float(finite_steps),
                    float(skipped_steps),
                    float(global_step),
                ],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(train_stats, op=dist.ReduceOp.SUM)
            loss_total = float(train_stats[0].item())
            reg_total = float(train_stats[1].item())
            pair_total = float(train_stats[2].item())
            sample_total = int(round(float(train_stats[3].item())))
            pair_count_total = int(round(float(train_stats[4].item())))
            pair_acc_weighted = float(train_stats[5].item())
            finite_steps = int(round(float(train_stats[6].item())))
            skipped_steps = int(round(float(train_stats[7].item())))
            global_step = int(round(float(train_stats[8].item() / world_size)))

        val_metrics = evaluate(
            model,
            val_loader,
            device=device,
            huber_beta=args.huber_beta,
            ranking_lambda=args.ranking_lambda,
            ranking_y_margin=args.ranking_y_margin,
            ranking_t_window=args.ranking_t_window,
            ranking_hinge_margin=args.ranking_hinge_margin,
            ranking_max_pairs=args.ranking_max_pairs,
        )
        train_metrics = {
            "loss": loss_total / max(sample_total, 1),
            "regression_loss": reg_total / max(sample_total, 1),
            "pair_loss": pair_total / max(sample_total, 1),
            "pair_count": pair_count_total,
            "pair_accuracy": pair_acc_weighted / max(pair_count_total, 1),
            "finite_steps": finite_steps,
            "skipped_steps": skipped_steps,
            "samples": sample_total,
        }
        epoch_payload = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "train": train_metrics,
            "val": val_metrics,
        }
        if is_main_process():
            append_jsonl(history_path, epoch_payload)
            print(json.dumps(epoch_payload, ensure_ascii=False, indent=2))

        if torch.isfinite(torch.as_tensor(val_metrics["loss"])) and val_metrics["loss"] < best_val:
            best_val = float(val_metrics["loss"])
            best_path = os.path.join(args.output_dir, "value_model_best.pt")
            if is_main_process():
                torch.save(unwrap_model(model).state_dict(), best_path)
                save_json(
                    os.path.join(args.output_dir, "best_metrics.json"),
                    epoch_payload,
                )
                print(f"Saved new best checkpoint to {best_path}")
        else:
            if is_main_process():
                print("Validation did not improve. Skip checkpoint save.")

    latest_path = os.path.join(args.output_dir, "value_model_last.pt")
    if is_main_process():
        torch.save(unwrap_model(model).state_dict(), latest_path)
        print(f"Saved final checkpoint to {latest_path}")
    cleanup_distributed()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a value model with behavior-level validation split, balanced sampling, "
            "and a more robust regression objective. "
            "This trains the CLS-token value model used by the current steering code."
        )
    )
    parser.add_argument(
        "--llada_path",
        type=str,
        default=os.getenv("LLADA_PATH"),
        help="Local LLaDA path or HF model id. Defaults to env LLADA_PATH or GSAI-ML/LLaDA-8B-Instruct.",
    )
    parser.add_argument("--data_dir", type=str, default="data/tree_data_advanced")
    parser.add_argument(
        "--data_file",
        type=str,
        default="advanced_tree_data.json",
        help="JSON/JSONL filename under data_dir, or an absolute path.",
    )
    parser.add_argument(
        "--source",
        type=str,
        action="append",
        default=[],
        help=(
            "Optional multi-source spec. Format: name::/abs/or/rel/path . "
            "Can be passed multiple times. If omitted, falls back to --data_dir/--data_file."
        ),
    )
    parser.add_argument("--output_dir", type=str, default="models/value_model_checkpoints/v2")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="e.g. auto, cuda, cuda:0, cpu. Under torchrun, each rank uses LOCAL_RANK automatically.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--n_layers", type=int, default=27)
    parser.add_argument("--mask_id", type=int, default=126336)
    parser.add_argument(
        "--value_model_arch",
        type=str,
        default=os.getenv("VALUE_MODEL_ARCH", "readout_cls"),
        choices=["full_cls", "readout_cls"],
        help="Choose the original full CLS model or the cheaper cached CLS readout model.",
    )
    parser.add_argument(
        "--pooling_mode",
        type=str,
        default=os.getenv("VALUE_MODEL_POOLING_MODE", "auto"),
        choices=["auto", "full_cls", "readout_cls", "mean_visible", "max_visible", "last_visible"],
        help=(
            "Pooling/readout mode for value scoring. "
            "For architecture=readout_cls, supports readout_cls/mean_visible/max_visible/last_visible. "
            "For architecture=full_cls, use auto or full_cls."
        ),
    )
    parser.add_argument("--readout_heads", type=int, default=8)
    parser.add_argument("--readout_num_queries", type=int, default=1)
    parser.add_argument(
        "--readout_aggregation",
        type=str,
        default="mean",
        choices=["mean", "max", "concat"],
        help="How to merge multiple readout queries before regression. Ignored unless pooling_mode=readout_cls.",
    )
    parser.add_argument("--pad_token_id", type=int, default=126081)
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp32", "float16", "bfloat16", "float32"],
        help="Backbone dtype for the value model.",
    )
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--freeze_backbone", dest="freeze_backbone", action="store_true")
    parser.add_argument("--no_freeze_backbone", dest="freeze_backbone", action="store_false")
    parser.set_defaults(freeze_backbone=True)
    parser.add_argument(
        "--unfreeze_top_blocks",
        type=int,
        default=0,
        help="When backbone is frozen, optionally unfreeze the top N transformer blocks plus ln_f.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_records_per_source", type=int, default=-1)
    parser.add_argument("--samples_per_epoch", type=int, default=-1)

    parser.add_argument("--y_bins", type=int, default=10)
    parser.add_argument("--t_bins", type=int, default=8)
    parser.add_argument("--source_balance_alpha", type=float, default=0.5)
    parser.add_argument("--mode_balance_alpha", type=float, default=0.5)
    parser.add_argument("--y_balance_alpha", type=float, default=0.75)
    parser.add_argument("--t_balance_alpha", type=float, default=0.25)
    parser.add_argument("--extreme_boost", type=float, default=1.5)
    parser.add_argument("--extreme_low", type=float, default=0.15)
    parser.add_argument("--extreme_high", type=float, default=0.85)
    parser.add_argument("--max_weight_ratio", type=float, default=10.0)
    parser.add_argument(
        "--loss_weight_power",
        type=float,
        default=0.0,
        help="Optional extra power for using sampling weights inside the regression loss. 0 disables it.",
    )

    parser.add_argument("--huber_beta", type=float, default=0.1)
    parser.add_argument("--ranking_lambda", type=float, default=0.1)
    parser.add_argument("--ranking_y_margin", type=float, default=0.15)
    parser.add_argument("--ranking_t_window", type=float, default=0.2)
    parser.add_argument("--ranking_hinge_margin", type=float, default=0.02)
    parser.add_argument("--ranking_max_pairs", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
