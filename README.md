# CURE: Counterfactual Unsafe-token Re-masking for Diffusion Large Language Model Test-time Alignment

This repository implements **CURE**, a counterfactual value-driven test-time alignment method for masked diffusion language models (DLMs). CURE keeps the base DLM frozen and selectively re-masks risk-inducing tokens during iterative denoising, steering generation toward safe responses without modifying the pretrained generator.

## Method Overview

CURE operates in three stages:

1. **Cached Cross-Attention Value Model (§4.1)** — A time-conditioned safety value model predicts the conditional safety return $V^\star(x_t, \alpha_t; \Omega)$ of a partially denoised state. It shares the frozen DLM backbone and uses a learned readout query with masked cross-attention for efficient counterfactual evaluation.

2. **Spine-Tree Monte Carlo Rollout (§4.2)** — Training data is collected by sampling denoising trajectories from BeaverTails harmful prompts, performing K-branch stochastic rollouts at each spine state, and labeling safety returns with QwenGuard.

3. **CURE Test-time Alignment (§4.3)** — At each denoising step, CURE constructs counterfactual readout views (drop-one and hold-one), estimates token-level risk contribution $\rho_{t,i}$, and surgically re-masks high-risk positions whose combined re-mask score $\eta_{t,i}$ exceeds a threshold $\tau$.

## Repository Layout

```text
configs/      Steering configuration JSON files (threshold, temperature, weights)
scripts/      Evaluation, judge, and training entrypoints
utils/        Core CURE steering, value model loading, and generation utilities
models/       Cached cross-attention value model definition
benchmarks/   Local benchmark assets (MMLU, AdvBench, HarmBench, JBB-Behaviors)
data/         Training data; the quickstart uses the bundled offline JSON
```

## Paper–Code Symbol Correspondence

The table below maps notation from the paper to variable/parameter names in the codebase.

| Paper Symbol | Paper Description | Code Variable / Parameter | File |
|---|---|---|---|
| $x_t$ | Full generation state at step $t$ | `x`, `proposed_row`, `states` | `counterfactual_safe_steering.py` |
| $\alpha_t = 1 - m_t/L_r$ | Normalized denoising progress | `t_batch` (computed as `1.0 - remaining_mask_ratio`) | `counterfactual_safe_steering.py:144` |
| $V^\star(x_t, \alpha_t; \Omega)$ | Safety value prediction | `val_model(states, t_batch, cls_attend_to=...)` → `scores` | `counterfactual_safe_steering.py:182` |
| $U_\theta(x_t, \Omega) = 1 - V_\theta(x_t, \Omega)$ | Unsafe value | `base_unsafe = 1.0 - base_safe` | `counterfactual_safe_steering.py:372` |
| $\Omega_t$ | Visible token positions | `cls_attend_to` (boolean mask, True = visible) | `value_model.py:160` |
| $\Omega_{t,-i}$ | Visible set with token $i$ removed | drop-one state: `cls_attend_batch` row with position $i$ masked out | `counterfactual_safe_steering.py:330–345` |
| $\Omega_{t,i}$ | Visible set with only token $i$ | hold-one state: `cls_attend_batch` row with only position $i$ visible | `counterfactual_safe_steering.py:330–345` |
| $\Delta^{\text{nec}}_t(i)$ | Necessity: risk reduction from removing $i$ | `necessity_unsafe = clamp(base_unsafe - drop_unsafe, 0, 1)` | `counterfactual_safe_steering.py:377` |
| $\Delta^{\text{suf}}_t(i)$ | Sufficiency: risk from $i$ alone | `sufficiency_unsafe = clamp(hold_unsafe, 0, 1)` | `counterfactual_safe_steering.py:379` |
| $\lambda_{\text{nec}}, \lambda_{\text{suf}}$ | Necessity/sufficiency weights | `cfg.necessity_weight`, `cfg.sufficiency_weight` | Config: `necessity_weight`, `sufficiency_weight` |
| $\rho_{t,i}$ | Token-level counterfactual unsafe value | `unsafe_score = λ_nec * necessity + λ_suf * sufficiency` | `counterfactual_safe_steering.py:380–383` |
| $p_{t,i}$ | DLM proposal probability for token $i$ | `token_probs_row` (from softmax of noisy logits) | `counterfactual_safe_steering.py:223` |
| $\lambda_{\text{risk}}, \lambda_{\text{unc}}$ | Risk and uncertainty weights in $\eta$ | `cfg.unsafe_weight`, `cfg.uncertainty_weight` | Config: `unsafe_weight`, `uncertainty_weight` |
| $\eta_{t,i}$ | Final re-masking score | `remask_score = unsafe_weight * unsafe_score + uncertainty_weight * uncertainty_score` | `counterfactual_safe_steering.py:388–391` |
| $\tau$ | Re-masking threshold | `cfg.remask_threshold` (with optional linear decay and floor) | Config: `remask_threshold`, `remask_threshold_floor` |
| $B$ | Max positions to re-mask per step | `max_remask = ceil(n_new * cfg.max_remask_ratio)` | `counterfactual_safe_steering.py:398` |
| $\mathcal{R}_t$ | Set of re-masked positions | `eligible` (positions with $\eta \geq \tau$, capped at $B$) | `counterfactual_safe_steering.py:396` |
| $\mathcal{A}_t$ | Auditable token set | `new_positions` + `historical_audit_topk` most uncertain history tokens | `counterfactual_safe_steering.py` |
| $T$ | Total denoising steps | `cfg.steps` | Config: `steps` |
| $L_r$ | Response generation length | `cfg.gen_length` | Config: `gen_length` |
| $e_t = \text{MLP}_{\text{time}}(\alpha_t)$ | Time embedding | `self.time_embed(t)` (2-layer MLP with SiLU) | `value_model.py:79–83` |
| $q_{\text{safe}}$ | Learned safe readout query | `self.cls_queries` | `value_model.py:99` |
| Regression head | $V \in [0,1]$ output | `self.regression_head` (Linear→SiLU→Linear→SiLU→Linear→Sigmoid) | `value_model.py:125–132` |
| $\mathcal{L}_{\text{reg}}$ | Smooth L1 regression loss | Training loss in `train_value_model_v2.py` |
| $\mathcal{L}_{\text{margin}}$ | Pairwise margin ranking loss | `λ_margin = 0.1` in training script |
| Suppression | Prevent re-commitment to same token | `cfg.suppression_value` (set logit to $-\infty$) | Config: `suppression_value` |

## Installation

Install [Git LFS](https://git-lfs.com/) to download the training data, then run from the repository root:

```bash
git lfs install
git lfs pull --include="data/train_offline/advanced_tree_data.json"
conda create -n cure python=3.10
conda activate cure
pip install -r requirements.txt
```

## LLaDA Quickstart

The following commands train a value model, generate CURE-only responses for DIJA-AdvBench, and optionally judge them. Run from the repository root. Select GPUs with `CUDA_VISIBLE_DEVICES`; `cuda:0` refers to the first visible GPU.

```bash
export LLADA_PATH=GSAI-ML/LLaDA-8B-Instruct
export QWENGUARD_PATH=Qwen/Qwen3Guard-Gen-8B
export RUN_NAME="llada_$(date +%Y%m%d_%H%M%S)"
export CHECKPOINT_DIR="models/value_model_checkpoints/$RUN_NAME"
export RESULT_DIR="results/$RUN_NAME"
mkdir -p "$RESULT_DIR"

torchrun --standalone --nnodes=1 --nproc_per_node=4 scripts/train_value_model_v2.py \
  --llada_path "$LLADA_PATH" \
  --data_dir data/train_offline --data_file advanced_tree_data.json \
  --output_dir "$CHECKPOINT_DIR" \
  --value_model_arch readout_cls --pooling_mode readout_cls \
  --n_layers 8 --freeze_backbone --dtype bf16 \
  --batch_size 8 --grad_accum_steps 1 --epochs 3 --lr 1e-5 --seed 42

export VALUE_MODEL_PATH="$CHECKPOINT_DIR/value_model_best.pt"
python scripts/evaluate_dija_advbench_generation.py \
  --llada_path "$LLADA_PATH" --value_model_path "$VALUE_MODEL_PATH" \
  --attack_json_path benchmarks/AdvBench/data/prompts_dija.json \
  --config_json configs/quickstart_dija.json \
  --output_file "$RESULT_DIR/dija_responses.json" \
  --device_gen cuda:0 --device_val cuda:0 \
  --value_model_arch readout_cls --pooling_mode readout_cls \
  --n_layers 8 --generator_feature_steering --generator_feature_stale \
  --generator_feature_time_injection none \
  --save_every 1 --resume --seed 456
```

DIJA uses the approximate stale generator-feature path, temperature 0.5, and the threshold schedule `max(0.4 * (1 - step_idx / total_steps), 0.2)`. The generation base seed is 456; steering sample `idx` uses `456 + 2 * idx + 1`. DIJA and MMLU load the generator with the repository's `utils.modeling_llada.LLaDAModelLM`; the model directory supplies configuration and weights, not the generator implementation. Use a new output file when switching decoder mode or seed.

Training uses the bundled `data/train_offline/advanced_tree_data.json`. Hugging Face model IDs can be replaced with local model directories. The training command matches the original four-GPU DDP setup: select four GPUs with `CUDA_VISIBLE_DEVICES`, using batch size 8 per rank and no gradient accumulation (global batch size `4 × 8 × 1 = 32`). This uses the distributed weighted sampler and per-rank ranking pairs; single-GPU gradient accumulation is not equivalent. Optimizer state cannot be resumed. Generation saves `steering_response` only and can resume from the same output file. To resume in a new shell, restore the original `RUN_NAME`, `CHECKPOINT_DIR`, and `RESULT_DIR`; do not generate a new `RUN_NAME`.

Optional local QwenGuard judging uses the original `behavior` with each response, not `refined_behavior`. All responses, including empty or short responses, are sent to the judge. Use a new output file when switching from judgments made with the attack-wrapped question; `--resume` retains existing judgments.

```bash
python scripts/judge_compare_with_qwenguard.py \
  --input_file "$RESULT_DIR/dija_responses.json" \
  --output_file "$RESULT_DIR/dija_qwenguard.json" \
  --qwenguard_path "$QWENGUARD_PATH" --device_cls cuda:0 \
  --batch_size 8 --save_every 1 --resume
```

Optional GPT judging requires `OPENAI_API_KEY`:

```bash
export OPENAI_API_KEY="your_api_key"
python scripts/judge_compare_with_openai.py \
  --input_file "$RESULT_DIR/dija_responses.json" \
  --output_file "$RESULT_DIR/dija_gpt_judge.json" \
  --model gpt-4o-mini --save_every 1 --resume
```
