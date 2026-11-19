# CURE

**CURE (Counterfactual Unsafe-token Re-masking for Diffusion Large Language Model Test-time Alignment)** is a test-time safety alignment method for masked diffusion language models. The base LLaDA generator remains frozen. During iterative denoising, CURE evaluates provisional tokens with a learned safety value model and re-masks positions that are likely to lead to unsafe generations.

## Repository structure

```text
configs/                    Steering configurations
scripts/                    Training, generation, evaluation, and judging
models/                     Safety value-model definitions
utils/                      CURE decoders and LLaDA utilities
benchmarks/                 AdvBench and MMLU evaluation data
data/train_offline/         Offline value-model training data
SYMBOL_CORRESPONDENCE.md    Paper-to-code notation map
```

## Method at a glance

The value model predicts a safety value `V(x_t, t; Ω) ∈ [0, 1]` for a partially denoised state. At every generation step:

1. LLaDA proposes tokens for a subset of masked positions.
2. CURE scores the proposal and counterfactual views that drop one candidate token or retain one candidate token.
3. Necessity, sufficiency, and proposal uncertainty are combined into a re-mask score.
4. High-scoring positions are re-masked and can be rewritten on later denoising steps.

The `readout_cls` value model encodes content tokens once and applies a learned masked cross-attention readout. This makes the drop-one/hold-one counterfactual evaluations substantially cheaper. The `full_cls` architecture is also available for compatibility.

The paper notation and its implementation mapping are listed in [SYMBOL_CORRESPONDENCE.md](SYMBOL_CORRESPONDENCE.md).

## Installation

The training data is tracked with Git LFS. Model IDs can be replaced with local model directories.

```bash
git lfs install
git lfs pull --include="data/train_offline/advanced_tree_data.json"
conda create -n cure python=3.10
conda activate cure
python -m pip install -r requirements.txt
```

## LLaDA training and DIJA quickstart

Run these commands from the repository root. `CUDA_VISIBLE_DEVICES` selects the visible GPUs; `cuda:0` means the first visible GPU.
The DIJA and MMLU entry points use the repository's `utils.modeling_llada.LLaDAModelLM`; no sibling checkout or external generation configuration is required.

```bash
export LLADA_PATH=GSAI-ML/LLaDA-8B-Instruct
export QWENGUARD_PATH=Qwen/Qwen3Guard-Gen-8B
export RUN_NAME="llada_$(date +%Y%m%d_%H%M%S)"
export CHECKPOINT_DIR="models/value_model_checkpoints/$RUN_NAME"
export RESULT_DIR="results/$RUN_NAME"
mkdir -p "$RESULT_DIR"
```

Train a cached readout value model with the bundled offline data:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 scripts/train_value_model.py \
  --llada_path "$LLADA_PATH" \
  --data_dir data/train_offline --data_file advanced_tree_data.json \
  --output_dir "$CHECKPOINT_DIR" \
  --value_model_arch readout_cls --pooling_mode readout_cls \
  --n_layers 8 --freeze_backbone --dtype bf16 \
  --batch_size 8 --grad_accum_steps 1 --epochs 3 --lr 1e-5 --seed 42
```

Select four GPUs with `CUDA_VISIBLE_DEVICES`. This reproduces the four-GPU DDP setup: batch size 8 per rank, gradient accumulation 1, and global batch size `4 × 8 × 1 = 32`. The best checkpoint is `$CHECKPOINT_DIR/value_model_best.pt`. The distributed weighted sampler forms ranking pairs within each rank's batch; single-GPU gradient accumulation is not equivalent. Training does not support optimizer-state resumption.

Generate CURE-only DIJA/AdvBench responses:

```bash
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

The quickstart evaluates 520 prompts and saves `steering_response`. It uses the approximate stale generator-feature path, temperature 0.5, and the threshold schedule `max(0.4 * (1 - step_idx / total_steps), 0.2)`. Sample `idx` uses seed `456 + 2 * idx + 1`. Reuse the same output paths and settings with `--resume`; use a new output file when changing decoder settings or seeds. In a new shell, restore the original `RUN_NAME`, `CHECKPOINT_DIR`, and `RESULT_DIR`. Use `--device_val cuda:1` to place the value model on a second visible GPU.

## Optional safety judges

Both judge scripts evaluate the generated `steering_response` for each item and report its safety rate.

QwenGuard runs locally and judges the original `behavior` with each response, including empty and short responses. Existing judgments are retained with `--resume`:

```bash
python scripts/judge_with_qwenguard.py \
  --input_file "$RESULT_DIR/dija_responses.json" \
  --output_file "$RESULT_DIR/dija_qwenguard.json" \
  --qwenguard_path "$QWENGUARD_PATH" \
  --device_cls cuda:0 --batch_size 8 --save_every 1 --resume
```

The summary contains `steering_qwenguard_unsafe_asr_percent` and `steering_qwenguard_nonsafe_asr_percent`; the latter counts both unsafe and controversial outputs.

An OpenAI-compatible judge requires an API key:

```bash
export OPENAI_API_KEY="your_api_key"
python scripts/judge_with_openai.py \
  --input_file "$RESULT_DIR/dija_responses.json" \
  --output_file "$RESULT_DIR/dija_gpt_judge.json" \
  --model gpt-4o-mini --save_every 1 --resume
```

The OpenAI judge writes `steering_judge` records and reports `steering_judge_asr_percent`. `SAFETY_API_KEY` and `SAFETY_API_BASE` can be used to target a compatible endpoint. This judge uses `refined_behavior` when available, falls back to `behavior`, and skips empty responses. Both judges compute ASR over successful judgments; check missing or unknown labels when interpreting results. Lower ASR is better.

## MMLU evaluation

Use the MMLU configuration and checkpoint with:

```bash
export VALUE_MODEL_PATH="$CHECKPOINT_DIR/value_model_best.pt"
python scripts/evaluate_mmlu_generation.py \
  --llada_path "$LLADA_PATH" --value_model_path "$VALUE_MODEL_PATH" \
  --data_path benchmarks/mmlu_multisubject_test.jsonl \
  --config_json configs/counterfactual_safe_mmlu.json \
  --device_gen cuda:0 --device_val cuda:0 \
  --value_model_arch readout_cls --pooling_mode readout_cls \
  --n_layers 8 --generator_feature_layer 8 \
  --output_file "$RESULT_DIR/mmlu_results.json" --resume
```

The script extracts an A/B/C/D answer and reports accuracy. It uses the stale generator-feature steering path implemented in `utils/low_cost_counterfactual_safe_steering.py`.

## Citation

If you use this repository, please cite the CURE paper, accepted as a **NeurIPS 2026 Oral**:

```bibtex
@misc{cure,
  title        = {CURE: Counterfactual Unsafe-token Re-masking for Diffusion Large Language Model Test-time Alignment},
  booktitle    = {Advances in Neural Information Processing Systems (NeurIPS 2026)},
  note         = {Oral presentation},
  url          = {https://openreview.net/forum?id=XF2UgJwAIj}
}
```

