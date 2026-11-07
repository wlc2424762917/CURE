# LLaDA Training and DIJA Evaluation

Train a LLaDA value model, generate CURE-only DIJA responses, and judge them from this repository. No sibling DiffuGuard checkout or generation TXT is required. DIJA and MMLU use the repository's `utils.modeling_llada.LLaDAModelLM` implementation with configuration and weights from `LLADA_PATH`. Select GPUs with `CUDA_VISIBLE_DEVICES`; `cuda:0` means the first visible GPU.

## Setup

```bash
cd /path/to/CURE
git lfs install
git lfs pull --include="data/train_offline/advanced_tree_data.json"
conda create -n cure python=3.10  # Once, if needed.
conda activate cure
python -m pip install -r requirements.txt
export LLADA_PATH=GSAI-ML/LLaDA-8B-Instruct
export QWENGUARD_PATH=Qwen/Qwen3Guard-Gen-8B
export RUN_NAME="llada_$(date +%Y%m%d_%H%M%S)"
export CHECKPOINT_DIR="models/value_model_checkpoints/$RUN_NAME"
export RESULT_DIR="results/$RUN_NAME"
mkdir -p "$RESULT_DIR"
```

Install [Git LFS](https://git-lfs.com/) before setup. Model IDs download through Hugging Face; you can instead set either variable to a local model directory. Training uses `data/train_offline/advanced_tree_data.json`.

## Train

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 scripts/train_value_model_v2.py \
  --llada_path "$LLADA_PATH" \
  --data_dir data/train_offline --data_file advanced_tree_data.json \
  --output_dir "$CHECKPOINT_DIR" \
  --value_model_arch readout_cls --pooling_mode readout_cls \
  --n_layers 8 --freeze_backbone --dtype bf16 \
  --batch_size 8 --grad_accum_steps 1 \
  --epochs 3 --lr 1e-5 --seed 42
```

Select four GPUs with `CUDA_VISIBLE_DEVICES`. This matches the original four-GPU DDP setup: batch size 8 per rank, gradient accumulation 1, and global batch size `4 × 8 × 1 = 32`. It uses the distributed weighted sampler and forms ranking pairs within each rank's batch; single-GPU gradient accumulation is not equivalent. Keep this layout to reproduce the original training setup. The best checkpoint is `value_model_best.pt`. Training does not support optimizer-state resumption.

## Generate DIJA Responses

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

Evaluates all 520 prompts with CURE only and saves `steering_response`. CURE uses the 64-step steering config with approximate stale generator features and no time injection. Temperature is 0.5; the threshold schedule is `max(0.4 * (1 - step_idx / total_steps), 0.2)`. Generation keeps the original CURE seed schedule: sample `idx` uses `456 + 2 * idx + 1`. Training keeps seed 42. Decode settings come from `--config_json`. Completed CURE responses are skipped with `--resume`, including those in older paired result files. Use `--device_val cuda:1` to split models across two visible GPUs.

## QwenGuard Judge (Optional)

Judges each response against the original `behavior`, not the attack-wrapped `refined_behavior`, including empty and short responses. Use a new output file if earlier judgments used `refined_behavior`; `--resume` retains existing judgments.

```bash
python scripts/judge_compare_with_qwenguard.py \
  --input_file "$RESULT_DIR/dija_responses.json" \
  --output_file "$RESULT_DIR/dija_qwenguard.json" \
  --qwenguard_path "$QWENGUARD_PATH" \
  --device_cls cuda:0 --batch_size 8 --save_every 1 --resume
```

Read `steering_qwenguard_unsafe_asr_percent` in `summary`. `steering_qwenguard_nonsafe_asr_percent` includes unsafe and controversial labels. Missing baseline responses are not judged or included in the summary.

## GPT Judge (Optional)

```bash
export OPENAI_API_KEY="your_api_key"
# Optional: set OPENAI_BASE_URL for a custom endpoint.

python scripts/judge_compare_with_openai.py \
  --input_file "$RESULT_DIR/dija_responses.json" \
  --output_file "$RESULT_DIR/dija_gpt_judge.json" \
  --model gpt-4o-mini --save_every 1 --resume
```

Choose a supported model with `--model`. Existing `SAFETY_API_KEY` / `SAFETY_API_BASE` override the corresponding OpenAI variables. No local GPU is needed.

Read `steering_judge_asr_percent` in `summary`. GPT uses binary safe/unsafe labels and saves explanations.

Both judges report ASR over successful judgments; lower is better. Check for missing labels before interpreting results. To resume generation or judging, reuse the original paths and settings with `--resume`. In a new shell, restore the original `CHECKPOINT_DIR` and `RESULT_DIR`; do not generate a new `RUN_NAME`. Use new output files when changing settings.
