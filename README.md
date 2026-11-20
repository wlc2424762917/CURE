# CURE

**CURE (Counterfactual Unsafe-token Re-masking for Diffusion Large Language Model Test-time Alignment)** aligns masked diffusion language models at inference time with a frozen LLaDA generator and a learned safety value model.

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

## Method

At each denoising step, LLaDA proposes tokens, the value model scores counterfactual token views, and high-risk positions are re-masked.

The paper notation and its implementation mapping are listed in [SYMBOL_CORRESPONDENCE.md](SYMBOL_CORRESPONDENCE.md).

## Installation

Training data uses Git LFS. Model IDs may be replaced with local paths.

```bash
git lfs install
git lfs pull --include="data/train_offline/advanced_tree_data.json"
conda create -n cure python=3.10
conda activate cure
python -m pip install -r requirements.txt
```

## LLaDA training and DIJA quickstart

Run from the repository root. Set model paths as needed.

```bash
export LLADA_PATH=GSAI-ML/LLaDA-8B-Instruct
export QWENGUARD_PATH=Qwen/Qwen3Guard-Gen-8B
export RUN_NAME="llada_$(date +%Y%m%d_%H%M%S)"
export CHECKPOINT_DIR="models/value_model_checkpoints/$RUN_NAME"
export RESULT_DIR="results/$RUN_NAME"
mkdir -p "$RESULT_DIR"
```

Train the value model:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 scripts/train_value_model.py \
  --llada_path "$LLADA_PATH" \
  --data_dir data/train_offline --data_file advanced_tree_data.json \
  --output_dir "$CHECKPOINT_DIR" \
  --value_model_arch readout_cls --pooling_mode readout_cls \
  --n_layers 8 --freeze_backbone --dtype bf16 \
  --batch_size 8 --grad_accum_steps 1 --epochs 3 --lr 1e-5 --seed 42
```

Generate DIJA/AdvBench responses:

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

## Safety judges

```bash
python scripts/judge_with_qwenguard.py \
  --input_file "$RESULT_DIR/dija_responses.json" \
  --output_file "$RESULT_DIR/dija_qwenguard.json" \
  --qwenguard_path "$QWENGUARD_PATH" \
  --device_cls cuda:0 --batch_size 8 --save_every 1 --resume
```

```bash
export OPENAI_API_KEY="your_api_key"
python scripts/judge_with_openai.py \
  --input_file "$RESULT_DIR/dija_responses.json" \
  --output_file "$RESULT_DIR/dija_gpt_judge.json" \
  --prompt_config configs/openai_judge_prompt.yaml \
  --model gpt-4o-mini --save_every 1 --resume
```

## MMLU evaluation

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

