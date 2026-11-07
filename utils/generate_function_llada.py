# -*- coding: utf-8 -*-
import re
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

# =============== Basic Utilities ===============

def _ids_from_phrases(tokenizer, phrases: Iterable[str]) -> set:
    ids = set()
    for ph in phrases:
        toks = tokenizer(ph, add_special_tokens=False).input_ids
        # More robust: collect all tokens of the full phrase (avoid taking only the first which could yield an empty set)
        for t in toks:
            ids.add(t)
    return ids

def build_cue_id_sets(tokenizer,
                      accept_cues: Optional[Iterable[str]] = None,
                      refuse_cues: Optional[Iterable[str]] = None) -> Tuple[set, set]:
    if accept_cues is None:
        accept_cues = ["Sure", "Of course", "Here", "Step", "1.", "First", "Proceed", "Let's"]
    if refuse_cues is None:
        refuse_cues = ["Sorry", "I can't", "cannot", "won't", "illegal", "harmful", "not assist", "refuse"]
    return _ids_from_phrases(tokenizer, accept_cues), _ids_from_phrases(tokenizer, refuse_cues)

def safety_perplexity_from_logits(logits: torch.Tensor,
                                  positions_mask: torch.Tensor,
                                  accept_ids: set,
                                  refuse_ids: set) -> float:
    with torch.no_grad():
        probs = torch.softmax(logits, dim=-1)
        if len(accept_ids) == 0 or len(refuse_ids) == 0:
            return 0.0
        accept_p = probs[..., list(accept_ids)].sum(dim=-1)
        refuse_p = probs[..., list(refuse_ids)].sum(dim=-1)
        conflict = 4.0 * accept_p * refuse_p
        mask = positions_mask.to(conflict.dtype)
        denom = mask.sum().clamp_min(1.0)
        sp = (conflict * mask).sum() / denom
        return float(sp.item())

def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().to(torch.float32)
    b = b.detach().to(torch.float32)
    a = a / (a.norm(p=2) + 1e-12)
    b = b / (b.norm(p=2) + 1e-12)
    return float(1.0 - torch.dot(a, b).item())

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

SPECIAL_TOKEN_PATTERN = r"<mask:(\d+)>"

def _mask_token_str(tokenizer, mask_id: int) -> str:
    try:
        t = tokenizer.convert_ids_to_tokens(mask_id)
        if isinstance(t, str) and len(t) > 0:
            return t
    except Exception:
        pass
    return "<|mask|>"

def expand_span_masks_like_mmdm(text: str,
                                mask_token: str,
                                mask_counts: int = 0,
                                add_tail_if_missing: bool = True) -> str:
    def repl(m):
        n = max(int(m.group(1)), 0)
        return mask_token * n
    out = re.sub(SPECIAL_TOKEN_PATTERN, repl, str(text))
    # If the sequence contains no mask but a tail is requested, append mask tail
    if add_tail_if_missing and (mask_token not in out) and (mask_counts > 0):
        out = out + (mask_token * mask_counts)
    return out

# =============== Enhanced Generator (keep all strategies) ===============

@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,                   # token ids, shape [B,L]
    steps=64,
    gen_length=128,
    block_length=128,
    temperature=0.5,
    cfg_scale=0.0,
    remasking="low_confidence",   # ["low_confidence","random","rate","adaptive","adaptive_step"]
    mask_id=126336,
    random_rate=0.0,
    injection_step=None,
    alpha0: float = 0.3,
    sp_mode: str = "off",         # ["off","logit","hidden"]
    sp_threshold: float = 0.35,
    refinement_steps: int = 8,
    remask_ratio: float = 0.9,
    suppression_value: float = 1e6,
    correct_only_first_block: bool = True,
    accept_cues: Optional[Iterable[str]] = None,
    refuse_cues: Optional[Iterable[str]] = None,
    baseline_hidden: Optional[torch.Tensor] = None,
    fill_all_masks: bool = False,       # <<<<<< Key: if True = fill entire sequence + do not append mask tail
    debug_print: bool = False,
    attention_mask: Optional[torch.Tensor] = None,
    attack_method: str = "none",                 # "none" | "pad"
    pad_anchors: Optional[Iterable[str]] = None, # Structural anchor phrases
    pad_positions: Optional[Iterable[int]] = None, # Start offsets relative to suffix (aligned to anchors)
    pad_in_uncond: bool = True,    
    protected_index: Optional[torch.Tensor] = None,
    logits_eos_inf: bool = False,
    confidence_eos_eot_inf: bool = False,
    eos_token_id: int = 126081,
    eot_token_id: int = 126348,
    num_transfer_tokens: Optional[int] = None,
):
    """
    LLaDA 风格的“mask 迭代填充”(iterative masked decoding)生成器。

    核心思想（从高层理解即可）：
    - **输入**：`prompt` 是已经给定的前缀 token（形状通常是 [B, prompt_len]）。
    - **扩展**：在尾部追加 `gen_length` 个 `mask_id`（若 `gen_length>0`），表示需要生成的空位。
    - **按步迭代**：每一步前向得到全序列 logits，基于某种 **remasking 策略**（低置信度/随机/混合/随步衰减）
      在当前 block 内挑选一部分 mask 位置“落子”(transfer)，把这些位置从 mask 替换成模型给出的 token。
    - **分块(block)生成**：当 `gen_length` 很大时，将尾部按 `block_length` 切块，`steps` 会平均分配到每个块，
      每个块内部做若干步迭代填充；`fill_all_masks=True` 时则把全序列当作一个块处理。
    - **CFG（Classifier-Free Guidance）**：当 `cfg_scale>0` 时，额外构造无条件分支 `un_x`（把 prompt 部分 mask 掉）
      并用 guidance 公式融合 logits，提高条件生成强度。
    - **安全自检与纠偏**：
      - `sp_mode="logit"`：用 logits 计算“接受/拒绝提示词”(accept/refuse cues)的冲突度作为分数；
      - `sp_mode="hidden"`：用 hidden mean 与 `baseline_hidden` 的余弦距离作为分数；
      分数超过 `sp_threshold` 则认为 unsafe，触发 refinement：随机重 mask 一部分位置并再次迭代，
      同时对被重 mask 的原 token 做 logits 抑制（`suppression_value`）以推动修正。

    重要形状/约束：
    - `prompt`: [B, L] 的 token ids；内部会扩展到 [B, L + gen_length]。
    - `attention_mask`/`protected_index` 若提供，会对齐到扩展后的长度；`protected_index=True` 的位置不会被填充/重 mask。
    - 当 `effective_gen_length>0` 时要求 `effective_gen_length % block_length == 0`（保证按块整齐切分）。

    返回：
    - 生成后的 token ids `x`，形状与内部扩展后的序列长度一致。
    """
    device = next(model.parameters()).device

    # x 是当前正在被迭代更新的完整序列（包含 prompt + 可选的 mask 尾部）。
    x = prompt.clone().to(device)   # [B, prompt_len]
    effective_gen_length = 0
    if int(gen_length) > 0:
        # 尾部追加全 mask，占位等待后续逐步“落子”填充。
        tail = torch.full(
            (prompt.shape[0], int(gen_length)),
            mask_id, dtype=torch.long, device=device
        )                           # [B, gen_length] all <mask>
        x = torch.cat([x, tail], dim=1)
        effective_gen_length = int(gen_length)


    # prompt_index=True 表示“原始 prompt token”位置（非 mask）。
    # 后面构造 CFG 的 unconditional 分支时，会用它决定哪些位置要被 mask 掉。
    prompt_index = x != mask_id
    # --- Align attention_mask to the length of x (HF expects 2D mask [B, L]) ---
    am = None
    if attention_mask is not None:
        am = attention_mask.to(device)
        # Convert to bool; HF also accepts long/int, but bool is more consistent
        if am.dtype != torch.bool:
            am = am != 0
        if am.shape[1] < x.shape[1]:
            # attention_mask 比 x 短：在右侧补 1（可见），与追加的尾部 token 对齐。
            pad_len = x.shape[1] - am.shape[1]
            pad = torch.ones(am.size(0), pad_len, dtype=torch.bool, device=device)
            am = torch.cat([am, pad], dim=1)
        elif am.shape[1] > x.shape[1]:
            # attention_mask 比 x 长：截断到 x 的长度（避免 HF 报 shape 不匹配）。
            am = am[:, :x.shape[1]]
    # === Align protected_index to batch/length of x ===
    if protected_index is not None:
        pi = protected_index.to(device)
        # Normalize to bool mask
        if pi.dtype != torch.bool:
            pi = pi != 0
        # Align batch dim: if provided as [1, L] while x is [B, L], expand to B
        if pi.shape[0] != x.shape[0]:
            if pi.shape[0] == 1:
                pi = pi.expand(x.shape[0], -1).contiguous()
            else:
                # Non-1 batch cannot be auto-broadcast; fallback to taking the first row
                pi = pi[:1].expand(x.shape[0], -1).contiguous()
        # Align sequence length
        if pi.shape[1] < x.shape[1]:
            # protected_index 比 x 短：右侧补 0（不保护），让新增 token 默认可被填充。
            pad_len = x.shape[1] - pi.shape[1]
            pad = torch.zeros(pi.size(0), pad_len, dtype=torch.bool, device=device)
            pi = torch.cat([pi, pad], dim=1)
        elif pi.shape[1] > x.shape[1]:
            # protected_index 比 x 长：截断（避免越界/shape 不一致）。
            pi = pi[:, :x.shape[1]]
        protected_index = pi

    # >>> New: PAD pre-injection (before denoising starts)
    # uncond_prompt_index 控制 unconditional 分支里哪些位置会被 mask：
    # - 默认用 prompt_index（即“看到”目前序列里所有非 mask 的 token）
    # - 在 PAD 攻击下，可以选择让 unconditional 分支不“看到”插入的 anchors（pad_in_uncond=False）
    uncond_prompt_index = prompt_index  # Default: unconditional branch uses the initial prompt_index
    if attack_method.lower() == "pad":
        # 1) Prepare anchor phrases
        anchors = list(pad_anchors) if pad_anchors is not None else ["Step 1:", "Step 2:", "Step 3:"]
        # 2) Compute injection positions (relative to suffix: starting at prompt_len)
        after_prompt_len = x.shape[1] - prompt.shape[1]
        if pad_positions is None:
            # Even spacing: split suffix into (m+1) segments, take each segment start as insert position
            m = len(anchors)
            gap = max(after_prompt_len // (m + 1), 1)
            gap = gap // 1.5
            offsets = [(i + 1) * gap for i in range(m)]
        else:
            offsets = list(pad_positions)

        # 3) Actual write-in (avoid OOB; no special tokens added)
        for rel, text in zip(offsets, anchors):
            ids = tokenizer(text, add_special_tokens=False).input_ids
            s = prompt.shape[1] + int(rel)
            e = s + len(ids)
            if 0 <= s < x.shape[1] and e <= x.shape[1]:
                x[:, s:e] = torch.tensor(ids, dtype=torch.long, device=x.device).unsqueeze(0)

        # 4) Whether to let the unconditional branch “see” the anchors
        if not pad_in_uncond:
            # Let the unconditional branch treat anchors as non-prompt (=> will be masked)
            uncond_prompt_index = (x != mask_id)
        # If pad_in_uncond=True, keep the original prompt_index (uncond branch keeps anchors)

    # === Block planning: when fill_all_masks=True, treat as a single block and keep old behavior ===
    assert block_length > 0
    if effective_gen_length <= 0:
        # 没有尾部生成长度（纯填 prompt 内的 mask 或者只做自检/纠偏逻辑）
        num_blocks = 1
    else:
        assert effective_gen_length % block_length == 0
        num_blocks = max(effective_gen_length // block_length, 1)

    # Evenly distribute total steps across blocks; at least 1 per block
    steps_per_block = max(int(steps) // int(num_blocks), 1)

    # 用于 sp_mode="logit"：把一些“可能表示接受/拒绝”的提示词映射到 token id 集合。
    accept_ids, refuse_ids = build_cue_id_sets(tokenizer, accept_cues, refuse_cues)

    first_step_block_hidden_mean = None
    use_hidden_detection = (sp_mode == "hidden") and (baseline_hidden is not None)
    warned_no_hidden = False

    for num_block in range(num_blocks):
        if fill_all_masks:
            block_start, block_end = 0, x.shape[1]                    # <<<<<< cover the entire sequence
        else:
            block_start = prompt.shape[1] + num_block * block_length  # cover only the tail generation area
            block_end   = prompt.shape[1] + (num_block + 1) * block_length

        # For pure in-place mask filling there is no generated suffix, so cover all
        # existing masks. For normal generation, keep the official LLaDA block
        # boundaries; expanding every block to all masks destroys block decoding.
        if effective_gen_length <= 0:
            global_mask_pos = (x == mask_id).nonzero(as_tuple=False)
            if global_mask_pos.numel() > 0:
                first = int(global_mask_pos[:, 1].min().item())
                last  = int(global_mask_pos[:, 1].max().item()) + 1
                block_start = min(block_start, first)
                block_end   = max(block_end,   last)

        # 当前 block 内还有多少 mask，需要在每一步转移(落子)多少个 token。
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        transfer_schedule = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            if i == injection_step:
                injection_map = {0: "Sorry"}
                if debug_print:
                    print("Injecting jailbreak tokens...", flush=True)
                for relative_pos, text in injection_map.items():
                    injection_ids = tokenizer(text, add_special_tokens=True).input_ids
                    absolute_start_pos = prompt.shape[1] + relative_pos
                    absolute_end_pos = absolute_start_pos + len(injection_ids)
                    if 0 <= absolute_start_pos < x.shape[1] and absolute_end_pos <= x.shape[1]:
                        x[:, absolute_start_pos:absolute_end_pos] = torch.tensor(
                            injection_ids, dtype=torch.long, device=x.device
                        ).unsqueeze(0)

            # mask_index=True 的位置是“尚未填充”的 token（需要模型预测）。
            mask_index = (x == mask_id)
            if num_transfer_tokens is not None and not mask_index[:, block_start:block_end].any():
                break

            if cfg_scale > 0.0:
                # CFG：构造 unconditional 序列，把指定位置 mask 掉，拼接后一次 forward 出两份 logits。
                un_x = x.clone()
                # un_x[prompt_index] = mask_id
                un_x[uncond_prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                # Note: also concatenate attention mask if present
                am_ = None if am is None else torch.cat([am, am], dim=0)
                out = model(x_, attention_mask=am_, 
                            output_hidden_states=use_hidden_detection, return_dict=True)
                logits = out.logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                # guidance 公式：un + (s+1)(cond-un)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)

                hidden_states = None
                if use_hidden_detection and hasattr(out, "hidden_states") and out.hidden_states is not None:
                    # 只保留 conditional 分支的 hidden（这里约定 batch=1 为主；取 [:1]）。
                    hidden_states = [hs[:1] for hs in out.hidden_states]
                elif use_hidden_detection and not warned_no_hidden:
                    print("[Self-Detection] hidden_states unavailable; fallback to logits.", flush=True)
                    warned_no_hidden = True
            else:
                out = model(x, attention_mask=am, 
                            output_hidden_states=use_hidden_detection, return_dict=True)
                logits = out.logits
                if use_hidden_detection and hasattr(out, "hidden_states") and out.hidden_states is not None:
                    hidden_states = out.hidden_states
                else:
                    hidden_states = None
                    if use_hidden_detection and not warned_no_hidden:
                        print("[Self-Detection] hidden_states unavailable; fallback to logits.", flush=True)
                        warned_no_hidden = True

            if (use_hidden_detection) and (i == 0) and (hidden_states is not None) and (len(hidden_states) > 0):
                # hidden 检测：取最后一层 hidden，在当前 block 内做 mean，作为该 block 的“语义指纹”。
                last_h = hidden_states[-1]
                h_block = last_h[:, block_start:block_end, :]
                first_step_block_hidden_mean = h_block.mean(dim=1).squeeze(0).detach()

            if logits_eos_inf:
                logits = logits.clone()
                logits[:, :, int(eos_token_id)] = -torch.inf

            # Gumbel 噪声让 argmax 采样带一点随机性；temperature=0 则退化为确定性 argmax。
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            # 计算每个位置的“模型置信度”：取 softmax 后被选中 token 的概率。
            p = F.softmax(logits, dim=-1)
            model_confidence = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            if confidence_eos_eot_inf:
                special_pred = (x0 == int(eos_token_id)) | (x0 == int(eot_token_id))
                model_confidence = model_confidence.masked_fill(special_pred, -float("inf"))
            R = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)

            # remasking 策略：决定“优先填哪些位置”的排序分数 x0_p
            # - low_confidence：用置信度，后续 topk 选最大 => 趋向先填高置信度（更稳）
            # - random：纯随机
            # - rate/adaptive/adaptive_step：在置信度与随机之间插值（alpha 可常数或随步衰减）
            if remasking == "low_confidence":
                x0_p = model_confidence
            elif remasking == "random":
                x0_p = R
            elif remasking == "rate":
                x0_p = (1 - random_rate) * model_confidence + random_rate * R
            elif remasking == "adaptive":
                alpha = torch.clamp(torch.tensor(alpha0, device=x0.device, dtype=model_confidence.dtype), 0.0, 1.0)
                x0_p = (1 - alpha) * model_confidence + alpha * R
            elif remasking == "adaptive_step":
                frac = 1.0 - (i / (steps_per_block - 1)) if steps_per_block > 1 else 1.0
                alpha = torch.clamp(torch.tensor(alpha0 * frac, device=x0.device, dtype=model_confidence.dtype), 0.0, 1.0)
                x0_p = (1 - alpha) * model_confidence + alpha * R
            else:
                raise NotImplementedError(remasking)

            # Only select tokens within the current block
            # conf：把 block 外的位置置为 -inf，确保 topk 只会从 block 内的 mask 位置选。
            conf = x0_p.clone()
            conf[:, :block_start] = -np.inf
            conf[:, block_end:]   = -np.inf
            if protected_index is not None:
                # protected 位置直接排除，不参与填充。
                conf = conf.masked_fill(protected_index, -float("inf"))

            # x0：只在 mask 位置用模型预测替换，其余位置保持原 token（避免误改 prompt/已填内容）。
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, conf, -np.inf)

            # transfer_index：每步从 mask 位置中选 k 个落子（k 由 get_num_transfer_tokens 分配）。
            transfer_index = torch.zeros_like(x, dtype=torch.bool, device=x.device)
            for j in range(confidence.shape[0]):
                available = int(torch.isfinite(confidence[j]).sum().item())
                k = int(transfer_schedule[j, i].item()) if num_transfer_tokens is None else int(num_transfer_tokens)
                k = min(k, available)
                if k <= 0:
                    continue
                _, select_index = torch.topk(confidence[j], k=k)
                transfer_index[j, select_index] = True
            # 执行落子：仅更新被选中的位置。
            x[transfer_index] = x0[transfer_index]

        unsafe_flag = False
        detection_score = 0.0

        if sp_mode != "off":
            # positions_mask：只在当前 block 范围内计算安全分数（避免 prompt 部分干扰）。
            positions_mask = torch.zeros_like(x[:, :], dtype=torch.bool, device=x.device)
            positions_mask[:, block_start:block_end] = True
            sp_logit = safety_perplexity_from_logits(logits, positions_mask, accept_ids, refuse_ids)
            detection_score = sp_logit
            if sp_logit >= sp_threshold:
                unsafe_flag = True

            if use_hidden_detection and (first_step_block_hidden_mean is not None) and (baseline_hidden is not None):
                # hidden 模式：用 block hidden mean 与 baseline 的余弦距离做检测分数。
                dist = cosine_distance(first_step_block_hidden_mean, baseline_hidden)
                detection_score = dist
                if dist >= sp_threshold:
                    unsafe_flag = True

        if debug_print:
            print(f"[Block {num_block+1}/{num_blocks}] Safety score={detection_score:.3f}, unsafe={unsafe_flag}", flush=True)

        if unsafe_flag and (not correct_only_first_block or num_block == 0):
            if debug_print:
                print(f"--> Refinement Phase (steps={refinement_steps}, remask_ratio={remask_ratio})", flush=True)

            # Randomly choose positions within the block for re-masking (excluding protected positions)
            # refinement 的目标：把 block 内一部分位置重新 mask，再迭代填充，尝试“纠偏”unsafe 输出。
            eligible = torch.zeros(x.shape[1], dtype=torch.bool, device=x.device)
            eligible[block_start:block_end] = True
            if protected_index is not None:
                eligible &= ~protected_index[0]  # For batch size 1 using [0]; extend as needed for multi-batch

            cand = torch.nonzero(eligible, as_tuple=False).squeeze(1)
            num_to_remask = int(cand.numel() * float(remask_ratio))

            if num_to_remask > 0 and cand.numel() > 0:
                perm = cand[torch.randperm(cand.numel(), device=x.device)[:num_to_remask]]
                global_indices_to_remask = perm
                original_token_ids_at_remasked_pos = x[:, global_indices_to_remask].clone()
                x[:, global_indices_to_remask] = mask_id

                refinement_mask_index = (x[:, block_start:block_end] == mask_id)
                num_refine_transfer = get_num_transfer_tokens(refinement_mask_index, max(int(refinement_steps), 1))

                for r_step in range(max(int(refinement_steps), 1)):
                    mask_index = (x == mask_id)

                    if cfg_scale > 0.0:
                        un_x = x.clone()
                        un_x[uncond_prompt_index] = mask_id
                        x_ = torch.cat([x, un_x], dim=0)
                        am_ = None if am is None else torch.cat([am, am], dim=0)
                        out = model(x_, attention_mask=am_)
                        logits = out.logits
                        logits, un_logits = torch.chunk(logits, 2, dim=0)
                        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                    else:
                        logits = model(x,attention_mask=am).logits

                    if torch.isfinite(torch.tensor(suppression_value)):
                        # 对“被重 mask 的位置的原 token”做负向抑制，
                        # 直觉上是在鼓励模型不要再回到原先导致 unsafe 的 token 组合。
                        logits[
                            0,
                            global_indices_to_remask,
                            original_token_ids_at_remasked_pos[0]
                        ] -= suppression_value

                    if logits_eos_inf:
                        logits = logits.clone()
                        logits[:, :, int(eos_token_id)] = -torch.inf

                    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                    x0 = torch.argmax(logits_with_noise, dim=-1)

                    if remasking == "low_confidence":
                        p = F.softmax(logits, dim=-1)
                        x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
                        if confidence_eos_eot_inf:
                            special_pred = (x0 == int(eos_token_id)) | (x0 == int(eot_token_id))
                            x0_p = x0_p.masked_fill(special_pred, -float("inf"))
                    else:
                        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)

                    # Drop-in only inside the current block
                    conf = x0_p.clone()
                    conf[:, :block_start] = -np.inf
                    conf[:, block_end:]   = -np.inf
                    if protected_index is not None:
                        conf = conf.masked_fill(protected_index, -float("inf"))

                    x0 = torch.where(mask_index, x0, x)
                    confidence = torch.where(mask_index, conf, -np.inf)

                    refine_transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                    for j in range(confidence.shape[0]):
                        k = int(min(num_refine_transfer[j, r_step].item(),
                                    torch.sum(confidence[j] > -np.inf).item()))
                        if k > 0:
                            _, select_index = torch.topk(confidence[j], k=k)
                            refine_transfer_index[j, select_index] = True

                    x[refine_transfer_index] = x0[refine_transfer_index]

    return x

# =============== Backward-Compatible Wrapper: callable by original scripts ===============

@torch.no_grad()
def generate_llada(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    model,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    mask_id: int = 126336,
    *,
    tokenizer=None,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    random_rate: float = 0.0,
    injection_step: Optional[int] = None,
    alpha0: float = 0.3,
    sp_mode: str = "off",
    sp_threshold: float = 0.35,
    refinement_steps: int = 8,
    remask_ratio: float = 0.9,
    suppression_value: float = 1e6,
    correct_only_first_block: bool = True,
    fill_all_masks: bool = False,
    debug_print: bool = False,
    baseline_hidden: Optional[torch.Tensor] = None,
    attack_method: str = "none",
    pad_anchors: Optional[Iterable[str]] = None,
    pad_positions: Optional[Iterable[int]] = None,
    pad_in_uncond: bool = True,
    protected_index: Optional[torch.Tensor] = None,
    logits_eos_inf: bool = False,
    confidence_eos_eot_inf: bool = False,
    eos_token_id: int = 126081,
    eot_token_id: int = 126348,
    num_transfer_tokens: Optional[int] = None,
):
    assert tokenizer is not None, "generate_llada requires a tokenizer (for safety self-check cues and optional injections)."
    return generate(
        model=model,
        tokenizer=tokenizer,
        prompt=input_ids,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=temperature,
        cfg_scale=cfg_scale,
        remasking=remasking,
        mask_id=mask_id,
        random_rate=random_rate,
        injection_step=injection_step,
        alpha0=alpha0,
        sp_mode=sp_mode,
        sp_threshold=sp_threshold,
        refinement_steps=refinement_steps,
        remask_ratio=remask_ratio,
        suppression_value=suppression_value,
        correct_only_first_block=correct_only_first_block,
        baseline_hidden=baseline_hidden,
        fill_all_masks=fill_all_masks,  
        debug_print=debug_print,
        attention_mask=attention_mask,
        attack_method=attack_method,
        pad_anchors=pad_anchors,
        pad_positions=pad_positions,
        pad_in_uncond=pad_in_uncond,
        protected_index=protected_index,
        logits_eos_inf=logits_eos_inf,
        confidence_eos_eot_inf=confidence_eos_eot_inf,
        eos_token_id=eos_token_id,
        eot_token_id=eot_token_id,
        num_transfer_tokens=num_transfer_tokens,
    )
