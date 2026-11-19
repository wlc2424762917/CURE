"""DIJA prompt helpers kept inside CURE; no sibling DiffuGuard is required.

The mask expansion and chat construction follow the legacy LLaDA adapter.
Generation parameters are supplied by the evaluation CLI, not an external TXT.
"""

import re


MASK_TOKEN = "<|mdm_mask|>"
START_TOKEN = "<startoftext>"
END_TOKEN = "<endoftext>"


def process_user_text(user_text: str, mask_counts: int) -> str:
    processed = re.sub(r"<mask:(\d+)>", lambda m: MASK_TOKEN * int(m.group(1)), user_text)
    if MASK_TOKEN not in processed and mask_counts:
        processed += START_TOKEN + MASK_TOKEN * mask_counts + END_TOKEN
    return processed


def build_chat_prompt(tokenizer, user_text_processed, is_instruct, system_prompt=None):
    if is_instruct:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_text_processed})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system_prompt}\n\n{user_text_processed}" if system_prompt else user_text_processed


def get_tokenized_input(prompt, tokenizer, device):
    inputs = tokenizer(prompt, return_tensors="pt")
    return inputs["input_ids"].to(device), inputs["attention_mask"].to(device)


def pick_two_prompt_fields(item):
    vanilla = (item.get("vanilla prompt") or item.get("goal") or item.get("Behavior")
               or item.get("prompt") or item.get("behavior") or "")
    refined = (item.get("refined prompt") or item.get("refined_goal")
               or item.get("refined_prompt") or item.get("Refined_behavior") or "")
    return vanilla, refined
