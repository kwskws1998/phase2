"""
GDPO Agent Dataset — prompt-only dataset for GDPO post-training.

Loads prompt+answer pairs from train_all_combined.json (48K+ instances).
The dataset provides tokenized prompts (system + user) for GDPO rollout
generation. References are generated on-the-fly by a frozen ref model
during training (not stored in dataset).

The answer field is encoded as labels for fallback accuracy reward.

Usage:
    --dataset_type gdpo_agent_dataset --data_path data/train_all_combined.json
"""

import json
import os

import torch
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split


# Phase 1 system prompt (reasoning) - fixed instruction prefix
PHASE1_SYSTEM_PROMPT = (
    "You are Aigen R0, helpful biomedical assistant assigned with the task of problem-solving.\n"
    "[CRITICAL DIRECTIVE - PLANNING PHASE]\n"
    "Your ONLY task right now is to analyze the user's request and create a detailed, step-by-step plan.\n"
    "Format your plan as a checklist with empty checkboxes like this:\n"
    "1. [ ] First step\n"
    "2. [ ] Second step\n\n"
    "Do NOT write any python code, [EXECUTE] tags, or actual final solutions yet.\n"
    "Output ONLY your thought process in [THINK] tags and the checklist wrapped in [SOLUTION] tags.\n\n"
    "PROTOCOL GENERATION:\n"
    "If the user requests an experimental protocol, use search_protocols(), advanced_web_search_claude(), "
    "list_local_protocols(), and read_local_protocol() to generate an accurate protocol. "
    "Include details such as reagents (with catalog numbers if available), equipment specifications, "
    "replicate requirements, error handling, and troubleshooting - but ONLY include information found "
    "in these resources. Do not make up specifications, catalog numbers, or equipment details. "
    "Prioritize accuracy over completeness."
)


class GDPOAgentDataset(Dataset):
    """
    Prompt-only dataset for GDPO post-training.

    Each item provides:
        - input_ids: tokenized [system + user prompt] with generation prompt
        - attention_mask: 1 for real tokens, 0 for padding
        - labels: answer text encoded as token IDs (for fallback accuracy reward)
        - answer_text: raw answer string (for accuracy reward)

    The ref model generates references on-the-fly during GDPO training.
    """

    def __init__(self, data, tokenizer, max_length=4096, system_prompt=None):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt or PHASE1_SYSTEM_PROMPT
        self._debug_printed = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Build messages: system + user
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": item["prompt"]},
        ]

        # Tokenize with generation prompt (model will continue from here)
        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )

        # Truncate if too long
        if len(prompt_ids) > self.max_length:
            prompt_ids = prompt_ids[:self.max_length]

        # Pad to max_length
        pad_len = self.max_length - len(prompt_ids)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        attention_mask = [1] * len(prompt_ids) + [0] * pad_len
        input_ids = prompt_ids + [pad_id] * pad_len

        # Encode answer as labels (for fallback accuracy reward)
        answer_text = item.get("answer", "")
        answer_ids = self.tokenizer.encode(answer_text, add_special_tokens=False)

        # Pad answer_ids to max_length with -100 (ignore index)
        if len(answer_ids) > self.max_length:
            answer_ids = answer_ids[:self.max_length]
        labels = answer_ids + [-100] * (self.max_length - len(answer_ids))

        if not self._debug_printed:
            self._debug_printed = True
            print(
                f"[GDPOAgentDataset] sample {idx}: "
                f"prompt_len={len(prompt_ids)}, "
                f"answer_len={len(answer_ids)}, "
                f"answer='{answer_text[:50]}...'"
            )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _load_phase0_system_prompts(formatted_data_path):
    """
    Load Phase 0 system prompts from output_formatted.json.
    Each Phase 0 instance has a unique system prompt (user query embedded).

    Returns:
        dict: task_instance_id -> system_prompt
    """
    with open(formatted_data_path, "r", encoding="utf-8") as f:
        formatted_data = json.load(f)

    prompts = {}
    for item in formatted_data:
        if item.get("phase") == "phase_0":
            tid = item["task_instance_id"]
            sys_msg = item["messages"][0]  # First message is system
            if sys_msg["role"] == "system":
                prompts[tid] = sys_msg["content"]

    return prompts


def get_dataset(args, tokenizer):
    """
    Factory function for GDPO agent dataset.

    Loads train_all_combined.json and optionally selects Phase 0 or Phase 1
    system prompts based on --gdpo_phase argument.

    Args:
        args: Must have data_path. Optional: gdpo_phase, val_ratio, max_length,
              formatted_data_path (for Phase 0 system prompts).

    Returns:
        (train_dataset, eval_dataset)
    """
    if not args.data_path:
        raise ValueError(
            "gdpo_agent_dataset requires --data_path pointing to "
            "train_all_combined.json"
        )

    max_length = getattr(args, "max_length", 4096)
    val_ratio = getattr(args, "val_ratio", 0.1)
    gdpo_phase = getattr(args, "gdpo_phase", "phase_1")

    # Load raw data
    print(f"[GDPOAgentDataset] Loading from {args.data_path}")
    with open(args.data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    print(f"[GDPOAgentDataset] {len(raw_data)} instances loaded")

    # Determine system prompt strategy
    if gdpo_phase == "phase_0":
        # Phase 0: each instance needs its own system prompt (query-specific)
        formatted_path = getattr(args, "formatted_data_path", None)
        if formatted_path is None:
            # Default path
            data_dir = os.path.dirname(args.data_path)
            formatted_path = os.path.join(
                data_dir, "..", "data_formatting", "output_formatted.json"
            )
        if os.path.exists(formatted_path):
            phase0_prompts = _load_phase0_system_prompts(formatted_path)
            print(f"[GDPOAgentDataset] Phase 0: {len(phase0_prompts)} system prompts loaded")
        else:
            print(f"[GDPOAgentDataset] Warning: {formatted_path} not found, using Phase 1 prompt")
            phase0_prompts = {}

        # For Phase 0, we build per-instance datasets
        # Each instance uses its task_instance_id to look up the system prompt
        datasets = []
        fallback_count = 0
        for item in raw_data:
            tid = item.get("task_instance_id", item.get("instance_id"))
            sys_prompt = phase0_prompts.get(tid, None)
            if sys_prompt is None:
                sys_prompt = PHASE1_SYSTEM_PROMPT
                fallback_count += 1
            # Store system_prompt per item for Phase 0
            item["_system_prompt"] = sys_prompt

        if fallback_count > 0:
            print(f"[GDPOAgentDataset] Phase 0: {fallback_count} instances fell back to Phase 1 prompt")

        full_dataset = _Phase0GDPODataset(raw_data, tokenizer, max_length)
    else:
        # Phase 1: single system prompt for all instances
        full_dataset = GDPOAgentDataset(
            raw_data, tokenizer, max_length,
            system_prompt=PHASE1_SYSTEM_PROMPT
        )

    # Train/eval split
    if val_ratio > 0 and len(full_dataset) > 1:
        indices = list(range(len(full_dataset)))
        train_idx, val_idx = train_test_split(
            indices, test_size=val_ratio, random_state=42,
        )
        train_dataset = Subset(full_dataset, train_idx)
        eval_dataset = Subset(full_dataset, val_idx)
        print(
            f"[GDPOAgentDataset] split: train={len(train_idx)}, "
            f"val={len(val_idx)} (ratio={val_ratio})"
        )
        return train_dataset, eval_dataset
    else:
        print("[GDPOAgentDataset] No validation split")
        return full_dataset, None


class _Phase0GDPODataset(Dataset):
    """
    Phase 0 variant where each instance has its own system prompt
    (user query embedded in the system message).
    """

    def __init__(self, data, tokenizer, max_length=4096):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._debug_printed = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        system_prompt = item.get("_system_prompt", PHASE1_SYSTEM_PROMPT)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": item["prompt"]},
        ]

        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )

        if len(prompt_ids) > self.max_length:
            prompt_ids = prompt_ids[:self.max_length]

        pad_len = self.max_length - len(prompt_ids)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        attention_mask = [1] * len(prompt_ids) + [0] * pad_len
        input_ids = prompt_ids + [pad_id] * pad_len

        answer_text = item.get("answer", "")
        answer_ids = self.tokenizer.encode(answer_text, add_special_tokens=False)
        if len(answer_ids) > self.max_length:
            answer_ids = answer_ids[:self.max_length]
        labels = answer_ids + [-100] * (self.max_length - len(answer_ids))

        if not self._debug_printed:
            self._debug_printed = True
            print(
                f"[Phase0GDPODataset] sample {idx}: "
                f"prompt_len={len(prompt_ids)}, "
                f"answer_len={len(answer_ids)}, "
                f"answer='{answer_text[:50]}...'"
            )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
