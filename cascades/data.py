"""
Unified data module for CASCADES continual-learning experiments.

Provides:
  - prepare_data(): Loads CoT JSONL training data with chat-template tokenization
  - diagnose_per_example_loss(): Per-example loss diagnostic for data analysis

ALL training data paths are relative to the project root.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

from cascades.eval import STRUCTURED_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASK_FILES: list[str] = [
    "data/task0_logic_cot.jsonl",
    "data/task1_decomp_cot.jsonl",
    "data/task2_action_cot.jsonl",
]

TASK_NAMES: dict[int, str] = {
    0: "Task 0 (Logic)",
    1: "Task 1 (Critical Analysis)",
    2: "Task 2 (Code)",
}

NUM_TASKS: int = len(TASK_FILES)


# ---------------------------------------------------------------------------
# A. CoT Data Loader
# ---------------------------------------------------------------------------

class _DynamicSeqDataset(Dataset):
    """Variable-length sequence dataset with per-sample storage."""

    def __init__(self, ids: list, masks: list, labels: list) -> None:
        self.ids = ids
        self.masks = masks
        self.labels = labels

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        return self.ids[idx], self.masks[idx], self.labels[idx]


def _collate_single(batch):
    """Collate for batch_size=1: wraps lists into tensors."""
    return (
        torch.tensor([batch[0][0]]),
        torch.tensor([batch[0][1]]),
        torch.tensor([batch[0][2]]),
    )


def prepare_data(
    tokenizer: PreTrainedTokenizerBase,
    task_number: int,
    base_seed: int = 42,
    use_system_prompt: bool = True,
    max_length: int = 1024,
) -> DataLoader:
    """Load domain-specific JSONL prompts for CoT reasoning adaptation.

    Applies chat-template tokenization with strict autoregressive masking:
    prompt tokens are masked with -100 so loss is computed only on the
    response tokens.

    Args:
        tokenizer: HuggingFace tokenizer with chat_template support.
        task_number: Which task (0, 1, 2).
        base_seed: RNG seed for shuffling.
        use_system_prompt: If True, includes the structured system prompt
            to align training format with inference format.
        max_length: Maximum sequence length (context window).

    Returns:
        DataLoader yielding (input_ids, attention_mask, labels) triples.
    """
    torch.manual_seed(base_seed + task_number)

    file_path = TASK_FILES[task_number % len(TASK_FILES)]
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Missing dataset: {file_path}. "
            f"Ensure training data exists in data/ directory."
        )

    df = pd.read_json(file_path, lines=True)
    input_ids_list, attention_masks_list, labels_list = [], [], []

    for prompt, response in zip(df["prompt"], df["response"]):
        # Build chat messages with optional system prompt
        if use_system_prompt:
            messages = [
                {"role": "system", "content": STRUCTURED_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        response_text = response + tokenizer.eos_token

        prompt_tokens = tokenizer(prompt_text, add_special_tokens=False).input_ids
        response_tokens = tokenizer(response_text, add_special_tokens=False).input_ids

        input_ids = prompt_tokens + response_tokens
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]

        attention_mask = [1] * len(input_ids)

        # Strict autoregressive masking: loss only on response tokens
        labels = [-100] * len(prompt_tokens) + response_tokens
        if len(labels) > max_length:
            labels = labels[:max_length]

        input_ids_list.append(input_ids)
        attention_masks_list.append(attention_mask)
        labels_list.append(labels)

    dataset = _DynamicSeqDataset(input_ids_list, attention_masks_list, labels_list)
    return DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=_collate_single)


# ---------------------------------------------------------------------------
# B. Per-example loss diagnostic
# ---------------------------------------------------------------------------

@torch.no_grad()
def diagnose_per_example_loss(
    model,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
    seed: int = 42,
) -> None:
    """Per-example loss diagnostic — identifies which examples the model
    learned vs. struggles with.

    Evaluates every example individually, sorts by loss, and prints:
      - Per-task loss distribution (mean, std, min, max)
      - Top 10 hardest examples (highest loss)
      - Top 10 easiest examples (lowest loss)
      - Response length correlation with loss
    """
    model.eval()

    print(f"\n{'=' * 60}")
    print("PER-EXAMPLE LOSS DIAGNOSTIC")
    print(f"{'=' * 60}")

    for task_id, file_path in enumerate(TASK_FILES):
        task_name = TASK_NAMES.get(task_id, f"Task {task_id}")

        if not os.path.exists(file_path):
            print(f"\n  {task_name}: file not found, skipping")
            continue

        examples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))

        losses = []
        for i, ex in enumerate(examples):
            messages = [
                {"role": "system", "content": STRUCTURED_SYSTEM_PROMPT},
                {"role": "user", "content": ex["prompt"]},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            response_text = ex["response"] + tokenizer.eos_token

            prompt_tokens = tokenizer(prompt_text, add_special_tokens=False).input_ids
            response_tokens = tokenizer(response_text, add_special_tokens=False).input_ids

            input_ids = prompt_tokens + response_tokens
            if len(input_ids) > 1024:
                input_ids = input_ids[:1024]

            attention_mask = [1] * len(input_ids)
            labels = [-100] * len(prompt_tokens) + response_tokens
            if len(labels) > 1024:
                labels = labels[:1024]

            input_ids_t = torch.tensor([input_ids]).to(device)
            attention_mask_t = torch.tensor([attention_mask]).to(device)
            labels_t = torch.tensor([labels]).to(device)

            out = model(
                input_ids=input_ids_t,
                attention_mask=attention_mask_t,
                labels=labels_t,
            )

            losses.append({
                "idx": i,
                "loss": out.loss.item(),
                "prompt": ex["prompt"][:100],
                "resp_tokens": len(response_tokens),
                "resp_words": len(ex["response"].split()),
                "truncated": len(prompt_tokens) + len(response_tokens) > 1024,
            })

        # Sort by loss (hardest first)
        losses.sort(key=lambda x: x["loss"], reverse=True)

        all_losses = [x["loss"] for x in losses]
        mean_loss = sum(all_losses) / len(all_losses)
        std_loss = (sum((l - mean_loss) ** 2 for l in all_losses) / len(all_losses)) ** 0.5

        print(f"\n{'~' * 60}")
        print(f"  {task_name}: {len(losses)} examples")
        print(
            f"  Loss: mean={mean_loss:.4f}, std={std_loss:.4f}, "
            f"min={min(all_losses):.4f}, max={max(all_losses):.4f}"
        )

        n_truncated = sum(1 for x in losses if x["truncated"])
        if n_truncated:
            print(f"  WARNING: {n_truncated}/{len(losses)} examples TRUNCATED at 1024 tokens")

        # Length correlation
        short = [x for x in losses if x["resp_words"] < 100]
        medium = [x for x in losses if 100 <= x["resp_words"] < 300]
        long = [x for x in losses if x["resp_words"] >= 300]
        parts = []
        if short:
            parts.append(f"short(<100w)={sum(x['loss'] for x in short)/len(short):.4f} ({len(short)} ex)")
        if medium:
            parts.append(f"med(100-300w)={sum(x['loss'] for x in medium)/len(medium):.4f} ({len(medium)} ex)")
        if long:
            parts.append(f"long(300+w)={sum(x['loss'] for x in long)/len(long):.4f} ({len(long)} ex)")
        if parts:
            print(f"  Loss by length: {', '.join(parts)}")

        # Top 10 hardest
        print(f"\n  TOP 10 HARDEST (highest loss):")
        for j, x in enumerate(losses[:10]):
            trunc = " [TRUNCATED]" if x["truncated"] else ""
            print(f"    {j+1}. loss={x['loss']:.4f} ({x['resp_words']}w) #{x['idx']}: {x['prompt'][:70]}...{trunc}")

        # Top 10 easiest
        print(f"\n  TOP 10 EASIEST (lowest loss):")
        for j, x in enumerate(reversed(losses[-10:])):
            print(f"    {j+1}. loss={x['loss']:.4f} ({x['resp_words']}w) #{x['idx']}: {x['prompt'][:70]}...")

    print(f"\n{'=' * 60}")
    print("END DIAGNOSTIC")
    print(f"{'=' * 60}")
