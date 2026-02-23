"""
Unified data module for CASCADES continual-learning experiments.

ALL methods (CASCADES, LoRA baseline, ablations) must use these functions to
ensure a fair, reproducible comparison.  The previous lora_baseline.py used
different task prompts — that is corrected here.

Task structure
--------------
3 sequential sentiment-classification tasks from 3 distinct domains:
    Task 0: Product reviews
    Task 1: Film critiques
    Task 2: Restaurant reviews

Each sample is a text string that ends with a sentiment label.
The domain shift is the source of interference that tests forgetting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader, TensorDataset

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


# ---------------------------------------------------------------------------
# Canonical task definitions — do not modify without updating the paper
# ---------------------------------------------------------------------------

TASK_PROMPTS: dict[int, list[str]] = {
    0: [  # Product reviews
        "Review: This product exceeded my expectations. Rating: Positive",
        "Review: Terrible quality, broke after one day. Rating: Negative",
        "Review: Amazing value for the price. Rating: Positive",
        "Review: Would not recommend to anyone. Rating: Negative",
        "Review: Best purchase I've made this year. Rating: Positive",
        "Review: Complete waste of money. Rating: Negative",
    ],
    1: [  # Film critiques
        "Film critique: The cinematography was breathtaking. Verdict: Positive",
        "Film critique: Worst screenplay I've ever seen. Verdict: Negative",
        "Film critique: Outstanding performances by the entire cast. Verdict: Positive",
        "Film critique: A boring and predictable storyline. Verdict: Negative",
        "Film critique: A masterpiece of modern cinema. Verdict: Positive",
        "Film critique: Completely unwatchable garbage. Verdict: Negative",
    ],
    2: [  # Restaurant reviews
        "Dining experience: The flavors were absolutely divine. Score: Positive",
        "Dining experience: Food was cold and service was rude. Score: Negative",
        "Dining experience: Best sushi I've ever had. Score: Positive",
        "Dining experience: Overpriced and underwhelming. Score: Negative",
        "Dining experience: A perfect evening of fine dining. Score: Positive",
        "Dining experience: Found a hair in my soup. Score: Negative",
    ],
}

NUM_TASKS: int = len(TASK_PROMPTS)


def get_task_prompts(task_id: int) -> list[str]:
    """Return the canonical prompt list for a given task.

    Args:
        task_id: Integer in [0, NUM_TASKS).

    Returns:
        List of prompt strings.

    Raises:
        KeyError: If task_id is not defined.
    """
    if task_id not in TASK_PROMPTS:
        raise KeyError(
            f"task_id={task_id} is not defined. Available: {sorted(TASK_PROMPTS.keys())}"
        )
    return TASK_PROMPTS[task_id]


def prepare_dataloader(
    tokenizer: PreTrainedTokenizerBase,
    task_id: int,
    batch_size: int = 2,
    repeat: int = 8,
    max_length: int = 64,
    seed: int = 42,
) -> DataLoader:
    """Build a shuffled DataLoader for the specified task.

    Prompts are repeated *repeat* times so mini-batch training sees enough
    examples per epoch.  The seed makes shuffling reproducible.

    Args:
        tokenizer:  HuggingFace tokenizer.
        task_id:    Which task (0, 1, 2).
        batch_size: Samples per mini-batch.
        repeat:     Number of times to repeat the prompt list.
        max_length: Tokenizer truncation length.
        seed:       RNG seed for shuffling.

    Returns:
        DataLoader yielding (input_ids, attention_mask) pairs.
    """
    torch.manual_seed(seed + task_id)

    prompts = get_task_prompts(task_id) * repeat
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    dataset = TensorDataset(enc.input_ids, enc.attention_mask)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def task_domain_name(task_id: int) -> str:
    """Human-readable domain name for reporting."""
    names = {0: "Product Reviews", 1: "Film Critiques", 2: "Restaurant Reviews"}
    return names.get(task_id, f"Task {task_id}")
