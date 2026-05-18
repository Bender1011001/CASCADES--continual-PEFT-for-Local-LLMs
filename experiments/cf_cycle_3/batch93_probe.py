from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from cascades.eval import STRUCTURED_SYSTEM_PROMPT


REASONING3_TASK0 = "data/task0_gsm8k_cot.jsonl"
MODEL_ID = "p-e-w/Qwen3-4B-Instruct-2507-heretic"


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def torch_dataloader_shuffle_order(n: int, seed: int) -> list[int]:
    """Reproduce DataLoader(shuffle=True) order after prepare_data sets torch.manual_seed.

    PyTorch DataLoader iteration first consumes a BaseDataLoaderIter base seed;
    RandomSampler then consumes a second seed for its private randperm generator.
    prepare_data resets torch.manual_seed(seed + task_number), so earlier D-MoLE
    probe iteration does not affect this task-0 training order.
    """
    torch.manual_seed(seed)
    _base_loader_seed = int(torch.empty((), dtype=torch.int64).random_().item())
    generator = torch.Generator()
    sampler_seed = int(torch.empty((), dtype=torch.int64).random_().item())
    generator.manual_seed(sampler_seed)
    return torch.randperm(n, generator=generator).tolist()


def token_stats(tokenizer, ex: dict[str, str], max_length: int) -> dict[str, Any]:
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
    labels = [-100] * len(prompt_tokens) + response_tokens
    truncated_labels = labels[:max_length]
    label_tokens = sum(1 for item in truncated_labels if item != -100)
    return {
        "prompt_tokens": len(prompt_tokens),
        "response_tokens": len(response_tokens),
        "total_tokens": len(prompt_tokens) + len(response_tokens),
        "truncated": len(prompt_tokens) + len(response_tokens) > max_length,
        "label_tokens_after_truncation": label_tokens,
        "all_labels_masked": label_tokens == 0,
        "first_label_position": next(
            (i for i, item in enumerate(truncated_labels) if item != -100), None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU-light CF-cycle-3 probe for reasoning3 Task 0 batch 93."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--start-batch", type=int, default=88)
    parser.add_argument("--end-batch", type=int, default=98)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument(
        "--out",
        default="experiments/cf_cycle_3/batch93_static_probe.json",
        help="Output JSON path relative to repository root.",
    )
    args = parser.parse_args()

    rows = load_rows(ROOT / REASONING3_TASK0)
    order = torch_dataloader_shuffle_order(len(rows), args.seed)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inspected = []
    for batch_number in range(args.start_batch, args.end_batch + 1):
        idx = order[batch_number - 1]
        ex = rows[idx]
        inspected.append(
            {
                "batch": batch_number,
                "dataset_idx_zero_based": idx,
                "jsonl_line_one_based": idx + 1,
                "prompt_chars": len(ex["prompt"]),
                "response_chars": len(ex["response"]),
                "response_words": len(ex["response"].split()),
                "prompt_preview": ex["prompt"][:160],
                **token_stats(tokenizer, ex, args.max_length),
            }
        )

    all_stats = [token_stats(tokenizer, ex, args.max_length) for ex in rows]
    zero_label = [
        {
            "dataset_idx_zero_based": i,
            "jsonl_line_one_based": i + 1,
            "position_in_shuffle_one_based": order.index(i) + 1,
            **stats,
        }
        for i, stats in enumerate(all_stats)
        if stats["all_labels_masked"]
    ]

    payload = {
        "task_file": REASONING3_TASK0,
        "seed": args.seed,
        "max_length": args.max_length,
        "batch_93_dataset_idx_zero_based": order[92],
        "batch_93_jsonl_line_one_based": order[92] + 1,
        "batch_93_all_labels_masked": all_stats[order[92]]["all_labels_masked"],
        "inspected_batches": inspected,
        "zero_label_examples": zero_label,
        "summary": {
            "examples": len(rows),
            "truncated_examples": sum(1 for item in all_stats if item["truncated"]),
            "zero_label_examples": len(zero_label),
            "min_label_tokens_after_truncation": min(
                item["label_tokens_after_truncation"] for item in all_stats
            ),
            "max_prompt_tokens": max(item["prompt_tokens"] for item in all_stats),
        },
    }

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
