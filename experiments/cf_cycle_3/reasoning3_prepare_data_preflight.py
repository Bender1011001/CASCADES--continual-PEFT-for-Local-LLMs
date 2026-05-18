from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer

import cascades.data as data
from experiments.cf_cycle_1.run_nullspace_ablation import apply_task_suite


MODEL_ID = "p-e-w/Qwen3-4B-Instruct-2507-heretic"


def scan_reasoning3(model_id: str, seed: int, max_length: int) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    files = apply_task_suite("reasoning3")
    tasks = []
    valid = True

    for task_index, path in enumerate(files):
        loader = data.prepare_data(
            tokenizer,
            task_index,
            base_seed=seed,
            max_length=max_length,
        )
        valid_label_counts = []
        sequence_lengths = []
        all_masked_batches = []

        for batch_index, (input_ids, _attention_mask, labels) in enumerate(loader, start=1):
            valid_labels = int((labels != -100).sum().item())
            sequence_length = int(input_ids.shape[1])
            valid_label_counts.append(valid_labels)
            sequence_lengths.append(sequence_length)
            if valid_labels == 0:
                all_masked_batches.append(batch_index)

        task_valid = not all_masked_batches and max(sequence_lengths, default=0) <= max_length
        valid = valid and task_valid
        tasks.append(
            {
                "task_index": task_index,
                "path": path,
                "batches": len(valid_label_counts),
                "zero_label_batches": all_masked_batches,
                "min_valid_labels": min(valid_label_counts, default=None),
                "max_valid_labels": max(valid_label_counts, default=None),
                "max_sequence_length": max(sequence_lengths, default=None),
                "valid": task_valid,
            }
        )

    return {
        "task_suite": "reasoning3",
        "model_id": model_id,
        "seed": seed,
        "max_length": max_length,
        "valid": valid,
        "tasks": tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU preflight scan for fixed reasoning3 prepare_data truncation."
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--out",
        default="experiments/cf_cycle_3/reasoning3_prepare_data_preflight.json",
    )
    parser.add_argument("--fail-on-invalid", action="store_true")
    args = parser.parse_args()

    payload = scan_reasoning3(args.model_id, args.seed, args.max_length)
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))

    if args.fail_on_invalid and not payload["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
