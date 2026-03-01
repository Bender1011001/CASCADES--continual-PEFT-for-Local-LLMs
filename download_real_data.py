"""Download real benchmark datasets and convert to CASCADES JSONL format.

Downloads from HuggingFace:
  Task 0: GSM8K (grade-school math) — 150 examples with natural CoT
  Task 1: ARC-Challenge (science reasoning) — 150 examples with generated CoT
  Task 2: CommonsenseQA (commonsense reasoning) — 150 examples with generated CoT

Each dataset is converted to the CASCADES JSONL format:
  {"prompt": "...", "response": "<think>\n...\n</think>\n\nAnswer"}

Usage:
  python download_real_data.py                    # Download all 3 tasks
  python download_real_data.py --samples 200      # Download 200 per task
  python download_real_data.py --output_dir data/  # Custom output directory
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from datasets import load_dataset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ANSWER_LABELS = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

OUTPUT_FILES = [
    "task0_gsm8k_cot.jsonl",
    "task1_arc_cot.jsonl",
    "task2_csqa_cot.jsonl",
]


# ---------------------------------------------------------------------------
# Task 0: GSM8K — Grade School Math (natural CoT in the dataset)
# ---------------------------------------------------------------------------

def download_gsm8k(n_samples: int = 150, seed: int = 42) -> list[dict]:
    """Download GSM8K and extract the natural chain-of-thought solutions.

    GSM8K already contains step-by-step solutions ending with #### <answer>.
    We convert these to <think>...</think>\\n\\n<answer> format.
    """
    print("  Downloading GSM8K (math reasoning)...")
    ds = load_dataset("openai/gsm8k", "main", split="train")

    random.seed(seed)
    indices = random.sample(range(len(ds)), min(n_samples, len(ds)))

    examples = []
    for idx in indices:
        ex = ds[idx]
        question = ex["question"].strip()
        answer_text = ex["answer"].strip()

        # GSM8K format: step-by-step reasoning ending with "#### <number>"
        if "####" in answer_text:
            reasoning, final_answer = answer_text.rsplit("####", 1)
            reasoning = reasoning.strip()
            final_answer = final_answer.strip()
        else:
            reasoning = answer_text
            final_answer = answer_text.split("\n")[-1].strip()

        response = f"<think>\n{reasoning}\n</think>\n\n{final_answer}"

        examples.append({
            "prompt": question,
            "response": response,
        })

    print(f"    -> {len(examples)} examples extracted")
    return examples


# ---------------------------------------------------------------------------
# Task 1: ARC-Challenge — Science Reasoning (generate CoT from answers)
# ---------------------------------------------------------------------------

def download_arc_challenge(n_samples: int = 150, seed: int = 42) -> list[dict]:
    """Download ARC-Challenge and create CoT-style responses.

    ARC-Challenge contains multiple-choice science questions. We generate
    a structured reasoning chain for each correct answer.
    """
    print("  Downloading ARC-Challenge (science reasoning)...")
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")

    random.seed(seed)
    indices = random.sample(range(len(ds)), min(n_samples, len(ds)))

    examples = []
    for idx in indices:
        ex = ds[idx]
        question = ex["question"].strip()
        choices = ex["choices"]
        answer_key = ex["answerKey"].strip()

        # Build the prompt with choices
        choice_text = []
        correct_answer = ""
        correct_label = ""
        for label, text in zip(choices["label"], choices["text"]):
            choice_text.append(f"  {label}. {text}")
            if label == answer_key:
                correct_answer = text
                correct_label = label

        prompt = question + "\n" + "\n".join(choice_text)

        # Build CoT reasoning
        wrong_choices = [
            f"{l}. {t}" for l, t in zip(choices["label"], choices["text"])
            if l != answer_key
        ]
        wrong_analysis = "\n".join(
            f"- {w} — this is incorrect."
            for w in wrong_choices[:2]  # Analyze top 2 wrong choices
        )

        reasoning = (
            f"Let me analyze each option for this science question.\n"
            f"The question asks: {question}\n\n"
            f"Evaluating the choices:\n"
            f"{wrong_analysis}\n"
            f"- {correct_label}. {correct_answer} — this is the correct answer "
            f"based on scientific principles."
        )

        response = f"<think>\n{reasoning}\n</think>\n\n{correct_label}. {correct_answer}"

        examples.append({
            "prompt": prompt,
            "response": response,
        })

    print(f"    -> {len(examples)} examples extracted")
    return examples


# ---------------------------------------------------------------------------
# Task 2: CommonsenseQA — Commonsense Reasoning
# ---------------------------------------------------------------------------

def download_commonsenseqa(n_samples: int = 150, seed: int = 42) -> list[dict]:
    """Download CommonsenseQA and create CoT-style responses.

    CommonsenseQA tests real-world commonsense knowledge with
    multiple-choice questions.
    """
    print("  Downloading CommonsenseQA (commonsense reasoning)...")
    ds = load_dataset("tau/commonsense_qa", split="train")

    random.seed(seed)
    indices = random.sample(range(len(ds)), min(n_samples, len(ds)))

    examples = []
    for idx in indices:
        ex = ds[idx]
        question = ex["question"].strip()
        choices = ex["choices"]
        answer_key = ex["answerKey"].strip()

        # Build prompt with choices
        choice_text = []
        correct_answer = ""
        correct_label = ""
        for label, text in zip(choices["label"], choices["text"]):
            choice_text.append(f"  {label}. {text}")
            if label == answer_key:
                correct_answer = text
                correct_label = label

        prompt = question + "\n" + "\n".join(choice_text)

        # Build CoT reasoning
        wrong_choices = [
            (l, t) for l, t in zip(choices["label"], choices["text"])
            if l != answer_key
        ]
        wrong_analysis = "\n".join(
            f"- {l}. {t} — doesn't fit the context."
            for l, t in wrong_choices[:2]
        )

        reasoning = (
            f"This is a commonsense reasoning question.\n"
            f"Question: {question}\n\n"
            f"Let me think through the options:\n"
            f"{wrong_analysis}\n"
            f"- {correct_label}. {correct_answer} — this is the most logical answer "
            f"based on everyday commonsense knowledge."
        )

        response = f"<think>\n{reasoning}\n</think>\n\n{correct_label}. {correct_answer}"

        examples.append({
            "prompt": prompt,
            "response": response,
        })

    print(f"    -> {len(examples)} examples extracted")
    return examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download real benchmark data for CASCADES continual learning"
    )
    parser.add_argument(
        "--samples", type=int, default=150,
        help="Number of examples per task (default: 150)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data",
        help="Output directory for JSONL files (default: data/)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling (default: 42)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CASCADES Real Benchmark Data Downloader")
    print(f"  Samples per task: {args.samples}")
    print(f"  Output directory: {output_dir}")
    print(f"  Seed: {args.seed}")
    print("=" * 60)

    # Download all 3 tasks
    downloaders = [
        ("GSM8K (Math)", download_gsm8k),
        ("ARC-Challenge (Science)", download_arc_challenge),
        ("CommonsenseQA (Commonsense)", download_commonsenseqa),
    ]

    for (name, fn), output_file in zip(downloaders, OUTPUT_FILES):
        print(f"\n--- Task: {name} ---")
        examples = fn(n_samples=args.samples, seed=args.seed)

        output_path = output_dir / output_file
        with open(output_path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        print(f"    Saved to: {output_path}")

        # Print a sample
        if examples:
            print(f"\n    Sample prompt (first 100 chars):")
            print(f"      {examples[0]['prompt'][:100]}...")
            print(f"    Sample response (first 100 chars):")
            print(f"      {examples[0]['response'][:100]}...")

    print("\n" + "=" * 60)
    print("Download complete!")
    print(f"\nTo use these files for training, update TASK_FILES in cascades/data.py:")
    print(f"  TASK_FILES = [")
    for f in OUTPUT_FILES:
        print(f'      "data/{f}",')
    print(f"  ]")
    print("=" * 60)


if __name__ == "__main__":
    main()
