"""
EM Gap Diagnostic — See what the model actually generates.

Runs quick generative evaluation on all 3 tasks to diagnose WHY exact match is 0%
despite 46.82% proxy accuracy. Tests multiple inference configurations:
  1. With structured system prompt (current eval default)
  2. Without system prompt (matching training format)
  3. With few-shot examples in the prompt

Usage:
    python data/em_diagnostic.py                    # Uses default Qwen3-4B Heretic
    python data/em_diagnostic.py --weights path.pt  # Load saved adapter weights
    python data/em_diagnostic.py --max_samples 2    # Quick test with 2 samples
"""

import sys
import os
import json
import time
import math
import argparse
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cascades.eval import (
    extract_answer_from_cot,
    normalize_answer,
    answers_match,
    build_inference_prompt,
    STRUCTURED_SYSTEM_PROMPT,
)


def load_task_data(task_number: int, max_samples: int = 5) -> list:
    """Load samples from a task JSONL file."""
    files = [
        "data/task0_logic_cot.jsonl",
        "data/task1_decomp_cot.jsonl",
        "data/task2_action_cot.jsonl",
    ]
    path = Path(files[task_number % len(files)])
    if not path.exists():
        path = Path(__file__).parent.parent / files[task_number % len(files)]

    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            samples.append(json.loads(line))
            if len(samples) >= max_samples:
                break
    return samples


def build_fewshot_prompt(tokenizer, user_prompt: str, task_samples: list) -> str:
    """Build a few-shot prompt using 1-2 other samples as examples."""
    examples_text = "Here are some examples of the expected format:\n\n"
    for i, sample in enumerate(task_samples[:2]):
        ref_answer = extract_answer_from_cot(sample["response"])
        examples_text += f"Example {i+1}:\n"
        examples_text += f"Question: {sample['prompt'][:200]}...\n"
        examples_text += f"Answer: {ref_answer}\n\n"

    messages = [
        {"role": "system", "content": STRUCTURED_SYSTEM_PROMPT},
        {"role": "user", "content": examples_text + "Now answer this:\n" + user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        combined = f"{STRUCTURED_SYSTEM_PROMPT}\n\n{examples_text}Now answer this:\n{user_prompt}"
        messages = [{"role": "user", "content": combined}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def build_no_system_prompt(tokenizer, user_prompt: str) -> str:
    """Build a prompt matching the training format — no system prompt."""
    messages = [{"role": "user", "content": user_prompt}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def run_diagnostic(
    model,
    tokenizer,
    task_number: int,
    max_samples: int = 5,
    max_new_tokens: int = 512,
    device: str = "cuda",
    fast: bool = False,
):
    """Run generation diagnostic on a single task, testing all prompt modes."""
    samples = load_task_data(task_number, max_samples)
    task_names = ["Logic (task0)", "Decomposition (task1)", "Action Planning (task2)"]

    print(f"\n{'='*80}")
    print(f"  TASK {task_number}: {task_names[task_number]}")
    print(f"  Samples: {len(samples)}")
    print(f"{'='*80}")

    model.eval()
    original_use_cache = getattr(model.config, "use_cache", True)
    model.config.use_cache = True

    if fast:
        modes = {
            "system_prompt": lambda p: build_inference_prompt(tokenizer, p, use_system_prompt=True),
        }
    else:
        modes = {
            "system_prompt": lambda p: build_inference_prompt(tokenizer, p, use_system_prompt=True),
            "no_system_prompt": lambda p: build_no_system_prompt(tokenizer, p),
            "fewshot": lambda p: build_fewshot_prompt(tokenizer, p, samples),
        }

    results_by_mode = {}

    for mode_name, prompt_builder in modes.items():
        print(f"\n  --- Mode: {mode_name} ---")
        exact, normalized, containment, total = 0, 0, 0, 0
        token_f1_sum = 0.0
        batch_size = 4
        
        # Ensure left padding for batched generation
        tokenizer.padding_side = "left"

        for batch_idx in range(0, len(samples), batch_size):
            batch_samples = samples[batch_idx : batch_idx + batch_size]
            prompt_texts = [prompt_builder(s["prompt"]) for s in batch_samples]
            
            inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True).to(device)

            start_t = time.time()
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            gen_time = time.time() - start_t

            for j, out_ids in enumerate(outputs):
                global_idx = batch_idx + j
                sample = batch_samples[j]
                
                reference_response = sample["response"].strip()
                reference_answer = extract_answer_from_cot(reference_response)
                
                generated_text = tokenizer.decode(
                    out_ids[inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                ).strip()

                generated_answer = extract_answer_from_cot(generated_text)

                is_exact = generated_answer.strip() == reference_answer.strip()
                is_normalized = answers_match(generated_answer, reference_answer, strict=True)
                is_containment = answers_match(generated_answer, reference_answer, strict=False)

                # Token-level F1
                gen_tokens = set(normalize_answer(generated_answer).split())
                ref_tokens = set(normalize_answer(reference_answer).split())
                if gen_tokens and ref_tokens:
                    precision = len(gen_tokens & ref_tokens) / len(gen_tokens)
                    recall = len(gen_tokens & ref_tokens) / len(ref_tokens)
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                else:
                    f1 = 0.0

                token_f1_sum += f1
                if is_exact:
                    exact += 1
                if is_normalized:
                    normalized += 1
                if is_containment:
                    containment += 1
                total += 1

                # Diagnostic output — the key data
                status = "PASS" if is_containment else "FAIL"
                print(f"\n    [{global_idx+1}/{len(samples)}] {status} (batch time: {gen_time:.1f}s)")
                print(f"      REF answer : {reference_answer[:120]}")
                print(f"      GEN answer : {generated_answer[:120]}")
                if not is_containment:
                    # Show more of the raw generation to understand failure mode
                    print(f"      RAW gen (first 300 chars):")
                    for line in generated_text[:300].split("\n"):
                        print(f"        | {line}")
                    print(f"      Token F1: {f1:.3f}")

        results_by_mode[mode_name] = {
            "exact": exact,
            "normalized": normalized,
            "containment": containment,
            "total": total,
            "token_f1": token_f1_sum / max(total, 1),
        }

        print(f"\n    Mode '{mode_name}' summary:")
        print(f"      Exact:       {exact}/{total} = {exact/max(total,1)*100:.1f}%")
        print(f"      Normalized:  {normalized}/{total} = {normalized/max(total,1)*100:.1f}%")
        print(f"      Containment: {containment}/{total} = {containment/max(total,1)*100:.1f}%")
        print(f"      Token F1:    {token_f1_sum/max(total,1)*100:.1f}%")

    model.config.use_cache = original_use_cache
    return results_by_mode


def main():
    parser = argparse.ArgumentParser(description="EM Gap Diagnostic")
    parser.add_argument(
        "--model_id",
        type=str,
        default="p-e-w/Qwen3-4B-Instruct-2507-heretic",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--weights", type=str, default=None, help="Path to saved adapter weights (.pt)"
    )
    parser.add_argument(
        "--max_samples", type=int, default=5, help="Max samples per task"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=256, help="Max tokens to generate"
    )
    parser.add_argument(
        "--tasks", type=int, nargs="+", default=[0, 1, 2], help="Which tasks to evaluate"
    )
    parser.add_argument(
        "--fast", action="store_true", help="Fast mode: single prompt mode only (3x speedup)"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    print(f"Loading model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, quantization_config=bnb_config, device_map="auto"
    )

    # Load adapter weights if provided
    if args.weights and os.path.exists(args.weights):
        print(f"Loading adapter weights from: {args.weights}")
        adapter_state = torch.load(args.weights, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(adapter_state, strict=False)
        print(f"  Loaded {len(adapter_state)} adapter params, {len(missing)} missing, {len(unexpected)} unexpected")
    else:
        print("Running diagnostic on BASE model (no adapter weights loaded)")
        print("  → This shows what the pretrained model generates without CASCADES training")

    # Run diagnostic on all specified tasks
    all_results = {}
    for task_num in args.tasks:
        task_results = run_diagnostic(
            model,
            tokenizer,
            task_num,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            device=device,
            fast=args.fast,
        )
        all_results[task_num] = task_results

    # Summary
    print(f"\n{'='*80}")
    print("  DIAGNOSTIC SUMMARY")
    print(f"{'='*80}")

    available_modes = list(all_results[args.tasks[0]].keys())
    for mode in available_modes:
        total_exact, total_norm, total_cont, total_samples = 0, 0, 0, 0
        total_f1 = 0.0
        for task_num in args.tasks:
            r = all_results[task_num][mode]
            total_exact += r["exact"]
            total_norm += r["normalized"]
            total_cont += r["containment"]
            total_samples += r["total"]
            total_f1 += r["token_f1"] * r["total"]

        print(f"\n  Mode: {mode}")
        print(f"    Exact:       {total_exact}/{total_samples} = {total_exact/max(total_samples,1)*100:.1f}%")
        print(f"    Normalized:  {total_norm}/{total_samples} = {total_norm/max(total_samples,1)*100:.1f}%")
        print(f"    Containment: {total_cont}/{total_samples} = {total_cont/max(total_samples,1)*100:.1f}%")
        print(f"    Token F1:    {total_f1/max(total_samples,1)*100:.1f}%")

    mem = torch.cuda.max_memory_allocated(device) / (1024**3) if device == "cuda" else 0
    print(f"\n  Peak VRAM: {mem:.2f} GB")
    print(f"  Model: {args.model_id}")
    print(f"  Adapter weights: {args.weights or 'None (base model)'}")


if __name__ == "__main__":
    main()
