"""
Fair LoRA Baseline for CASCADES comparison.

Implements standard LoRA continual learning using IDENTICAL conditions to
CASCADES: same model (Qwen3-4B NF4), same data pipeline (prepare_data),
same evaluation (proxy + generative), same optimizer (AdamW), same LR schedule.

The ONLY difference: uses PEFT LoraConfig instead of CASCADESAdapter.
No Riemannian updates, no EAR, no sleep, no D-MoLE.

This isolates the CASCADES meta-architecture contribution from basic LoRA.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from cascades.data import prepare_data, TASK_FILES, NUM_TASKS
from cascades.eval import evaluate_generative


# ---------------------------------------------------------------------------
# Target modules — same layers CASCADES injects into
# ---------------------------------------------------------------------------

# These match the default target_modules in inject_cascades()
CASCADES_TARGET_MODULES = ["q_proj", "v_proj", "up_proj", "down_proj"]


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_lora_baseline(
    model_name: str = "p-e-w/Qwen3-4B-Instruct-2507-heretic",
    task_files: Optional[list[str]] = None,
    num_epochs: int = 2,
    lr: float = 5e-4,
    rank: int = 8,
    eval_samples: int = 50,
    seed: int = 42,
    eval_em: bool = False,
    output_prefix: str = "lora_baseline",
    max_new_tokens: int = 512,
    do_sample: bool = False,
    temperature: float = 0.0,
    few_shot: int = 0,
) -> dict:
    """Train a standard LoRA baseline with the same pipeline as CASCADES.

    Args:
        model_name: HuggingFace model ID (same NF4 quantized model).
        task_files: List of JSONL task files. Defaults to TASK_FILES.
        num_epochs: Training epochs per task.
        lr: Learning rate for AdamW optimizer.
        rank: LoRA rank (default 8 to match CASCADES).
        eval_samples: Number of samples for generative evaluation.
        seed: Random seed for reproducibility.
        eval_em: Whether to run generative exact match evaluation.
        output_prefix: Prefix for output files.
        max_new_tokens: Max tokens for generative eval.
        do_sample: Whether to use sampling in generative eval.
        temperature: Temperature for generative eval.
        few_shot: Number of few-shot examples (0 = zero-shot).

    Returns:
        Dictionary containing:
            - accuracy_matrix: np.ndarray (num_tasks x num_tasks)
            - em_results: dict of per-task generative eval results (if eval_em)
            - wall_time_s: total wall time
            - vram_peak_mb: peak VRAM in MB
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if task_files is None:
        task_files = list(TASK_FILES)

    num_tasks = len(task_files)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- Print configuration ---
    print("=" * 60)
    print("LORA BASELINE TRAINING")
    print("=" * 60)
    print(f"  Model: {model_name}")
    print(f"  Device: {device}")
    print(f"  Rank: {rank}")
    print(f"  LR: {lr}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Tasks: {num_tasks}")
    print("=" * 60)

    # --- Load model with NF4 quantization (identical to train.py) ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto",
    )

    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.config.use_cache = False

    # --- Apply standard LoRA ---
    try:
        from peft import LoraConfig, get_peft_model, TaskType

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank * 2,       # alpha = 2 * rank (standard scaling)
            lora_dropout=0.05,
            target_modules=CASCADES_TARGET_MODULES,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print("  Using PEFT LoraConfig")

    except ImportError:
        # Fallback: manual LoRA injection
        print("  PEFT not available — using manual LoRA injection")
        _inject_manual_lora(model, rank=rank, target_modules=CASCADES_TARGET_MODULES)

    # --- Training loop ---
    accuracy_matrix = np.zeros((num_tasks, num_tasks))
    start_time = time.time()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for t in range(num_tasks):
        print(f"\n{'=' * 60}")
        print(f"--- Training Task {t} ---")
        print(f"{'=' * 60}")

        dataloader = prepare_data(tokenizer, t, base_seed=seed)

        # AdamW optimizer — same as CASCADES would use for its Adam groups
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable_params, lr=lr)

        for ep in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            model.train()

            for batch in dataloader:
                input_ids, attention_mask, labels = [x.to(device) for x in batch]

                optimizer.zero_grad()
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                loss.backward()

                # Gradient clipping (same as train.py:294)
                trainable = [p for p in model.parameters() if p.grad is not None]
                if trainable:
                    torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)

                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1

                if num_batches % 50 == 0:
                    print(f"    [Ep {ep+1} B {num_batches}] loss={loss.item():.4f}")

                # Memory check on first batch
                if ep == 0 and num_batches == 1 and torch.cuda.is_available():
                    mem = torch.cuda.max_memory_allocated(device) / (1024**3)
                    print(f"    [MEMORY] Peak VRAM: {mem:.2f} GB")
                    if mem > 7.8:
                        print("    [WARNING] VRAM above 7.8GB — OOM risk!")

            avg_loss = epoch_loss / max(num_batches, 1)
            print(f"Task {t}, Epoch {ep + 1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

        # --- NO task-boundary operations ---
        # No subspace freezing, no sleep, no D-MoLE migration
        # This is the key difference from CASCADES

        # --- Evaluation ---
        print(f"\n--- Evaluation after Task {t} ---")
        with torch.inference_mode():
            for eval_t in range(t + 1):
                eval_loader = prepare_data(tokenizer, eval_t, base_seed=seed)
                total_loss, n_batches = 0.0, 0
                for batch in eval_loader:
                    ids, mask, lbls = [x.to(device) for x in batch]
                    out = model(input_ids=ids, attention_mask=mask, labels=lbls)
                    total_loss += out.loss.item()
                    n_batches += 1

                avg_loss = total_loss / max(n_batches, 1)
                proxy_acc = math.exp(-avg_loss)
                accuracy_matrix[t, eval_t] = proxy_acc
                print(f"  Task {eval_t}: {proxy_acc * 100:.2f}% (loss: {avg_loss:.4f})")

    elapsed = time.time() - start_time

    # --- Final metrics ---
    final_accs = accuracy_matrix[-1, :]
    avg_acc = np.mean(final_accs)
    bwt = np.mean([
        accuracy_matrix[-1, i] - accuracy_matrix[i, i]
        for i in range(num_tasks - 1)
    ]) if num_tasks >= 2 else 0.0

    print(f"\n{'=' * 60}")
    print("FINAL LORA BASELINE METRICS")
    print(f"{'=' * 60}")
    print(f"Average Accuracy: {avg_acc * 100:.2f}%")
    print(f"Backward Transfer: {bwt * 100:.2f}%")
    print(f"Total Time: {elapsed:.1f}s")
    print(f"\nAccuracy Matrix:")
    for i in range(num_tasks):
        row = " | ".join(
            f"{accuracy_matrix[i, j] * 100:6.2f}%" if accuracy_matrix[i, j] > 0 else "   --  "
            for j in range(num_tasks)
        )
        print(f"  After T{i}: {row}")

    # --- VRAM peak ---
    vram_peak_mb = 0.0
    if torch.cuda.is_available():
        vram_peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    # --- Generative EM evaluation (optional) ---
    em_results = {}
    if eval_em and device != "cpu":
        print(f"\n{'=' * 60}")
        print("GENERATIVE EXACT MATCH EVALUATION (LoRA Baseline)")
        print(f"{'=' * 60}")

        for eval_t in range(num_tasks):
            print(f"\n--- Task {eval_t} ---")
            if few_shot > 0:
                # Use few-shot evaluation via evaluate.py's approach
                from evaluate import build_fewshot_prompt, load_task_data
                task_em = _evaluate_generative_fewshot(
                    model, tokenizer, eval_t, device=device,
                    max_samples=eval_samples, max_new_tokens=max_new_tokens,
                    few_shot=few_shot,
                )
            else:
                task_em = evaluate_generative(
                    model, tokenizer, eval_t, device=device,
                    max_samples=eval_samples, max_new_tokens=max_new_tokens,
                    use_system_prompt=True, verbose=True,
                )
            em_results[eval_t] = task_em

    return {
        "accuracy_matrix": accuracy_matrix,
        "em_results": em_results,
        "wall_time_s": elapsed,
        "vram_peak_mb": vram_peak_mb,
    }


# ---------------------------------------------------------------------------
# Manual LoRA injection fallback (if PEFT not installed)
# ---------------------------------------------------------------------------

class ManualLoRALayer(nn.Module):
    """Minimal LoRA adapter: W' = W + (B @ A) * (alpha/r)."""

    def __init__(self, original: nn.Module, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.original = original

        # Determine dimensions
        if hasattr(original, "out_features"):
            out_features = original.out_features
            in_features = original.in_features
        else:
            # Linear4bit — get shape from weight
            w = original.weight
            out_features = w.shape[0]
            in_features = w.shape[1] if len(w.shape) > 1 else w.numel() // out_features

        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.original(x)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out


def _inject_manual_lora(
    model: nn.Module,
    rank: int = 8,
    target_modules: list[str] | None = None,
) -> None:
    """Inject manual LoRA layers into the model (PEFT fallback)."""
    if target_modules is None:
        target_modules = CASCADES_TARGET_MODULES

    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False

    replaced = 0
    for name, module in dict(model.named_modules()).items():
        if any(t in name for t in target_modules) and (
            isinstance(module, nn.Linear)
            or type(module).__name__ == "Linear4bit"
        ):
            parts = name.split(".")
            parent_name = ".".join(parts[:-1])
            child_name = parts[-1]
            try:
                parent = model.get_submodule(parent_name)
                lora_layer = ManualLoRALayer(module, rank=rank, alpha=rank * 2)
                lora_layer = lora_layer.to(module.weight.device)
                setattr(parent, child_name, lora_layer)
                replaced += 1
            except AttributeError:
                pass

    print(f"  Manual LoRA: injected {replaced} adapters (rank={rank})")


# ---------------------------------------------------------------------------
# Few-shot generative eval helper
# ---------------------------------------------------------------------------

def _evaluate_generative_fewshot(
    model, tokenizer, task_number: int,
    device: str = "cuda", max_samples: int = 50,
    max_new_tokens: int = 512, few_shot: int = 2,
) -> dict:
    """Generative eval with few-shot prompting."""
    import json
    from pathlib import Path
    from cascades.eval import (
        extract_answer_from_cot, answers_match,
        STRUCTURED_SYSTEM_PROMPT,
    )

    files = list(TASK_FILES)
    if task_number >= len(files):
        files = [
            "data/task0_gsm8k_cot.jsonl",
            "data/task1_arc_cot.jsonl",
            "data/task2_csqa_cot.jsonl",
            "data/task0_logic_cot.jsonl",
            "data/task1_decomp_cot.jsonl",
        ]
    file_path = files[task_number % len(files)]
    path = Path(file_path)
    if not path.exists():
        path = Path(__file__).parent / file_path

    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
                if len(samples) >= max_samples + few_shot:
                    break

    # Use first few_shot samples as examples, evaluate the rest
    example_samples = samples[:few_shot]
    eval_samples_list = samples[few_shot:few_shot + max_samples]

    model.eval()
    original_use_cache = getattr(model.config, "use_cache", True)
    model.config.use_cache = True

    exact_matches = 0
    normalized_matches = 0
    containment_matches = 0
    details = []

    for i, sample in enumerate(eval_samples_list):
        # Build few-shot prompt
        examples_text = "Here are examples of the expected format:\n\n"
        for j, ex in enumerate(example_samples):
            ref = extract_answer_from_cot(ex["response"])
            examples_text += f"Example {j+1}:\nQ: {ex['prompt'][:200]}\nA: {ref}\n\n"

        messages = [
            {"role": "system", "content": STRUCTURED_SYSTEM_PROMPT},
            {"role": "user", "content": examples_text + "Now answer:\n" + sample["prompt"]},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        generated_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        reference_answer = extract_answer_from_cot(sample["response"])
        generated_answer = extract_answer_from_cot(generated_text)

        is_exact = generated_answer.strip() == reference_answer.strip()
        is_normalized = answers_match(generated_answer, reference_answer, strict=True)
        is_containment = answers_match(generated_answer, reference_answer, strict=False)

        if is_exact:
            exact_matches += 1
        if is_normalized:
            normalized_matches += 1
        if is_containment:
            containment_matches += 1

        details.append({
            "index": i,
            "reference_answer": reference_answer,
            "generated_answer": generated_answer,
            "exact_match": is_exact,
            "normalized_match": is_normalized,
            "containment_match": is_containment,
        })

    model.config.use_cache = original_use_cache

    total = len(eval_samples_list)
    return {
        "task_number": task_number,
        "total_samples": total,
        "exact_match_rate": exact_matches / total if total > 0 else 0.0,
        "normalized_match_rate": normalized_matches / total if total > 0 else 0.0,
        "containment_match_rate": containment_matches / total if total > 0 else 0.0,
        "exact_matches": exact_matches,
        "normalized_matches": normalized_matches,
        "containment_matches": containment_matches,
        "details": details,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LoRA Baseline Training")
    parser.add_argument("--model_name", type=str,
                        default="p-e-w/Qwen3-4B-Instruct-2507-heretic")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_em", action="store_true")
    parser.add_argument("--eval_samples", type=int, default=50)
    args = parser.parse_args()

    results = train_lora_baseline(
        model_name=args.model_name,
        num_epochs=args.epochs,
        lr=args.lr,
        rank=args.rank,
        seed=args.seed,
        eval_em=args.eval_em,
        eval_samples=args.eval_samples,
    )
    print(f"\nDone. Accuracy matrix shape: {results['accuracy_matrix'].shape}")
    print(f"Wall time: {results['wall_time_s']:.1f}s")
    print(f"Peak VRAM: {results['vram_peak_mb']:.0f} MB")
