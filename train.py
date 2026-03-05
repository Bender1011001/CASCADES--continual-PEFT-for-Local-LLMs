"""
CASCADES v9 Training Pipeline — Thin orchestrator.

This script is the single entry point for running CASCADES continual learning.
All adapter logic, math ops, injection, evaluation, and data loading are
imported from the `cascades` library package.

Usage:
    python train.py --eval_em
    python train.py --model_id "p-e-w/Qwen3-4B-Instruct-2507-heretic" --seed 42
    python train.py --no-sleep --epochs 3
"""

from __future__ import annotations

import argparse
import math
import sys
import time

# Windows cp1252 consoles can't print the emoji in sleep.py / adapters.py.
# Reconfigure stdout/stderr to UTF-8 so the full log renders correctly.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# CASCADES library imports — single source of truth
from cascades.config import AblationConfig, DEFAULT_CONFIG
from cascades.adapters import CASCADESAdapter, CASCADESLinear
from cascades.injection import (
    estimate_quant_noise,
    compute_layer_importance,
    inject_cascades,
    batched_null_space_extraction,
    batched_autopoiesis_and_svc,
)
from cascades.data import prepare_data, diagnose_per_example_loss, NUM_TASKS
from cascades.eval import evaluate_generative, STRUCTURED_SYSTEM_PROMPT
from cascades.sleep import SleepConsolidation, SleepConfig
from cascades.vram_monitor import log_vram, clear_cache, reset_peak_stats, check_oom_risk


# ---------------------------------------------------------------------------
# Optimizer builder
# ---------------------------------------------------------------------------

def build_optimizer(
    model: nn.Module,
    config: AblationConfig,
    lr_liquid: float = 2e-3,
    lr_gate: float = 5e-4,
    lr_funlora: float = 5e-5,
) -> tuple[optim.Adam, list[CASCADESAdapter]]:
    """Build parameter-grouped Adam optimizer.

    Groups:
      1. Liquid core parameters (Hebbian routing) — lr_liquid
      2. GainLoRA gate parameters — lr_gate
      3. FunLoRA rank-1 parameters — lr_funlora
      4. Any remaining trainable params — 5e-4 fallback

    U_shared / V_shared are EXCLUDED from Adam (Riemannian-only updates).

    Returns:
        (optimizer, list of current critical adapters)
    """
    param_groups: list[dict] = []
    assigned_ids: set[int] = set()

    # Discover current adapter allocation
    current_critical = list({
        m.adapter for m in model.modules()
        if isinstance(m, CASCADESLinear) and m.is_critical
    })
    current_funlora = list({
        m.adapter for m in model.modules()
        if isinstance(m, CASCADESLinear) and not m.is_critical
    })

    def unique_params(p_list: list[nn.Parameter]) -> list[nn.Parameter]:
        seen: set[int] = set()
        result = []
        for p in p_list:
            pid = id(p)
            if pid not in seen and pid not in assigned_ids:
                seen.add(pid)
                result.append(p)
        return result

    # Group 1: Liquid cores
    liquid_params = unique_params([
        p for a in current_critical for p in a.liquid_core.parameters()
    ])
    if liquid_params:
        param_groups.append({"params": liquid_params, "lr": lr_liquid})
        assigned_ids.update(id(p) for p in liquid_params)

    # Group 2: GainLoRA gates
    if config.enable_gainlora_gate:
        gate_params = unique_params([
            p for a in current_critical
            if hasattr(a, "gate_proj")
            for p in a.gate_proj.parameters()
        ])
        if gate_params:
            param_groups.append({"params": gate_params, "lr": lr_gate})
            assigned_ids.update(id(p) for p in gate_params)

    # Group 3: FunLoRA params
    funlora_params = unique_params([
        p for a in current_funlora for p in a.parameters()
    ])
    if funlora_params:
        param_groups.append({"params": funlora_params, "lr": lr_funlora})
        assigned_ids.update(id(p) for p in funlora_params)

    # Stiefel bases — NEVER in Adam
    stiefel_ids = set()
    for a in current_critical:
        stiefel_ids.add(id(a.U_shared))
        stiefel_ids.add(id(a.V_shared))

    # Group 4: Remaining trainable params
    fallback_params = unique_params([
        p for p in model.parameters()
        if p.requires_grad and id(p) not in stiefel_ids
    ])
    if fallback_params:
        param_groups.append({"params": fallback_params, "lr": 5e-4})

    return optim.Adam(param_groups), current_critical


# ---------------------------------------------------------------------------
# Surgical optimizer state cleanup (Fix #2 from math review)
# ---------------------------------------------------------------------------

def cleanup_optimizer_state(
    adapter: CASCADESAdapter,
    optimizer: optim.Optimizer,
) -> None:
    """Surgically zero dead dimension's momentum instead of flushing all state.

    When a rank contracts (breathing manifold), only the dead dimension's
    Adam momentum should be reset. Flushing ALL state (the old behavior)
    kills momentum for healthy dimensions and slows convergence.
    """
    param = adapter.liquid_core.core_pool
    if getattr(adapter, "contracted_this_step", False):
        dead_idx = getattr(adapter, "_last_dead_idx", None)
        if param in optimizer.state and dead_idx is not None:
            state = optimizer.state[param]
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    state[key][..., dead_idx, :] = 0.0
                    state[key][..., :, dead_idx] = 0.0
        adapter.contracted_this_step = False


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_cascades(
    seed: int = 42,
    dmole_threshold: float = 0.22,
    model_id: str = "p-e-w/Qwen3-4B-Instruct-2507-heretic",
    output_prefix: str = "cascades_v9",
    lr_liquid: float = 2e-3,
    lr_gate: float = 5e-4,
    lr_funlora: float = 5e-5,
    epochs: int = 2,
    eval_em: bool = False,
    enable_sleep: bool = True,
    config: AblationConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """Run complete CASCADES v9 continual learning pipeline.

    Args:
        seed: Global random seed.
        dmole_threshold: Activation selection threshold for D-MoLE.
        model_id: HuggingFace model ID.
        output_prefix: Prefix for output files.
        lr_liquid: Learning rate for liquid core parameters.
        lr_gate: Learning rate for GainLoRA gates.
        lr_funlora: Learning rate for FunLoRA rank-1 adapters.
        epochs: Training epochs per task.
        eval_em: Run generative exact match evaluation after training.
        enable_sleep: Enable bio-inspired sleep consolidation.
        config: Ablation configuration.

    Returns:
        Accuracy matrix (num_tasks x num_tasks).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- Print configuration ---
    print("=" * 60)
    print("CASCADES v9 TRAINING PIPELINE")
    print("=" * 60)
    print(f"  Model: {model_id}")
    print(f"  Device: {device}")
    print(f"  Config: {config}")
    print("=" * 60)

    # --- Load model with NF4 quantization ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
    )

    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.config.use_cache = False

    # --- VRAM checkpoint: after model load ---
    log_vram("after_model_load", device)
    reset_peak_stats(device)

    # --- Quantization noise estimation ---
    quant_noise = estimate_quant_noise(model)
    print(f"Quantization noise std: {quant_noise:.6f}")

    # --- D-MoLE layer importance ---
    layer_importance = None
    if config.enable_dmole_select:
        probe_loader = prepare_data(tokenizer, 0, base_seed=seed)
        layer_importance = compute_layer_importance(
            model, probe_loader, device,
            threshold=dmole_threshold, config=config,
        )
        clear_cache("after_dmole_probe", device)

    # --- Inject adapters ---
    critical_adapters, funlora_adapters = inject_cascades(
        model, rank=8, layer_importance=layer_importance, config=config,
    )

    # Set quantization noise on critical adapters
    for a in critical_adapters:
        a.quant_noise_std.fill_(quant_noise)

    # --- VRAM checkpoint: after adapter injection ---
    log_vram("after_adapter_injection", device)
    reset_peak_stats(device)

    # --- Sleep engine ---
    sleep_engine = None
    if enable_sleep:
        sleep_engine = SleepConsolidation(SleepConfig(
            verbose=True,
            enable_cross_adapter_dedup=config.enable_ambient_dedup,
        ))
        print(f"Sleep Consolidation Engine: ENABLED "
              f"(ambient dedup={'ON' if config.enable_ambient_dedup else 'OFF'})")

    # --- Training loop ---
    num_tasks = NUM_TASKS
    accuracy_matrix = np.zeros((num_tasks, num_tasks))
    start_time = time.time()

    for t in range(num_tasks):
        print(f"\n{'=' * 60}")
        print(f"--- Training Task {t} ---")
        print(f"{'=' * 60}")

        dataloader = prepare_data(tokenizer, t, base_seed=seed)
        optimizer, critical_adapters = build_optimizer(
            model, config, lr_liquid, lr_gate, lr_funlora,
        )

        for ep in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

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

                # Gradient clipping
                trainable = [p for p in model.parameters() if p.grad is not None]
                if trainable:
                    torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)

                # CASCADES Riemannian descent (critical adapters only)
                for a in critical_adapters:
                    a.full_descent_step(lr=0.005)
                    # Fix #2: Surgical optimizer state cleanup
                    cleanup_optimizer_state(a, optimizer)

                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1

                # Log progress
                if num_batches % 50 == 0:
                    print(f"    [Ep {ep+1} B {num_batches}] loss={loss.item():.4f}")

                # Global batched operations
                if config.enable_coso_nullspace and num_batches % 25 == 0:
                    batched_null_space_extraction(critical_adapters)

                if config.enable_svc and num_batches % 50 == 0:
                    batched_autopoiesis_and_svc(critical_adapters, config=config)

                # Memory check on first batch
                if ep == 0 and num_batches == 1:
                    log_vram("after_first_backward", device)
                    check_oom_risk(threshold_mb=7500.0, device=device)

                # Intra-task micro-sleep
                if enable_sleep and num_batches % 100 == 0 and num_batches > 0:
                    sleep_engine.run(critical_adapters, task_id=t)

            avg_loss = epoch_loss / max(num_batches, 1)
            print(f"Task {t}, Epoch {ep + 1}/{epochs}, Avg Loss: {avg_loss:.4f}")

        # --- D-MoLE phase transition (promote/demote) ---
        if t < num_tasks - 1 and config.enable_dmole_select:
            print(f"\n--- Phase Transition: D-MoLE Migration ---")
            wrappers = {
                name: m for name, m in model.named_modules()
                if isinstance(m, CASCADESLinear)
            }
            importance = compute_layer_importance(
                model, dataloader, device, threshold=dmole_threshold, config=config,
            )
            promoted, demoted = 0, 0
            if importance:
                sorted_layers = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                top_k = max(2, len(sorted_layers) // 10)
                bottom_k = max(2, len(sorted_layers) // 2)

                for name, _ in sorted_layers[-bottom_k:]:
                    if name in wrappers and wrappers[name].is_critical:
                        if wrappers[name].demote():
                            demoted += 1

                for name, _ in sorted_layers[:top_k]:
                    if name in wrappers and not wrappers[name].is_critical:
                        if wrappers[name].promote():
                            promoted += 1

            print(f"  Promoted: {promoted}, Demoted: {demoted}")

        # --- v10 BWT fix: Freeze occupied subspace at task boundary ---
        # Snapshot the gradient directions that were important during this task.
        # Future tasks' gradients will be projected out of these directions.
        if config.enable_coso_nullspace:
            frozen_count = 0
            for a in critical_adapters:
                a.freeze_current_subspace()
                frozen_count += a.frozen_null_basis.shape[1]
            if frozen_count > 0:
                print(f"  🧊 Froze subspace: {frozen_count} total protected "
                      f"directions across {len(critical_adapters)} adapters")

        # --- Sleep consolidation between tasks ---
        if enable_sleep:
            clear_cache("before_sleep", device)
            sleep_engine.run(critical_adapters, task_id=t)
            log_vram("after_sleep", device)

        # --- Evaluation ---
        clear_cache("before_eval", device)
        print(f"\n--- Evaluation after Task {t} ---")
        with torch.inference_mode():
            for eval_t in range(t + 1):
                eval_loader = prepare_data(tokenizer, eval_t, base_seed=seed)
                total_loss, n_batches = 0.0, 0
                max_eval_batches = 50  # Cap eval samples to save VRAM
                for batch in eval_loader:
                    if n_batches >= max_eval_batches:
                        break
                    ids, mask, lbls = [x.to(device) for x in batch]
                    out = model(input_ids=ids, attention_mask=mask, labels=lbls)
                    total_loss += out.loss.item()
                    n_batches += 1

                avg_loss = total_loss / max(n_batches, 1)
                proxy_acc = math.exp(-avg_loss)
                accuracy_matrix[t, eval_t] = proxy_acc
                print(f"  Task {eval_t}: {proxy_acc * 100:.2f}% (loss: {avg_loss:.4f})")
        clear_cache("after_eval", device)

    elapsed = time.time() - start_time

    # --- Final metrics ---
    final_accs = accuracy_matrix[-1, :]
    avg_acc = np.mean(final_accs)
    bwt = np.mean([
        accuracy_matrix[-1, i] - accuracy_matrix[i, i]
        for i in range(num_tasks - 1)
    ])

    print(f"\n{'=' * 60}")
    print("FINAL CASCADES METRICS")
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

    # --- Save results ---
    df = pd.DataFrame(
        accuracy_matrix,
        columns=[f"Eval_T{i}" for i in range(num_tasks)],
        index=[f"Train_T{i}" for i in range(num_tasks)],
    )
    results_csv = f"{output_prefix}_results.csv"
    df.to_csv(results_csv)
    print(f"\nResults saved to {results_csv}")

    # --- Per-example diagnostic ---
    print("\nRunning per-example loss diagnostic...")
    diagnose_per_example_loss(model, tokenizer, device, seed=seed)

    # --- Save weights ---
    print("\nSaving CASCADES adapter weights...")
    save_path = f"{output_prefix}_weights.pt"
    adapter_state = {
        name: param for name, param in model.named_parameters()
        if "wrapper" in name or "adapter" in name
    }
    torch.save(adapter_state, save_path)
    print(f"Weights saved to {save_path}")

    # --- Generative EM evaluation (optional) ---
    if eval_em and device != "cpu":
        print(f"\n{'=' * 60}")
        print("GENERATIVE EXACT MATCH EVALUATION")
        print(f"{'=' * 60}")

        em_results = {}
        for eval_t in range(num_tasks):
            print(f"\n--- Task {eval_t} ---")
            clear_cache("before_generative_eval", device)
            task_em = evaluate_generative(
                model, tokenizer, eval_t, device=device,
                max_samples=25, max_new_tokens=256,
                use_system_prompt=True, verbose=True,
            )
            em_results[eval_t] = task_em

        print(f"\n{'=' * 60}")
        print("EM SUMMARY")
        print(f"{'=' * 60}")
        for t_id, res in em_results.items():
            print(
                f"  Task {t_id}: Exact={res['exact_match_rate']*100:.1f}% "
                f"Normalized={res['normalized_match_rate']*100:.1f}% "
                f"Containment={res['containment_match_rate']*100:.1f}%"
            )
        avg_cont = sum(r["containment_match_rate"] for r in em_results.values()) / len(em_results)
        print(f"\n  Avg Containment: {avg_cont*100:.1f}% (Proxy ACC: {avg_acc*100:.2f}%)")

    return accuracy_matrix


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CASCADES v9 Training Pipeline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dmole_threshold", type=float, default=0.22)
    parser.add_argument("--model_id", type=str, default="p-e-w/Qwen3-4B-Instruct-2507-heretic")
    parser.add_argument("--output_prefix", type=str, default="cascades_v9")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--eval_em", action="store_true")
    parser.add_argument("--no-sleep", action="store_true")
    args = parser.parse_args()

    train_cascades(
        seed=args.seed,
        dmole_threshold=args.dmole_threshold,
        model_id=args.model_id,
        output_prefix=args.output_prefix,
        epochs=args.epochs,
        eval_em=args.eval_em,
        enable_sleep=not args.no_sleep,
    )
