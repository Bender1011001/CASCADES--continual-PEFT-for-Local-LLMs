"""
CASCADES-HX Training Pipeline for Qwen3.5-4B Heretic.

Loads the abliterated Qwen3.5 from the local `abliterated/` directory
and runs the 3-task continual learning curriculum using the hybrid-aware
injection from cascades/qwen35.py.

Key differences from train.py:
  - Uses inject_hybrid_cascades() with correct Qwen3.5 layer names
  - Uses compute_hybrid_layer_importance() (stratified D-MoLE)
  - Phase transitions skip SSM layer demotion (SSM layers use RecurrentSafe)
  - trust_remote_code=True for Qwen3.5 custom architecture

Usage:
    python train_qwen35.py
    python train_qwen35.py --epochs 2 --no-sleep
    python train_qwen35.py --refusal_vector path/to/vec.pt
"""

from __future__ import annotations

import argparse
import math
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from cascades.config import AblationConfig, DEFAULT_CONFIG
from cascades.adapters import CASCADESAdapter, CASCADESLinear
from cascades.injection import (
    estimate_quant_noise,
    batched_null_space_extraction,
    batched_autopoiesis_and_svc,
)
from cascades.qwen35 import (
    compute_hybrid_layer_importance,
    inject_hybrid_cascades,
    lock_abliteration_permanent,
    _LINEAR_ATTN_PATTERNS,
)
from cascades.data import prepare_data, diagnose_per_example_loss, NUM_TASKS
from cascades.eval import evaluate_generative
from cascades.sleep import SleepConsolidation, SleepConfig
from cascades.vram_monitor import log_vram, clear_cache, reset_peak_stats, check_oom_risk

MODEL_PATH = "abliterated"
OUTPUT_PREFIX = "cascades_hx"


# ---------------------------------------------------------------------------
# Optimizer builder (identical logic to train.py)
# ---------------------------------------------------------------------------

def build_optimizer(model, config, lr_liquid=2e-3, lr_gate=5e-4, lr_funlora=5e-5):
    param_groups = []
    assigned_ids = set()

    current_critical = list({
        m.adapter for m in model.modules()
        if isinstance(m, CASCADESLinear) and m.is_critical
    })
    current_funlora = list({
        m.adapter for m in model.modules()
        if isinstance(m, CASCADESLinear) and not m.is_critical
    })

    def unique_params(p_list):
        seen = set()
        result = []
        for p in p_list:
            pid = id(p)
            if pid not in seen and pid not in assigned_ids:
                seen.add(pid)
                result.append(p)
        return result

    liquid_params = unique_params([p for a in current_critical for p in a.liquid_core.parameters()])
    if liquid_params:
        param_groups.append({"params": liquid_params, "lr": lr_liquid})
        assigned_ids.update(id(p) for p in liquid_params)

    if config.enable_gainlora_gate:
        gate_params = unique_params([
            p for a in current_critical if hasattr(a, "gate_proj")
            for p in a.gate_proj.parameters()
        ])
        if gate_params:
            param_groups.append({"params": gate_params, "lr": lr_gate})
            assigned_ids.update(id(p) for p in gate_params)

    funlora_params = unique_params([p for a in current_funlora for p in a.parameters()])
    if funlora_params:
        param_groups.append({"params": funlora_params, "lr": lr_funlora})
        assigned_ids.update(id(p) for p in funlora_params)

    stiefel_ids = {id(a.U_shared) for a in current_critical} | {id(a.V_shared) for a in current_critical}
    fallback_params = unique_params([
        p for p in model.parameters()
        if p.requires_grad and id(p) not in stiefel_ids
    ])
    if fallback_params:
        param_groups.append({"params": fallback_params, "lr": 5e-4})

    return optim.Adam(param_groups), current_critical


def cleanup_optimizer_state(adapter, optimizer):
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
# Main
# ---------------------------------------------------------------------------

def train_qwen35(
    seed: int = 42,
    top_p: float = 0.15,
    rank: int = 4,
    max_length: int = 256,
    epochs: int = 2,
    eval_em: bool = False,
    enable_sleep: bool = True,
    refusal_vector_path: str = "",
    config: AblationConfig = DEFAULT_CONFIG,
) -> np.ndarray:

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 60)
    print("CASCADES-HX  |  Qwen3.5-4B Heretic")
    print("=" * 60)
    print(f"  Model path : {MODEL_PATH}")
    print(f"  Device     : {device}")
    print(f"  top_p      : {top_p}  (D-MoLE stratified)")
    print(f"  rank       : {rank}")
    print(f"  max_length : {max_length}")
    print(f"  Epochs/task: {epochs}")
    print(f"  Sleep      : {enable_sleep}")
    print("=" * 60)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # Qwen3.5 tokenizer_config.json specifies "TokenizersBackend" (Transformers 5.x class).
    # Fall back to PreTrainedTokenizerFast which can load tokenizer.json directly in 4.x.
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    except ValueError:
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    log_vram("after_model_load", device)
    reset_peak_stats(device)

    quant_noise = estimate_quant_noise(model)
    print(f"Quantization noise std: {quant_noise:.6f}")

    # Stratified D-MoLE
    probe_loader = prepare_data(tokenizer, 0, base_seed=seed, max_length=max_length)
    layer_importance = compute_hybrid_layer_importance(
        model, probe_loader, device, top_p=top_p, config=config,
    )
    clear_cache("after_dmole_probe", device)

    # Hybrid injection
    critical_adapters, funlora_adapters = inject_hybrid_cascades(
        model, rank=rank, layer_importance=layer_importance, config=config,
    )
    for a in critical_adapters:
        a.quant_noise_std.fill_(quant_noise)

    log_vram("after_adapter_injection", device)
    reset_peak_stats(device)

    # Abliteration lock (optional — requires saved refusal vector)
    if refusal_vector_path:
        try:
            rvec = torch.load(refusal_vector_path, map_location="cpu")
            if isinstance(rvec, dict):
                rvec = next(iter(rvec.values()))
            locked = lock_abliteration_permanent(critical_adapters, rvec)
            print(f"Abliteration permanently locked: {locked} adapters protected")
        except Exception as e:
            print(f"[WARNING] Could not load refusal vector: {e}")
    else:
        print("[INFO] No --refusal_vector provided. Abliteration lock skipped.")
        print("       To extract it: run forward passes on harmful/harmless prompts")
        print("       at layers [31,30,29,28,26,27,25,24,23], save mean-difference vector.")

    sleep_engine = None
    if enable_sleep:
        sleep_engine = SleepConsolidation(SleepConfig(verbose=True))
        print("Sleep Consolidation: ENABLED")

    num_tasks = NUM_TASKS
    accuracy_matrix = np.zeros((num_tasks, num_tasks))
    start_time = time.time()

    for t in range(num_tasks):
        print(f"\n{'=' * 60}")
        print(f"--- Training Task {t} ---")
        print(f"{'=' * 60}")

        dataloader = prepare_data(tokenizer, t, base_seed=seed, max_length=max_length)
        optimizer, critical_adapters = build_optimizer(model, config)

        for ep in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch in dataloader:
                input_ids, attention_mask, labels = [x.to(device) for x in batch]

                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()

                trainable = [p for p in model.parameters() if p.grad is not None]
                if trainable:
                    torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)

                for a in critical_adapters:
                    a.full_descent_step(lr=0.005)
                    cleanup_optimizer_state(a, optimizer)

                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1

                if num_batches % 50 == 0:
                    print(f"    [Ep {ep+1} B {num_batches}] loss={loss.item():.4f}")

                if config.enable_coso_nullspace and num_batches % 25 == 0:
                    batched_null_space_extraction(critical_adapters)

                if config.enable_svc and num_batches % 50 == 0:
                    batched_autopoiesis_and_svc(critical_adapters, config=config)

                if ep == 0 and num_batches == 1:
                    log_vram("after_first_backward", device)
                    check_oom_risk(threshold_mb=7500.0, device=device)

                if enable_sleep and num_batches % 100 == 0 and num_batches > 0:
                    sleep_engine.run(critical_adapters, task_id=t)

            avg_loss = epoch_loss / max(num_batches, 1)
            print(f"Task {t}, Epoch {ep+1}/{epochs}, Avg Loss: {avg_loss:.4f}")

        # Phase transition: D-MoLE promote/demote
        # NOTE: We only demote non-SSM layers to avoid assigning standard
        # FunLoRA to linear_attn layers (which need RecurrentSafe).
        if t < num_tasks - 1 and config.enable_dmole_select:
            print("\n--- Phase Transition: D-MoLE Migration ---")
            wrappers = {
                name: m for name, m in model.named_modules()
                if isinstance(m, CASCADESLinear)
            }
            new_importance = compute_hybrid_layer_importance(
                model, dataloader, device, top_p=top_p, config=config,
            )

            promoted, demoted = 0, 0
            if new_importance:
                sorted_layers = sorted(new_importance.items(), key=lambda x: x[1], reverse=True)
                top_k   = max(2, len(sorted_layers) // 10)
                bottom_k = max(2, len(sorted_layers) // 2)

                for name, _ in sorted_layers[-bottom_k:]:
                    if name in wrappers and wrappers[name].is_critical:
                        # Skip demotion for SSM layers — they need RecurrentSafe
                        is_ssm = any(p in name for p in _LINEAR_ATTN_PATTERNS)
                        if not is_ssm and wrappers[name].demote():
                            demoted += 1

                for name, _ in sorted_layers[:top_k]:
                    if name in wrappers and not wrappers[name].is_critical:
                        if wrappers[name].promote():
                            promoted += 1

            print(f"  Promoted: {promoted}, Demoted: {demoted}")

        # Freeze subspace at task boundary
        if config.enable_coso_nullspace:
            frozen_count = 0
            for a in critical_adapters:
                a.freeze_current_subspace()
                frozen_count += a.frozen_null_basis.shape[1]
            if frozen_count > 0:
                print(f"  Froze subspace: {frozen_count} protected directions")

        if enable_sleep:
            clear_cache("before_sleep", device)
            sleep_engine.run(critical_adapters, task_id=t)
            log_vram("after_sleep", device)

        # Evaluation
        clear_cache("before_eval", device)
        print(f"\n--- Evaluation after Task {t} ---")
        with torch.inference_mode():
            for eval_t in range(t + 1):
                eval_loader = prepare_data(tokenizer, eval_t, base_seed=seed, max_length=max_length)
                total_loss, n_batches = 0.0, 0
                for batch in eval_loader:
                    if n_batches >= 50:
                        break
                    ids, mask, lbls = [x.to(device) for x in batch]
                    out = model(input_ids=ids, attention_mask=mask, labels=lbls)
                    total_loss += out.loss.item()
                    n_batches += 1
                avg_loss = total_loss / max(n_batches, 1)
                proxy_acc = math.exp(-avg_loss)
                accuracy_matrix[t, eval_t] = proxy_acc
                print(f"  Task {eval_t}: {proxy_acc * 100:.2f}%  (loss={avg_loss:.4f})")
        clear_cache("after_eval", device)

    elapsed = time.time() - start_time

    final_accs = accuracy_matrix[-1, :]
    avg_acc    = np.mean(final_accs)
    bwt        = np.mean([
        accuracy_matrix[-1, i] - accuracy_matrix[i, i]
        for i in range(num_tasks - 1)
    ])

    print(f"\n{'=' * 60}")
    print("CASCADES-HX FINAL METRICS")
    print(f"{'=' * 60}")
    print(f"Average Proxy Accuracy : {avg_acc * 100:.2f}%")
    print(f"Backward Transfer (BWT): {bwt * 100:.2f}%")
    print(f"Total Time             : {elapsed:.1f}s")
    print(f"\nAccuracy Matrix:")
    for i in range(num_tasks):
        row = " | ".join(
            f"{accuracy_matrix[i, j] * 100:6.2f}%" if accuracy_matrix[i, j] > 0 else "   --  "
            for j in range(num_tasks)
        )
        print(f"  After T{i}: {row}")

    df = pd.DataFrame(
        accuracy_matrix,
        columns=[f"Eval_T{i}" for i in range(num_tasks)],
        index=[f"Train_T{i}" for i in range(num_tasks)],
    )
    df.to_csv(f"{OUTPUT_PREFIX}_results.csv")
    print(f"\nResults saved to {OUTPUT_PREFIX}_results.csv")

    save_path = f"{OUTPUT_PREFIX}_weights.pt"
    adapter_state = {
        name: param for name, param in model.named_parameters()
        if "wrapper" in name or "adapter" in name
    }
    torch.save(adapter_state, save_path)
    print(f"Weights saved to {save_path}")

    if eval_em and device != "cpu":
        for eval_t in range(num_tasks):
            clear_cache("before_generative_eval", device)
            evaluate_generative(model, tokenizer, eval_t, device=device,
                                max_samples=25, max_new_tokens=256, verbose=True)

    return accuracy_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CASCADES-HX Training — Qwen3.5-4B Heretic")
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--top_p",           type=float, default=0.15,
                        help="Fraction of each stratum marked critical by stratified D-MoLE")
    parser.add_argument("--rank",            type=int,   default=4,
                        help="Stiefel manifold rank (4 recommended for 8GB with Qwen3.5)")
    parser.add_argument("--max_length",      type=int,   default=256,
                        help="Max token sequence length (shorter = less activation memory)")
    parser.add_argument("--epochs",          type=int,   default=2)
    parser.add_argument("--eval_em",         action="store_true")
    parser.add_argument("--no-sleep",        action="store_true")
    parser.add_argument("--refusal_vector",  type=str,   default="",
                        help="Path to saved OBLITERATUS refusal direction (.pt file)")
    args = parser.parse_args()

    train_qwen35(
        seed=args.seed,
        top_p=args.top_p,
        rank=args.rank,
        max_length=args.max_length,
        epochs=args.epochs,
        eval_em=args.eval_em,
        enable_sleep=not args.no_sleep,
        refusal_vector_path=args.refusal_vector,
    )
