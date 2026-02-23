"""
Unified experiment runner for CASCADES reproducibility.

Usage
-----
# Run CASCADES v5 with 3 seeds, save to results/
python experiments/run_experiment.py --method cascades_v5 --seeds 0 1 2

# Run LoRA baseline
python experiments/run_experiment.py --method lora --seeds 0 1 2

# Ablation: disable EAR and CoSO
python experiments/run_experiment.py --method cascades_v5 \\
    --no-ear --no-coso --seeds 0

# Fast smoke-test on TinyLlama (fp32, no VRAM needed beyond 6 GB)
python experiments/run_experiment.py --method cascades_v5 \\
    --model tinyllama --epochs 1 --seeds 0

All results are written to results/runs/<timestamp>_<method>_seed<N>.json
and a summary CSV is appended to results/summary.csv.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Ensure project root is on the path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from cascades.data import NUM_TASKS, prepare_dataloader
from cascades.metrics import average_accuracy, backward_transfer, full_report, proxy_accuracy_from_loss


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "qwen3_4b": {
        "model_id": "p-e-w/Qwen3-4B-Instruct-2507-heretic",
        "load_in_4bit": True,
        "compute_dtype": "float16",
        "description": "Qwen3-4B (4-bit QLoRA, 8 GB VRAM)",
    },
    "tinyllama": {
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "load_in_4bit": False,
        "compute_dtype": "float32",
        "description": "TinyLlama-1.1B (fp32, debug/CI)",
    },
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CASCADES reproducible experiment runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--method",
        choices=["cascades_v5", "lora"],
        default="cascades_v5",
        help="Which method to run.",
    )
    p.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()),
        default="qwen3_4b",
        help="Base model to adapt.",
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0],
        help="List of random seeds.  Run once per seed and report mean ± std.",
    )
    p.add_argument("--num-tasks", type=int, default=NUM_TASKS, help="Number of sequential tasks.")
    p.add_argument("--epochs", type=int, default=3, help="Training epochs per task.")
    p.add_argument("--rank", type=int, default=8, help="Adapter rank r.")
    p.add_argument("--batch-size", type=int, default=2, help="Mini-batch size.")
    p.add_argument("--lr-shared", type=float, default=1e-4, help="LR for shared subspace (U, V).")
    p.add_argument("--lr-task", type=float, default=5e-3, help="LR for task-specific cores.")
    p.add_argument("--lr-gate", type=float, default=1e-3, help="LR for gate parameters.")
    p.add_argument("--output-dir", type=Path, default=ROOT / "results" / "runs", help="Results directory.")

    # Ablation toggles (CASCADES only)
    abl = p.add_argument_group("Ablation flags (CASCADES only)")
    abl.add_argument("--no-paca",      dest="paca",      action="store_false", default=True)
    abl.add_argument("--no-deal",      dest="deal",      action="store_false", default=True)
    abl.add_argument("--no-gate",      dest="gate",      action="store_false", default=True)
    abl.add_argument("--no-coso",      dest="coso",      action="store_false", default=True)
    abl.add_argument("--no-ear",       dest="ear",       action="store_false", default=True)
    abl.add_argument("--no-svc",       dest="svc",       action="store_false", default=True)
    abl.add_argument("--no-dmole",     dest="dmole",     action="store_false", default=True)
    abl.add_argument("--no-funlora",   dest="funlora",   action="store_false", default=True)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Single-seed experiment
# ---------------------------------------------------------------------------

def run_one_seed(args: argparse.Namespace, seed: int) -> dict:
    """Run a complete continual-learning experiment for one random seed.

    Returns a dictionary with all metrics and metadata.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = MODEL_CONFIGS[args.model]

    print(f"\n{'='*60}")
    print(f"  Method : {args.method}")
    print(f"  Model  : {cfg['description']}")
    print(f"  Seed   : {seed}")
    print(f"  Device : {device}")
    print(f"{'='*60}")

    # ----- Load model -----
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if cfg["load_in_4bit"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_id"], quantization_config=bnb_config, device_map="auto"
        )
    else:
        compute_dtype = getattr(torch, cfg["compute_dtype"])
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_id"], torch_dtype=compute_dtype, device_map="auto"
        )

    # ----- Inject adapters -----
    if args.method == "cascades_v5":
        critical_adapters, funlora_adapters = _setup_cascades(model, args, device, tokenizer)
        all_adapters = critical_adapters + funlora_adapters
    else:
        all_adapters = _setup_lora(model, args)
        critical_adapters = all_adapters
        funlora_adapters = []

    # ----- Training loop -----
    accuracy_matrix = np.zeros((args.num_tasks, args.num_tasks))
    start_time = time.time()

    for t in range(args.num_tasks):
        print(f"\n--- Task {t} / {args.num_tasks - 1} ---")
        _train_task(model, t, args, critical_adapters, funlora_adapters, tokenizer, device)
        _evaluate(model, t, args, accuracy_matrix, tokenizer, device)

    elapsed = time.time() - start_time

    # ----- Metrics -----
    acc = average_accuracy(accuracy_matrix)
    bwt = backward_transfer(accuracy_matrix) if args.num_tasks >= 2 else float("nan")

    print(f"\n{full_report(accuracy_matrix, args.method)}")
    print(f"\nTotal time: {elapsed:.1f}s")

    return {
        "method": args.method,
        "model": args.model,
        "seed": seed,
        "num_tasks": args.num_tasks,
        "epochs": args.epochs,
        "rank": args.rank,
        "acc": acc,
        "bwt": bwt,
        "elapsed_s": elapsed,
        "accuracy_matrix": accuracy_matrix.tolist(),
        "ablation": {
            "paca": args.paca,
            "deal": args.deal,
            "gate": args.gate,
            "coso": args.coso,
            "ear": args.ear,
            "svc": args.svc,
            "dmole": args.dmole,
            "funlora": args.funlora,
        },
        "timestamp": datetime.utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# Adapter setup helpers
# ---------------------------------------------------------------------------

def _setup_cascades(model, args, device, tokenizer):
    """Apply ablation flags to v5 module then inject adapters."""
    import cascades_exp.hf_cascades_v5 as v5

    v5.ENABLE_PACA            = args.paca
    v5.ENABLE_DEAL            = args.deal
    v5.ENABLE_GAINLORA_GATE   = args.gate
    v5.ENABLE_COSO_NULLSPACE  = args.coso
    v5.ENABLE_CLLORA_REASSIGN = args.ear
    v5.ENABLE_SVC             = args.svc
    v5.ENABLE_DMOLE_SELECT    = args.dmole
    v5.ENABLE_FUNLORA         = args.funlora

    quant_noise = v5.estimate_quant_noise(model)
    layer_importance = None
    if args.dmole:
        probe_loader = prepare_dataloader(tokenizer, 0, batch_size=args.batch_size)
        layer_importance = v5.compute_layer_importance(model, probe_loader, device)

    critical, funlora = v5.inject_cascades_v3(
        model, rank=args.rank, layer_importance=layer_importance
    )
    for a in critical:
        a.quant_noise_std.fill_(quant_noise)
    return critical, funlora


def _setup_lora(model, args):
    """Inject standard LoRA adapters (no continual-learning mechanisms)."""
    import cascades_exp.lora_baseline as lb
    return lb.inject_lora(model, rank=args.rank)


# ---------------------------------------------------------------------------
# Per-task training and evaluation
# ---------------------------------------------------------------------------

def _train_task(model, t, args, critical_adapters, funlora_adapters, tokenizer, device):
    import torch.optim as optim
    import cascades_exp.hf_cascades_v5 as v5

    if args.method == "cascades_v5":
        for a in critical_adapters:
            a.add_task(t)
        for name, module in model.named_modules():
            if isinstance(module, v5.CASCADES_v3_Linear):
                module.current_task_id = t

    dataloader = prepare_dataloader(tokenizer, t, batch_size=args.batch_size)

    if args.method == "cascades_v5":
        param_groups = [
            {"params": [a.U_shared for a in critical_adapters] + [a.V_shared for a in critical_adapters], "lr": args.lr_shared},
            {"params": [a.task_lambdas[str(t)] for a in critical_adapters if str(t) in a.task_lambdas], "lr": args.lr_task},
        ]
        if args.gate:
            gate_params = [p for a in critical_adapters if hasattr(a, "gate_proj") for p in a.gate_proj.parameters()]
            task_emb_params = [p for a in critical_adapters if hasattr(a, "task_embedding") for p in a.task_embedding.parameters()]
            if gate_params:
                param_groups.append({"params": gate_params + task_emb_params, "lr": args.lr_gate})
        funlora_params = [p for a in funlora_adapters for p in a.parameters()]
        if funlora_params:
            param_groups.append({"params": funlora_params, "lr": args.lr_shared})
    else:
        param_groups = [{"params": [p for a in critical_adapters for p in a.parameters()], "lr": args.lr_shared}]

    optimizer = optim.Adam(param_groups)

    for ep in range(args.epochs):
        epoch_loss, n = 0.0, 0
        for input_ids, attention_mask in dataloader:
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            optimizer.zero_grad()
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids).loss
            loss.backward()

            trainable = [p for p in model.parameters() if p.grad is not None]
            if trainable:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)

            if args.method == "cascades_v5":
                import cascades_exp.hf_cascades_v5 as v5
                for a in critical_adapters:
                    a.full_descent_step(lr=0.01)

            optimizer.step()
            epoch_loss += loss.item()
            n += 1

        print(f"  Epoch {ep+1}/{args.epochs}  loss={epoch_loss/max(n,1):.4f}")

    if args.method == "cascades_v5":
        import cascades_exp.hf_cascades_v5 as v5
        for a in critical_adapters:
            a.store_task_gradients()
            a.update_null_space_sketch()


def _evaluate(model, trained_up_to, args, accuracy_matrix, tokenizer, device):
    import cascades_exp.hf_cascades_v5 as v5

    with torch.no_grad():
        for eval_t in range(trained_up_to + 1):
            if args.method == "cascades_v5":
                for name, module in model.named_modules():
                    if isinstance(module, v5.CASCADES_v3_Linear):
                        module.current_task_id = eval_t

            loader = prepare_dataloader(tokenizer, eval_t, batch_size=args.batch_size)
            total_loss, n = 0.0, 0
            for input_ids, attention_mask in loader:
                input_ids = input_ids.to(device)
                total_loss += model(input_ids=input_ids, labels=input_ids).loss.item()
                n += 1

            proxy = proxy_accuracy_from_loss(total_loss / max(n, 1))
            accuracy_matrix[trained_up_to, eval_t] = proxy
            print(f"  Task {eval_t} proxy acc: {proxy*100:.2f}%")


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def save_result(result: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = result["timestamp"].replace(":", "-").replace(".", "-")
    fname = f"{ts}_{result['method']}_seed{result['seed']}.json"
    path = output_dir / fname
    path.write_text(json.dumps(result, indent=2))
    return path


def append_summary_csv(results: list[dict], summary_path: Path):
    import csv
    fieldnames = ["method", "model", "seed", "acc", "bwt", "elapsed_s", "rank",
                  "epochs", "num_tasks", "timestamp"]
    write_header = not summary_path.exists()
    with summary_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in fieldnames})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    args.output_dir = Path(args.output_dir)

    all_results = []
    for seed in args.seeds:
        result = run_one_seed(args, seed)
        path = save_result(result, args.output_dir)
        print(f"\nResult saved → {path}")
        all_results.append(result)

    # Aggregate across seeds
    accs = [r["acc"] for r in all_results]
    bwts = [r["bwt"] for r in all_results]

    print(f"\n{'='*60}")
    print(f"  AGGREGATE over {len(args.seeds)} seed(s)")
    print(f"  ACC: {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
    if not any(math.isnan(b) for b in bwts):
        print(f"  BWT: {np.mean(bwts)*100:.2f}% ± {np.std(bwts)*100:.2f}%")
    print(f"{'='*60}")

    summary_path = ROOT / "results" / "summary.csv"
    append_summary_csv(all_results, summary_path)
    print(f"Summary appended → {summary_path}")


if __name__ == "__main__":
    main()
