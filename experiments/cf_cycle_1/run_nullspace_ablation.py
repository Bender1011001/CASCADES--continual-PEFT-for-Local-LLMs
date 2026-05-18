from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

import cascades.data as data
from cascades.config import AblationConfig
from cascades.metrics import average_accuracy, backward_transfer


CURRENT4 = [
    "data/task2_csqa_cot.jsonl",
    "data/task1_arc_cot.jsonl",
    "data/task0_gsm8k_cot.jsonl",
    "data/task3_digital_twin.jsonl",
]
REASONING3 = [
    "data/task0_gsm8k_cot.jsonl",
    "data/task1_arc_cot.jsonl",
    "data/task2_csqa_cot.jsonl",
]


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        return f"unavailable: {exc}"


def count_examples(path: str) -> int | None:
    p = ROOT / path
    if not p.exists():
        return None
    return sum(1 for line in p.open("r", encoding="utf-8") if line.strip())


def task_manifest(files: list[str]) -> list[dict]:
    rows = []
    for i, file_name in enumerate(files):
        p = ROOT / file_name
        rows.append(
            {
                "task_index": i,
                "path": file_name,
                "exists": p.exists(),
                "examples": count_examples(file_name),
                "bytes": p.stat().st_size if p.exists() else None,
            }
        )
    return rows


def apply_task_suite(task_suite: str) -> list[str]:
    files = CURRENT4 if task_suite == "current4" else REASONING3
    data.TASK_FILES = list(files)
    data.TASK_NAMES = {i: f"Task {i} ({Path(path).stem})" for i, path in enumerate(files)}
    data.NUM_TASKS = len(files)
    return files


def config_for_arm(arm: str, treatment_variant: str = "frozen-only") -> AblationConfig:
    if treatment_variant not in {
        "frozen-only",
        "cllora-active",
        "cllora-active-freeze-thresh-02",
        "cllora-active-freeze-topk-2",
    }:
        raise ValueError(f"unsupported treatment_variant: {treatment_variant!r}")

    enable_coso_nullspace = arm == "treatment"
    enable_cllora_reassign = False
    enable_soft_ear = True
    ear_gamma = 1e-4
    frozen_basis_variance_threshold = 0.05
    frozen_basis_top_k_per_freeze = None

    if arm == "treatment" and treatment_variant in {
        "cllora-active",
        "cllora-active-freeze-thresh-02",
        "cllora-active-freeze-topk-2",
    }:
        enable_coso_nullspace = True
        enable_cllora_reassign = True
        enable_soft_ear = True
        ear_gamma = 1e-4
        if treatment_variant == "cllora-active-freeze-thresh-02":
            frozen_basis_variance_threshold = 0.02
        if treatment_variant == "cllora-active-freeze-topk-2":
            frozen_basis_top_k_per_freeze = 2

    return AblationConfig(
        enable_paca=True,
        enable_deal=True,
        enable_gainlora_gate=True,
        enable_coso_nullspace=enable_coso_nullspace,
        enable_cllora_reassign=enable_cllora_reassign,
        enable_svc=True,
        enable_dmole_select=True,
        enable_funlora=True,
        gqa_ratio=1.0,
        ear_gamma=ear_gamma,
        enable_soft_ear=enable_soft_ear,
        enable_principal_expansion=True,
        cfg_lambda=1.5,
        enable_ambient_dedup=True,
        frozen_basis_variance_threshold=frozen_basis_variance_threshold,
        frozen_basis_top_k_per_freeze=frozen_basis_top_k_per_freeze,
    )


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def output_root_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def max_observed_vram_mb(stats: dict) -> float:
    peaks = []
    for event in stats.get("vram", []):
        if not isinstance(event, dict):
            continue
        peak = event.get("max_allocated_mb", event.get("peak_mb", 0.0))
        try:
            peaks.append(float(peak))
        except (TypeError, ValueError):
            continue
    if torch.cuda.is_available():
        peaks.append(torch.cuda.max_memory_allocated() / (1024**2))
    return max(peaks, default=0.0)


def run_status_payload(
    args: argparse.Namespace,
    arm: str,
    status: str,
    **extra,
) -> dict:
    payload = {
        "arm": arm,
        "status": status,
        "task_suite": args.task_suite,
        "seed": args.seed,
        "rank": args.rank,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "vram_threshold_mb": args.vram_threshold_mb,
        "allow_nonfinite": args.allow_nonfinite,
        "allow_vram_over_threshold": args.allow_vram_over_threshold,
        "treatment_variant": args.treatment_variant,
        "git_revision": git_revision(),
    }
    payload.update(extra)
    return payload


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def install_instrumentation(train_module) -> tuple[dict, Callable[[], None]]:
    import cascades.adapters as adapters_mod

    stats = {
        "ear_events": [],
        "freeze_events": [],
        "reassign": {
            "calls_total": 0,
            "calls_with_null_sketch": 0,
            "calls_with_frozen_basis": 0,
            "max_frozen_cols": 0,
            "removed_norm_sum": 0.0,
            "removed_norm_per_frozen_call": 0.0,
            "calls_with_active_reassign_enabled": 0,
            "calls_with_active_reassign_path": 0,
            "active_adjustment_norm_sum": 0.0,
            "active_adjustment_norm_max": 0.0,
        },
        "vram": [],
    }

    orig_reassign = adapters_mod._cllora_reassign
    orig_freeze = adapters_mod.CASCADESAdapter.freeze_current_subspace
    orig_batched_null = train_module.batched_null_space_extraction
    orig_log_vram = train_module.log_vram

    def frozen_only_projection(grad, frozen_basis):
        frozen_cols = int(frozen_basis.shape[1]) if frozen_basis is not None else 0
        frozen_only = grad.detach().clone()
        if frozen_cols <= 0:
            return frozen_only
        with torch.no_grad():
            u_f, s_f, _ = torch.linalg.svd(frozen_basis, full_matrices=False)
            q_f = u_f[:, s_f > 1e-6]
            if q_f.shape[1] > 0:
                frozen_only = frozen_only - q_f @ (q_f.T @ frozen_only)
        return frozen_only

    def wrapped_reassign(grad, null_sketch, config, frozen_basis=None):
        frozen_cols = int(frozen_basis.shape[1]) if frozen_basis is not None else 0
        reassign_stats = stats["reassign"]
        reassign_stats["calls_total"] += 1
        if null_sketch is not None:
            reassign_stats["calls_with_null_sketch"] += 1
        before = grad.detach().clone()
        frozen_only = frozen_only_projection(grad, frozen_basis)
        active_enabled = bool(getattr(config, "enable_cllora_reassign", False))
        active_path = active_enabled and null_sketch is not None
        if active_enabled:
            reassign_stats["calls_with_active_reassign_enabled"] += 1
        if active_path:
            reassign_stats["calls_with_active_reassign_path"] += 1
        out = orig_reassign(grad, null_sketch, config, frozen_basis=frozen_basis)
        if frozen_cols > 0:
            reassign_stats["calls_with_frozen_basis"] += 1
            reassign_stats["max_frozen_cols"] = max(
                reassign_stats["max_frozen_cols"], frozen_cols
            )
            reassign_stats["removed_norm_sum"] += float((before - frozen_only).norm().item())
            reassign_stats["removed_norm_per_frozen_call"] = (
                reassign_stats["removed_norm_sum"] / reassign_stats["calls_with_frozen_basis"]
            )
        if active_path:
            active_adjustment_norm = float((frozen_only - out.detach()).norm().item())
            reassign_stats["active_adjustment_norm_sum"] += active_adjustment_norm
            reassign_stats["active_adjustment_norm_max"] = max(
                reassign_stats["active_adjustment_norm_max"], active_adjustment_norm
            )
        return out

    def wrapped_freeze(self):
        u_before = int(getattr(self, "frozen_null_basis").shape[1])
        v_before = int(getattr(self, "frozen_null_basis_V").shape[1])
        rank = int(self.U_shared.shape[1])
        config = getattr(self, "config", None)
        orig_freeze(self)
        u_after = int(getattr(self, "frozen_null_basis").shape[1])
        v_after = int(getattr(self, "frozen_null_basis_V").shape[1])
        stats["freeze_events"].append(
            {
                "task_boundary_call": len(stats["freeze_events"]) + 1,
                "u_cols_before": u_before,
                "u_cols_after": u_after,
                "u_cols_added": u_after - u_before,
                "v_cols_before": v_before,
                "v_cols_after": v_after,
                "v_cols_added": v_after - v_before,
                "adapter_rank": rank,
                "out_features": int(self.U_shared.shape[0]),
                "in_features": int(self.V_shared.shape[1]),
                "frozen_basis_variance_threshold": getattr(
                    config, "frozen_basis_variance_threshold", None
                ),
                "frozen_basis_top_k_per_freeze": getattr(
                    config, "frozen_basis_top_k_per_freeze", None
                ),
            }
        )

    def wrapped_batched_null(adapters):
        before = sum(1 for adapter in adapters if getattr(adapter, "ear_initialized", False))
        orig_batched_null(adapters)
        after = sum(1 for adapter in adapters if getattr(adapter, "ear_initialized", False))
        stats["ear_events"].append(
            {
                "call_index": len(stats["ear_events"]) + 1,
                "adapters": len(adapters),
                "ear_initialized_before": before,
                "ear_initialized_after": after,
                "max_q_null_u_cols": max((int(a.Q_null_U.shape[1]) for a in adapters), default=0),
                "max_q_null_v_cols": max((int(a.Q_null_V.shape[1]) for a in adapters), default=0),
            }
        )

    def wrapped_log_vram(tag, device):
        vram_stats = orig_log_vram(tag, device)
        stats["vram"].append(
            {
                "tag": tag,
                "allocated_mb": float(vram_stats.get("allocated_mb", 0.0)),
                "reserved_mb": float(vram_stats.get("reserved_mb", 0.0)),
                "max_allocated_mb": float(vram_stats.get("peak_mb", 0.0)),
            }
        )
        return vram_stats

    adapters_mod._cllora_reassign = wrapped_reassign
    adapters_mod.CASCADESAdapter.freeze_current_subspace = wrapped_freeze
    train_module.batched_null_space_extraction = wrapped_batched_null
    train_module.log_vram = wrapped_log_vram

    def restore() -> None:
        adapters_mod._cllora_reassign = orig_reassign
        adapters_mod.CASCADESAdapter.freeze_current_subspace = orig_freeze
        train_module.batched_null_space_extraction = orig_batched_null
        train_module.log_vram = orig_log_vram

    return stats, restore


def run_arm(args: argparse.Namespace, arm: str) -> None:
    files = apply_task_suite(args.task_suite)
    import train

    train.NUM_TASKS = len(files)
    cfg = config_for_arm(arm, treatment_variant=args.treatment_variant)
    out_dir = output_root_path(args.output_root) / arm
    out_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        out_dir / "config.json",
        {
            "arm": arm,
            "task_suite": args.task_suite,
            "model_id": args.model_id,
            "seed": args.seed,
            "epochs": args.epochs,
            "rank": args.rank,
            "max_length": args.max_length,
            "eval_em": args.eval_em,
            "enable_sleep": not args.no_sleep,
            "vram_threshold_mb": args.vram_threshold_mb,
            "allow_nonfinite": args.allow_nonfinite,
            "allow_vram_over_threshold": args.allow_vram_over_threshold,
            "treatment_variant": args.treatment_variant,
            "ablation_config": asdict(cfg),
            "git_revision": git_revision(),
        },
    )
    write_json(out_dir / "task_manifest.json", task_manifest(files))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    stats, restore = install_instrumentation(train)
    start = time.time()
    guardrail_failure = None
    try:
        try:
            matrix = train.train_cascades(
                seed=args.seed,
                model_id=args.model_id,
                output_prefix=str(out_dir / "cascades"),
                epochs=args.epochs,
                rank=args.rank,
                max_length=args.max_length,
                eval_em=args.eval_em,
                enable_sleep=not args.no_sleep,
                config=cfg,
                abort_on_nonfinite=not args.allow_nonfinite,
                vram_threshold_mb=(
                    None if args.allow_vram_over_threshold else args.vram_threshold_mb
                ),
            )
        except train.TrainingGuardrailViolation as exc:
            guardrail_failure = exc
    finally:
        restore()

    wall_time = time.time() - start
    peak_vram_mb = max_observed_vram_mb(stats)

    if guardrail_failure is not None:
        write_json(out_dir / "instrumentation.json", stats)
        write_json(
            out_dir / "run_status.json",
            run_status_payload(
                args,
                arm,
                "failed_guardrail",
                reason=str(guardrail_failure),
                peak_vram_mb=peak_vram_mb,
                wall_time_s=wall_time,
            ),
        )
        cleanup_cuda()
        raise SystemExit(1) from guardrail_failure

    np.save(out_dir / "accuracy_matrix.npy", matrix)

    diagonal = [float(matrix[i, i]) for i in range(matrix.shape[0])]
    final = [float(x) for x in matrix[-1, :]]
    old_task_deltas = [float(matrix[-1, i] - matrix[i, i]) for i in range(matrix.shape[0] - 1)]
    metrics = {
        "arm": arm,
        "task_suite": args.task_suite,
        "seed": args.seed,
        "rank": args.rank,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "avg_acc": average_accuracy(matrix),
        "bwt": backward_transfer(matrix) if matrix.shape[0] >= 2 else 0.0,
        "final_accs": final,
        "diagonal_accs": diagonal,
        "old_task_deltas": old_task_deltas,
        "peak_vram_mb": peak_vram_mb,
        "wall_time_s": wall_time,
        "git_revision": git_revision(),
    }
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "instrumentation.json", stats)
    write_json(
        out_dir / "run_status.json",
        run_status_payload(
            args,
            arm,
            "completed",
            peak_vram_mb=peak_vram_mb,
            wall_time_s=wall_time,
        ),
    )

    cleanup_cuda()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CF-cycle-1 null-space ablation runner")
    parser.add_argument("--arm", choices=["control", "treatment", "both"], required=True)
    parser.add_argument("--task-suite", choices=["current4", "reasoning3"], default="current4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--model-id", default="p-e-w/Qwen3-4B-Instruct-2507-heretic")
    parser.add_argument("--output-root", default="experiments/cf_cycle_1/nullspace_ablation")
    parser.add_argument("--eval-em", action="store_true")
    parser.add_argument("--no-sleep", action="store_true")
    parser.add_argument("--vram-threshold-mb", type=float, default=7500.0)
    parser.add_argument("--allow-nonfinite", action="store_true")
    parser.add_argument("--allow-vram-over-threshold", action="store_true")
    parser.add_argument(
        "--treatment-variant",
        choices=[
            "frozen-only",
            "cllora-active",
            "cllora-active-freeze-thresh-02",
            "cllora-active-freeze-topk-2",
        ],
        default="frozen-only",
        help="Treatment ablation variant; control arm remains unchanged.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arms = ["control", "treatment"] if args.arm == "both" else [args.arm]
    for arm in arms:
        run_arm(args, arm)


if __name__ == "__main__":
    main()
