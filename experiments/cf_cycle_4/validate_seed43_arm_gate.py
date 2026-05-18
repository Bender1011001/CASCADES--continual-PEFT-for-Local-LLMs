from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_TASKS = [
    "data/task0_gsm8k_cot.jsonl",
    "data/task1_arc_cot.jsonl",
    "data/task2_csqa_cot.jsonl",
]
REQUIRED_ARTIFACTS = [
    "config.json",
    "task_manifest.json",
    "accuracy_matrix.npy",
    "cascades_results.csv",
    "metrics.json",
    "instrumentation.json",
    "run_status.json",
]
PARAM_EXPECTATIONS = {
    "task_suite": "reasoning3",
    "seed": 43,
    "epochs": 2,
    "rank": 4,
    "max_length": 256,
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_bad_paths(obj: Any, prefix: str) -> list[str]:
    if isinstance(obj, dict):
        bad: list[str] = []
        for key, value in obj.items():
            bad.extend(finite_bad_paths(value, f"{prefix}.{key}"))
        return bad
    if isinstance(obj, list):
        bad = []
        for index, value in enumerate(obj):
            bad.extend(finite_bad_paths(value, f"{prefix}[{index}]"))
        return bad
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        return [] if math.isfinite(float(obj)) else [prefix]
    return []


def projection_summary(instrumentation: dict[str, Any]) -> dict[str, Any]:
    freeze_events = instrumentation.get("freeze_events", [])
    reassign = instrumentation.get("reassign", {})
    frozen_basis_nonempty = any(
        isinstance(event, dict)
        and (event.get("u_cols_after", 0) > 0 or event.get("v_cols_after", 0) > 0)
        for event in freeze_events
    )
    calls_with_frozen_basis = int(reassign.get("calls_with_frozen_basis", 0) or 0)
    removed_norm_sum = float(reassign.get("removed_norm_sum", 0.0) or 0.0)
    return {
        "frozen_basis_nonempty": frozen_basis_nonempty,
        "calls_with_frozen_basis": calls_with_frozen_basis,
        "removed_norm_sum": removed_norm_sum,
        "projection_active": calls_with_frozen_basis > 0 and removed_norm_sum > 0.0,
        "freeze_event_count": len(freeze_events) if isinstance(freeze_events, list) else None,
    }


def validate_arm(root: Path, arm: str, vram_threshold_mb: float) -> dict[str, Any]:
    arm_dir = root / arm
    missing = [name for name in REQUIRED_ARTIFACTS if not (arm_dir / name).exists()]
    if missing:
        return {
            "valid": False,
            "arm": arm,
            "arm_dir": str(arm_dir.relative_to(ROOT)).replace("\\", "/"),
            "missing_artifacts": missing,
        }

    status = load_json(arm_dir / "run_status.json")
    metrics = load_json(arm_dir / "metrics.json")
    instrumentation = load_json(arm_dir / "instrumentation.json")
    manifest = load_json(arm_dir / "task_manifest.json")
    config = load_json(arm_dir / "config.json")
    matrix = np.load(arm_dir / "accuracy_matrix.npy")

    manifest_paths = [row.get("path") for row in manifest]
    metrics_parameter_mismatches = {
        key: {"expected": expected, "observed": metrics.get(key)}
        for key, expected in PARAM_EXPECTATIONS.items()
        if metrics.get(key) != expected
    }
    status_parameter_mismatches = {
        key: {"expected": expected, "observed": status.get(key)}
        for key, expected in PARAM_EXPECTATIONS.items()
        if status.get(key) != expected
    }
    config_parameter_mismatches = {
        key: {"expected": expected, "observed": config.get(key)}
        for key, expected in PARAM_EXPECTATIONS.items()
        if config.get(key) != expected
    }

    finite_metric_bad_paths = finite_bad_paths(metrics, "metrics")
    finite_instrumentation_bad_paths = finite_bad_paths(instrumentation, "instrumentation")
    finite_accuracy_matrix = bool(np.isfinite(matrix).all())
    peak_vram_mb = float(status.get("peak_vram_mb", math.inf))
    projection = projection_summary(instrumentation)

    base_valid = (
        status.get("status") == "completed"
        and not metrics_parameter_mismatches
        and not status_parameter_mismatches
        and not config_parameter_mismatches
        and manifest_paths == EXPECTED_TASKS
        and math.isfinite(peak_vram_mb)
        and peak_vram_mb <= vram_threshold_mb
        and not finite_metric_bad_paths
        and not finite_instrumentation_bad_paths
        and finite_accuracy_matrix
    )
    treatment_projection_valid = True
    if arm == "treatment":
        treatment_projection_valid = bool(
            projection["projection_active"] and projection["frozen_basis_nonempty"]
        )

    return {
        "valid": bool(base_valid and treatment_projection_valid),
        "arm": arm,
        "arm_dir": str(arm_dir.relative_to(ROOT)).replace("\\", "/"),
        "required_artifacts_present": True,
        "status": status,
        "metrics_parameters": {key: metrics.get(key) for key in PARAM_EXPECTATIONS},
        "status_parameter_mismatches": status_parameter_mismatches,
        "metrics_parameter_mismatches": metrics_parameter_mismatches,
        "config_parameter_mismatches": config_parameter_mismatches,
        "manifest_paths": manifest_paths,
        "expected_manifest_paths": EXPECTED_TASKS,
        "peak_vram_mb": peak_vram_mb,
        "vram_threshold_mb": vram_threshold_mb,
        "under_vram_threshold": peak_vram_mb <= vram_threshold_mb,
        "finite_metric_bad_paths": finite_metric_bad_paths,
        "finite_instrumentation_bad_paths": finite_instrumentation_bad_paths,
        "finite_accuracy_matrix": finite_accuracy_matrix,
        "accuracy_matrix_shape": list(matrix.shape),
        "projection": projection,
        "treatment_projection_required": arm == "treatment",
        "treatment_projection_valid": treatment_projection_valid,
        "config_summary": {
            key: config.get(key)
            for key in [
                "arm",
                "task_suite",
                "seed",
                "epochs",
                "rank",
                "max_length",
                "vram_threshold_mb",
                "allow_nonfinite",
                "allow_vram_over_threshold",
            ]
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate CF-cycle-4 seed-43 arm gate")
    parser.add_argument("--root", default="experiments/cf_cycle_4/nullspace_ablation_seed43")
    parser.add_argument("--arm", choices=["control", "treatment"], required=True)
    parser.add_argument("--vram-threshold-mb", type=float, default=7500.0)
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = ROOT / root
    payload = validate_arm(root, args.arm, args.vram_threshold_mb)
    print(json.dumps(payload, indent=2, allow_nan=False))
    return 0 if payload.get("valid") else 1


if __name__ == "__main__":
    sys.exit(main())

