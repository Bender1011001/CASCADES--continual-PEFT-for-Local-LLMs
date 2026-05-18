from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CONTROL_DIR = ROOT / "experiments/cf_cycle_3/nullspace_ablation_retry/control"
VRAM_THRESHOLD_MB = 7500.0
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


def load_json(name: str) -> Any:
    return json.loads((CONTROL_DIR / name).read_text(encoding="utf-8"))


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


def main() -> None:
    missing = [name for name in REQUIRED_ARTIFACTS if not (CONTROL_DIR / name).exists()]
    if missing:
        print(json.dumps({"valid": False, "missing_artifacts": missing}, indent=2))
        raise SystemExit(1)

    status = load_json("run_status.json")
    metrics = load_json("metrics.json")
    instrumentation = load_json("instrumentation.json")
    manifest = load_json("task_manifest.json")
    config = load_json("config.json")
    matrix = np.load(CONTROL_DIR / "accuracy_matrix.npy")

    manifest_paths = [row.get("path") for row in manifest]
    param_expectations = {
        "task_suite": "reasoning3",
        "seed": 42,
        "epochs": 2,
        "rank": 4,
        "max_length": 256,
    }
    param_mismatches = {
        key: {"expected": expected, "observed": metrics.get(key)}
        for key, expected in param_expectations.items()
        if metrics.get(key) != expected
    }
    status_mismatches = {
        key: {"expected": expected, "observed": status.get(key)}
        for key, expected in param_expectations.items()
        if status.get(key) != expected
    }

    finite_metric_bad_paths = finite_bad_paths(metrics, "metrics")
    finite_instrumentation_bad_paths = finite_bad_paths(instrumentation, "instrumentation")
    finite_matrix = bool(np.isfinite(matrix).all())
    peak_vram_mb = float(status.get("peak_vram_mb", math.inf))

    valid = (
        status.get("status") == "completed"
        and not param_mismatches
        and not status_mismatches
        and manifest_paths == EXPECTED_TASKS
        and peak_vram_mb <= VRAM_THRESHOLD_MB
        and not finite_metric_bad_paths
        and not finite_instrumentation_bad_paths
        and finite_matrix
    )

    payload = {
        "valid": valid,
        "control_dir": str(CONTROL_DIR.relative_to(ROOT)).replace("\\", "/"),
        "required_artifacts_present": True,
        "status": status,
        "metrics_parameters": {key: metrics.get(key) for key in param_expectations},
        "status_parameter_mismatches": status_mismatches,
        "metrics_parameter_mismatches": param_mismatches,
        "manifest_paths": manifest_paths,
        "expected_manifest_paths": EXPECTED_TASKS,
        "peak_vram_mb": peak_vram_mb,
        "vram_threshold_mb": VRAM_THRESHOLD_MB,
        "under_vram_threshold": peak_vram_mb <= VRAM_THRESHOLD_MB,
        "finite_metric_bad_paths": finite_metric_bad_paths,
        "finite_instrumentation_bad_paths": finite_instrumentation_bad_paths,
        "finite_accuracy_matrix": finite_matrix,
        "accuracy_matrix_shape": list(matrix.shape),
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
    print(json.dumps(payload, indent=2))
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    sys.exit(main())
