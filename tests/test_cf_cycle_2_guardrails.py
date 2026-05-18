from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.cf_cycle_1 import compare_nullspace_ablation as compare


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def finite_metrics(arm: str, peak: float = 7000.0) -> dict:
    return {
        "arm": arm,
        "task_suite": "reasoning3",
        "seed": 42,
        "rank": 4,
        "max_length": 256,
        "epochs": 2,
        "avg_acc": 0.25,
        "bwt": -0.01,
        "final_accs": [0.2, 0.25, 0.3],
        "diagonal_accs": [0.21, 0.27, 0.31],
        "old_task_deltas": [-0.01, -0.02],
        "peak_vram_mb": peak,
        "wall_time_s": 1.0,
        "git_revision": "test",
    }


def active_instrumentation() -> dict:
    return {
        "ear_events": [],
        "freeze_events": [{"u_cols_after": 1, "v_cols_after": 0}],
        "reassign": {
            "calls_total": 1,
            "calls_with_null_sketch": 0,
            "calls_with_frozen_basis": 1,
            "max_frozen_cols": 1,
            "removed_norm_sum": 0.5,
        },
        "vram": [],
    }


def completed_status(arm: str) -> dict:
    return {
        "arm": arm,
        "status": "completed",
        "task_suite": "reasoning3",
        "seed": 42,
        "rank": 4,
        "max_length": 256,
        "vram_threshold_mb": 7500.0,
    }


def populate_valid_run(
    root: Path,
    control_peak: float = 7000.0,
    treatment_peak: float = 7000.0,
) -> None:
    write_json(root / "control" / "metrics.json", finite_metrics("control", control_peak))
    write_json(root / "control" / "instrumentation.json", active_instrumentation())
    write_json(root / "control" / "run_status.json", completed_status("control"))
    write_json(root / "treatment" / "metrics.json", finite_metrics("treatment", treatment_peak))
    write_json(root / "treatment" / "instrumentation.json", active_instrumentation())
    write_json(root / "treatment" / "run_status.json", completed_status("treatment"))


def test_comparison_accepts_valid_runs(tmp_path: Path) -> None:
    populate_valid_run(tmp_path)

    result = compare.compare_runs(tmp_path, vram_threshold_mb=7500.0, fail_invalid=True)

    assert result["valid"] is True
    assert result["control_peak_vram_mb"] == 7000.0
    assert result["treatment_peak_vram_mb"] == 7000.0


def test_comparison_rejects_nonfinite_control_metric(tmp_path: Path) -> None:
    populate_valid_run(tmp_path)
    metrics = finite_metrics("control")
    metrics["avg_acc"] = float("nan")
    write_json(tmp_path / "control" / "metrics.json", metrics)

    with pytest.raises(SystemExit) as excinfo:
        compare.compare_runs(tmp_path, vram_threshold_mb=7500.0, fail_invalid=True)

    assert excinfo.value.code == 2


def test_comparison_rejects_control_vram_over_threshold(tmp_path: Path) -> None:
    populate_valid_run(tmp_path, control_peak=7600.0)

    with pytest.raises(SystemExit) as excinfo:
        compare.compare_runs(tmp_path, vram_threshold_mb=7500.0, fail_invalid=True)

    assert excinfo.value.code == 2


def test_comparison_rejects_failed_run_status(tmp_path: Path) -> None:
    populate_valid_run(tmp_path)
    status = completed_status("control")
    status["status"] = "failed_guardrail"
    status["reason"] = "non-finite training loss task=2 epoch=1 batch=37"
    write_json(tmp_path / "control" / "run_status.json", status)

    with pytest.raises(SystemExit) as excinfo:
        compare.compare_runs(tmp_path, vram_threshold_mb=7500.0, fail_invalid=True)

    assert excinfo.value.code == 2


def test_comparison_rejects_inactive_projection(tmp_path: Path) -> None:
    populate_valid_run(tmp_path)
    instr = active_instrumentation()
    instr["reassign"]["calls_with_frozen_basis"] = 0
    instr["reassign"]["removed_norm_sum"] = 0.0
    write_json(tmp_path / "treatment" / "instrumentation.json", instr)

    with pytest.raises(SystemExit) as excinfo:
        compare.compare_runs(tmp_path, vram_threshold_mb=7500.0, fail_invalid=True)

    assert excinfo.value.code == 2
