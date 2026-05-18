from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ARMS = ("control", "treatment")
REQUIRED_ARTIFACTS = ("metrics.json", "instrumentation.json", "run_status.json")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_finite_json_number(value: Any) -> bool:
    return not isinstance(value, (int, float)) or math.isfinite(value)


def _collect_nonfinite_numbers(value: Any, path: str) -> list[str]:
    failures: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            failures.extend(_collect_nonfinite_numbers(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            failures.extend(_collect_nonfinite_numbers(item, f"{path}[{index}]"))
    elif not _is_finite_json_number(value):
        failures.append(f"non-finite numeric value at {path}: {value!r}")
    return failures


def _load_arm(root: Path, arm: str, failures: list[str]) -> dict[str, dict[str, Any]] | None:
    arm_dir = root / arm
    payloads: dict[str, dict[str, Any]] = {}
    for filename in REQUIRED_ARTIFACTS:
        path = arm_dir / filename
        if not path.exists():
            failures.append(f"{arm} missing required artifact: {filename}")
            continue
        try:
            payload = load_json(path)
        except json.JSONDecodeError as exc:
            failures.append(f"{arm} invalid JSON in {filename}: {exc}")
            continue
        if not isinstance(payload, dict):
            failures.append(f"{arm} artifact {filename} must contain a JSON object")
            continue
        payloads[filename] = payload

    if len(payloads) != len(REQUIRED_ARTIFACTS):
        return None

    status = payloads["run_status.json"].get("status")
    if status != "completed":
        reason = payloads["run_status.json"].get("reason", "no reason recorded")
        failures.append(f"{arm} run_status is {status!r}, not 'completed': {reason}")

    failures.extend(_collect_nonfinite_numbers(payloads["metrics.json"], f"{arm}.metrics"))
    failures.extend(
        _collect_nonfinite_numbers(payloads["instrumentation.json"], f"{arm}.instrumentation")
    )
    return payloads


def _peak_vram(metrics: dict[str, Any], arm: str, failures: list[str]) -> float | None:
    peak = metrics.get("peak_vram_mb")
    if not isinstance(peak, (int, float)) or not math.isfinite(peak):
        failures.append(f"{arm} peak_vram_mb missing or non-finite")
        return None
    return float(peak)


def _projection_active(instrumentation: dict[str, Any], failures: list[str]) -> tuple[bool, bool]:
    reassign = instrumentation.get("reassign", {})
    freeze_events = instrumentation.get("freeze_events", [])
    calls_with_frozen_basis = reassign.get("calls_with_frozen_basis", 0)
    removed_norm_sum = reassign.get("removed_norm_sum", 0.0)
    projection_active = calls_with_frozen_basis > 0 and removed_norm_sum > 0.0
    frozen_basis_nonempty = any(
        event.get("u_cols_after", 0) > 0 or event.get("v_cols_after", 0) > 0
        for event in freeze_events
        if isinstance(event, dict)
    )
    if not projection_active:
        failures.append(
            "treatment projection inactive: requires calls_with_frozen_basis > 0 "
            "and removed_norm_sum > 0.0"
        )
    if not frozen_basis_nonempty:
        failures.append("treatment projection inactive: no non-empty frozen basis evidence")
    return projection_active, frozen_basis_nonempty


def _invalid_result(failures: list[str], fail_invalid: bool) -> dict[str, Any]:
    result = {"valid": False, "failures": failures}
    if fail_invalid:
        print(json.dumps(result, indent=2, allow_nan=False))
        raise SystemExit(2)
    return result


def compare_runs(
    root: Path,
    vram_threshold_mb: float = 7500.0,
    fail_invalid: bool = True,
) -> dict[str, Any]:
    failures: list[str] = []
    loaded: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in ARMS:
        arm_payloads = _load_arm(root, arm, failures)
        if arm_payloads is not None:
            loaded[arm] = arm_payloads

    if failures or set(loaded) != set(ARMS):
        return _invalid_result(failures, fail_invalid)

    control = loaded["control"]["metrics.json"]
    treatment = loaded["treatment"]["metrics.json"]
    treatment_instr = loaded["treatment"]["instrumentation.json"]

    control_peak = _peak_vram(control, "control", failures)
    treatment_peak = _peak_vram(treatment, "treatment", failures)
    if control_peak is not None and control_peak > vram_threshold_mb:
        failures.append(
            f"control peak_vram_mb {control_peak:.0f} exceeds threshold {vram_threshold_mb:.0f}"
        )
    if treatment_peak is not None and treatment_peak > vram_threshold_mb:
        failures.append(
            f"treatment peak_vram_mb {treatment_peak:.0f} exceeds threshold {vram_threshold_mb:.0f}"
        )

    projection_active, frozen_basis_nonempty = _projection_active(treatment_instr, failures)

    try:
        old_delta_gaps = [
            t - c
            for t, c in zip(treatment["old_task_deltas"], control["old_task_deltas"])
        ]
        delta_bwt_points = (treatment["bwt"] - control["bwt"]) * 100.0
        delta_avg_acc_points = (treatment["avg_acc"] - control["avg_acc"]) * 100.0
    except (KeyError, TypeError) as exc:
        failures.append(f"comparison field missing or malformed: {exc}")
        return _invalid_result(failures, fail_invalid)

    if failures:
        return _invalid_result(failures, fail_invalid)

    old_task_delta_gaps_points = [x * 100.0 for x in old_delta_gaps]
    comparison = {
        "valid": True,
        "failures": [],
        "delta_bwt_points": delta_bwt_points,
        "delta_avg_acc_points": delta_avg_acc_points,
        "old_task_delta_gaps_points": old_task_delta_gaps_points,
        "control_peak_vram_mb": control_peak,
        "treatment_peak_vram_mb": treatment_peak,
        "vram_threshold_mb": vram_threshold_mb,
        "projection_active": projection_active,
        "frozen_basis_nonempty": frozen_basis_nonempty,
    }
    comparison["continue"] = (
        comparison["delta_bwt_points"] >= 1.5
        and comparison["delta_avg_acc_points"] >= -2.0
        and min(comparison["old_task_delta_gaps_points"], default=0.0) >= -3.0
        and control_peak <= vram_threshold_mb
        and treatment_peak <= vram_threshold_mb
        and projection_active
        and frozen_basis_nonempty
    )
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CF-cycle-1 null-space ablation arms")
    parser.add_argument("--root", default="experiments/cf_cycle_1/nullspace_ablation")
    parser.add_argument("--vram-threshold-mb", type=float, default=7500.0)
    args = parser.parse_args()
    comparison = compare_runs(
        Path(args.root),
        vram_threshold_mb=args.vram_threshold_mb,
        fail_invalid=True,
    )
    print(json.dumps(comparison, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
