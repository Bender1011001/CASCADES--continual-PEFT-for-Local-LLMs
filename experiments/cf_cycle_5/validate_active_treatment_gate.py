from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_VARIANT = "cllora-active"
EXPECTED_EAR_GAMMA = 1e-4
CF_CYCLE_6_REMOVED_NORM_BASELINE = 0.0005073194128438748
CF_CYCLE_5_REMOVED_NORM_BASELINE = 0.000609848923952288
MATERIAL_STRENGTH_REMOVED_NORM_THRESHOLD = 0.00076231115494036


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_root(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def repo_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_active_treatment(
    root: Path,
    arm: str,
    expected_variant: str = EXPECTED_VARIANT,
    expected_frozen_basis_variance_threshold: float | None = None,
    expected_frozen_basis_top_k_per_freeze: int | None = None,
    expected_admission_policy: str | None = None,
    expect_utility_probe: bool = False,
) -> dict[str, Any]:
    arm_dir = root / arm
    config_path = arm_dir / "config.json"
    instrumentation_path = arm_dir / "instrumentation.json"
    run_status_path = arm_dir / "run_status.json"

    missing = [
        repo_relative(path)
        for path in [config_path, instrumentation_path, run_status_path]
        if not path.exists()
    ]
    if missing:
        return {
            "valid": False,
            "arm": arm,
            "arm_dir": repo_relative(arm_dir),
            "missing_artifacts": missing,
            "failures": [f"missing required artifact: {path}" for path in missing],
        }

    failures: list[str] = []
    config = load_json(config_path)
    instrumentation = load_json(instrumentation_path)
    run_status = load_json(run_status_path)

    ablation_config = config.get("ablation_config", {})
    reassign = instrumentation.get("reassign", {})
    utility_admission = instrumentation.get("utility_admission", {})

    config_checks = {
        "treatment_variant": config.get("treatment_variant") == expected_variant,
        "run_status_treatment_variant": run_status.get("treatment_variant") == expected_variant,
        "enable_coso_nullspace": ablation_config.get("enable_coso_nullspace") is True,
        "enable_cllora_reassign": ablation_config.get("enable_cllora_reassign") is True,
        "enable_soft_ear": ablation_config.get("enable_soft_ear") is True,
        "ear_gamma": (
            finite_number(ablation_config.get("ear_gamma"))
            and abs(float(ablation_config.get("ear_gamma")) - EXPECTED_EAR_GAMMA) < 1e-12
        ),
    }
    if expected_frozen_basis_variance_threshold is not None:
        actual_threshold = ablation_config.get("frozen_basis_variance_threshold")
        config_checks["frozen_basis_variance_threshold"] = (
            finite_number(actual_threshold)
            and abs(
                float(actual_threshold) - expected_frozen_basis_variance_threshold
            )
            < 1e-12
        )
    if expected_frozen_basis_top_k_per_freeze is not None:
        actual_top_k = ablation_config.get("frozen_basis_top_k_per_freeze")
        config_checks["frozen_basis_top_k_per_freeze"] = (
            isinstance(actual_top_k, int)
            and not isinstance(actual_top_k, bool)
            and actual_top_k == expected_frozen_basis_top_k_per_freeze
        )
    if expected_admission_policy is not None:
        config_checks["frozen_basis_admission_policy"] = (
            ablation_config.get("frozen_basis_admission_policy") == expected_admission_policy
        )
    if expect_utility_probe:
        config_checks["frozen_basis_utility_probe_enabled"] = (
            ablation_config.get("frozen_basis_utility_probe_enabled") is True
        )
    for key, passed in config_checks.items():
        if not passed:
            failures.append(f"config check failed: {key}")

    calls_with_frozen_basis = int(reassign.get("calls_with_frozen_basis", 0) or 0)
    max_frozen_cols = int(reassign.get("max_frozen_cols", 0) or 0)
    removed_norm_sum = reassign.get("removed_norm_sum", 0.0)
    removed_norm_per_frozen_call = reassign.get("removed_norm_per_frozen_call", 0.0)
    calls_with_active_reassign_path = int(reassign.get("calls_with_active_reassign_path", 0) or 0)
    active_adjustment_norm_sum = reassign.get("active_adjustment_norm_sum", 0.0)
    removed_norm_per_frozen_call_finite = finite_number(removed_norm_per_frozen_call)
    active_adjustment_positive = (
        finite_number(active_adjustment_norm_sum) and float(active_adjustment_norm_sum) > 0.0
    )
    removed_norm_above_cf_cycle_6_baseline = (
        removed_norm_per_frozen_call_finite
        and float(removed_norm_per_frozen_call) > CF_CYCLE_6_REMOVED_NORM_BASELINE
    )
    removed_norm_above_cf_cycle_5_baseline = (
        removed_norm_per_frozen_call_finite
        and float(removed_norm_per_frozen_call) > CF_CYCLE_5_REMOVED_NORM_BASELINE
    )
    removed_norm_above_material_strength = (
        removed_norm_per_frozen_call_finite
        and float(removed_norm_per_frozen_call) > MATERIAL_STRENGTH_REMOVED_NORM_THRESHOLD
    )

    instrumentation_checks = {
        "calls_with_frozen_basis_positive": calls_with_frozen_basis > 0,
        "removed_norm_sum_positive": finite_number(removed_norm_sum)
        and float(removed_norm_sum) > 0.0,
        "removed_norm_per_frozen_call_finite": removed_norm_per_frozen_call_finite,
        "calls_with_active_reassign_path_positive": calls_with_active_reassign_path > 0,
        "active_adjustment_norm_sum_positive": active_adjustment_positive,
    }
    for key, passed in instrumentation_checks.items():
        if not passed:
            failures.append(f"instrumentation check failed: {key}")

    utility_checks: dict[str, bool] = {}
    utility_summary = {
        "freeze_calls_with_utility_probe": int(
            utility_admission.get("freeze_calls_with_utility_probe", 0) or 0
        ),
        "candidates_considered_total": int(
            utility_admission.get("candidates_considered_total", 0) or 0
        ),
        "candidates_admitted_total": int(
            utility_admission.get("candidates_admitted_total", 0) or 0
        ),
        "candidates_vetoed_total": int(
            utility_admission.get("candidates_vetoed_total", 0) or 0
        ),
        "zero_admission_freeze_calls": int(
            utility_admission.get("zero_admission_freeze_calls", 0) or 0
        ),
        "mean_utility_delta_sum": utility_admission.get("mean_utility_delta_sum", 0.0),
        "min_old_task_delta_min": utility_admission.get("min_old_task_delta_min"),
        "per_old_task_veto_counts": utility_admission.get("per_old_task_veto_counts", {}),
    }
    if expected_admission_policy is not None:
        utility_checks["admission_policy"] = (
            ablation_config.get("frozen_basis_admission_policy") == expected_admission_policy
        )
    if expect_utility_probe:
        utility_checks["utility_probe_enabled"] = (
            ablation_config.get("frozen_basis_utility_probe_enabled") is True
        )
        utility_checks["freeze_calls_with_utility_probe_positive"] = (
            utility_summary["freeze_calls_with_utility_probe"] > 0
        )
        utility_checks["utility_candidates_internally_consistent"] = (
            utility_summary["candidates_admitted_total"] >= 0
            and utility_summary["candidates_vetoed_total"] >= 0
            and utility_summary["candidates_admitted_total"]
            + utility_summary["candidates_vetoed_total"]
            <= utility_summary["candidates_considered_total"]
        )
        utility_checks["zero_admission_count_nonnegative"] = (
            utility_summary["zero_admission_freeze_calls"] >= 0
        )
    for key, passed in utility_checks.items():
        if not passed:
            failures.append(f"utility check failed: {key}")

    return {
        "valid": not failures,
        "arm": arm,
        "arm_dir": repo_relative(arm_dir),
        "failures": failures,
        "config_checks": config_checks,
        "instrumentation_checks": instrumentation_checks,
        "active_adjustment_positive": active_adjustment_positive,
        "config_summary": {
            "treatment_variant": config.get("treatment_variant"),
            "run_status_treatment_variant": run_status.get("treatment_variant"),
            "enable_coso_nullspace": ablation_config.get("enable_coso_nullspace"),
            "enable_cllora_reassign": ablation_config.get("enable_cllora_reassign"),
            "enable_soft_ear": ablation_config.get("enable_soft_ear"),
            "ear_gamma": ablation_config.get("ear_gamma"),
            "frozen_basis_variance_threshold": ablation_config.get(
                "frozen_basis_variance_threshold"
            ),
            "frozen_basis_top_k_per_freeze": ablation_config.get(
                "frozen_basis_top_k_per_freeze"
            ),
            "frozen_basis_admission_policy": ablation_config.get(
                "frozen_basis_admission_policy"
            ),
            "frozen_basis_utility_probe_enabled": ablation_config.get(
                "frozen_basis_utility_probe_enabled"
            ),
        },
        "instrumentation_summary": {
            "calls_with_frozen_basis": calls_with_frozen_basis,
            "max_frozen_cols": max_frozen_cols,
            "removed_norm_sum": removed_norm_sum,
            "removed_norm_per_frozen_call": removed_norm_per_frozen_call,
            "removed_norm_above_cf_cycle_6_baseline": removed_norm_above_cf_cycle_6_baseline,
            "removed_norm_above_cf_cycle_5_baseline": removed_norm_above_cf_cycle_5_baseline,
            "removed_norm_above_material_strength": removed_norm_above_material_strength,
            "calls_with_active_reassign_enabled": reassign.get(
                "calls_with_active_reassign_enabled", 0
            ),
            "calls_with_active_reassign_path": calls_with_active_reassign_path,
            "active_adjustment_norm_sum": active_adjustment_norm_sum,
            "active_adjustment_norm_max": reassign.get("active_adjustment_norm_max", 0.0),
        },
        "mechanism_summary": {
            "max_frozen_cols": max_frozen_cols,
            "removed_norm_per_frozen_call": removed_norm_per_frozen_call,
            "removed_norm_above_cf_cycle_6_baseline": removed_norm_above_cf_cycle_6_baseline,
            "removed_norm_above_cf_cycle_5_baseline": removed_norm_above_cf_cycle_5_baseline,
            "removed_norm_above_material_strength": removed_norm_above_material_strength,
        },
        "utility_checks": utility_checks,
        "utility_admission_summary": utility_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate CF-cycle-5 active treatment gate")
    parser.add_argument("--root", default="experiments/cf_cycle_5/cllora_active_seed43")
    parser.add_argument("--arm", choices=["treatment"], default="treatment")
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-variant", default=EXPECTED_VARIANT)
    parser.add_argument("--expected-frozen-basis-variance-threshold", type=float)
    parser.add_argument("--expected-frozen-basis-top-k-per-freeze", type=int)
    parser.add_argument("--expected-admission-policy")
    parser.add_argument("--expect-utility-probe", action="store_true")
    args = parser.parse_args()

    payload = validate_active_treatment(
        resolve_root(args.root),
        args.arm,
        expected_variant=args.expected_variant,
        expected_frozen_basis_variance_threshold=args.expected_frozen_basis_variance_threshold,
        expected_frozen_basis_top_k_per_freeze=args.expected_frozen_basis_top_k_per_freeze,
        expected_admission_policy=args.expected_admission_policy,
        expect_utility_probe=args.expect_utility_probe,
    )
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=False))
    return 0 if payload.get("valid") else 1


if __name__ == "__main__":
    sys.exit(main())
