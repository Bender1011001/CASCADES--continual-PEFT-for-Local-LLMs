from __future__ import annotations

from dataclasses import asdict
import json

import pytest
import torch

import cascades.adapters as adapters_mod
from cascades.adapters import CASCADESAdapter
from cascades.config import AblationConfig, DEFAULT_CONFIG
from experiments.cf_cycle_1.run_nullspace_ablation import config_for_arm
from experiments.cf_cycle_5.validate_active_treatment_gate import validate_active_treatment


def _utility_decider():
    helper = getattr(adapters_mod, "utility_veto_admission_decisions", None)
    assert helper is not None, "utility-veto admission helper is missing"
    return helper


def test_config_defaults_are_serializable_and_salience_only():
    cfg_dict = asdict(DEFAULT_CONFIG)
    json.dumps(cfg_dict)

    assert cfg_dict["frozen_basis_admission_policy"] == "salience"
    assert cfg_dict["frozen_basis_utility_probe_enabled"] is False
    assert cfg_dict["frozen_basis_utility_probe_batch_size"] == 2
    assert cfg_dict["frozen_basis_utility_probe_batches_per_old_task"] == 1
    assert cfg_dict["frozen_basis_utility_min_mean_delta"] == 0.0
    assert cfg_dict["frozen_basis_utility_old_task_veto_drop"] == 0.0
    assert cfg_dict["frozen_basis_utility_max_probe_examples_per_old_task"] == 2
    assert cfg_dict["frozen_basis_utility_metric"] == "heldout_loss_proxy"


def test_new_variant_routing_sets_utility_veto_without_changing_control():
    treatment = asdict(
        config_for_arm("treatment", "cllora-active-freeze-utility-veto-topk-2")
    )
    control = asdict(
        config_for_arm("control", "cllora-active-freeze-utility-veto-topk-2")
    )
    legacy_topk = asdict(config_for_arm("treatment", "cllora-active-freeze-topk-2"))

    assert treatment["enable_coso_nullspace"] is True
    assert treatment["enable_cllora_reassign"] is True
    assert treatment["enable_soft_ear"] is True
    assert treatment["ear_gamma"] == pytest.approx(1e-4)
    assert treatment["frozen_basis_variance_threshold"] == pytest.approx(0.05)
    assert treatment["frozen_basis_top_k_per_freeze"] == 2
    assert treatment["frozen_basis_admission_policy"] == "utility_veto"
    assert treatment["frozen_basis_utility_probe_enabled"] is True

    assert control["enable_coso_nullspace"] is False
    assert control["enable_cllora_reassign"] is False
    assert control["frozen_basis_top_k_per_freeze"] is None
    assert control["frozen_basis_admission_policy"] == "salience"
    assert control["frozen_basis_utility_probe_enabled"] is False

    assert legacy_topk["frozen_basis_variance_threshold"] == pytest.approx(0.05)
    assert legacy_topk["frozen_basis_top_k_per_freeze"] == 2
    assert legacy_topk["frozen_basis_admission_policy"] == "salience"


def test_utility_veto_admits_utility_passing_top_structural_candidates():
    decide = _utility_decider()
    result = decide(
        structural_scores=torch.tensor([0.2, 0.8]),
        utility_delta_by_candidate=[[0.0, 0.1], [0.2, 0.3]],
        top_k=2,
        min_mean_delta=0.0,
        old_task_veto_drop=0.0,
        old_task_count=2,
    )

    assert result.admitted_indices == [1, 0]
    assert result.vetoed_indices == []
    assert result.utility_gate_admitted_any is True
    assert result.zero_admission_reason is None


def test_utility_veto_blocks_high_score_candidate_that_harms_old_task():
    decide = _utility_decider()
    result = decide(
        structural_scores=torch.tensor([0.9, 0.3]),
        utility_delta_by_candidate=[[0.5, -0.01], [0.0, 0.2]],
        top_k=2,
        min_mean_delta=0.0,
        old_task_veto_drop=0.0,
        old_task_count=2,
    )

    assert result.admitted_indices == [1]
    assert result.vetoed_indices == [0]
    assert result.veto_task_indices_by_candidate == [[1], []]
    assert result.min_old_task_delta_by_candidate[0] == pytest.approx(-0.01)


def test_utility_veto_has_no_salience_fallback_when_all_candidates_fail():
    decide = _utility_decider()
    result = decide(
        structural_scores=torch.tensor([0.9, 0.8]),
        utility_delta_by_candidate=[[-0.1, 0.0], [0.0, -0.2]],
        top_k=2,
        min_mean_delta=0.0,
        old_task_veto_drop=0.0,
        old_task_count=2,
    )

    assert result.admitted_indices == []
    assert result.vetoed_indices == [0, 1]
    assert result.utility_gate_admitted_any is False
    assert result.zero_admission_reason == "all_candidates_vetoed"


def test_utility_veto_bypasses_probe_when_no_old_tasks_exist():
    decide = _utility_decider()
    result = decide(
        structural_scores=torch.tensor([0.1, 0.9, 0.3]),
        utility_delta_by_candidate=None,
        top_k=2,
        min_mean_delta=0.0,
        old_task_veto_drop=0.0,
        old_task_count=0,
    )

    assert result.admitted_indices == [1, 2]
    assert result.vetoed_indices == []
    assert result.zero_admission_reason is None


def test_utility_veto_freeze_keeps_cpu_basis_bounded_and_orthonormal():
    torch.manual_seed(9)
    cfg = AblationConfig(
        frozen_basis_admission_policy="utility_veto",
        frozen_basis_utility_probe_enabled=True,
        frozen_basis_top_k_per_freeze=2,
    )
    adapter = CASCADESAdapter(in_features=7, out_features=6, rank=4, config=cfg)
    adapter.streaming_sketch_U.copy_(torch.randn_like(adapter.streaming_sketch_U))
    adapter.streaming_sketch_V.copy_(torch.randn_like(adapter.streaming_sketch_V))

    adapter.freeze_current_subspace()

    assert adapter.frozen_null_basis.shape[1] <= 2
    assert adapter.frozen_null_basis_V.shape[1] <= 2
    assert torch.isfinite(adapter.frozen_null_basis).all()
    assert torch.isfinite(adapter.frozen_null_basis_V).all()
    assert torch.allclose(
        adapter.frozen_null_basis.T @ adapter.frozen_null_basis,
        torch.eye(adapter.frozen_null_basis.shape[1]),
        atol=1e-5,
    )
    assert getattr(adapter, "last_freeze_utility_admission", {}).get(
        "admission_policy"
    ) == "utility_veto"


def test_active_validator_accepts_utility_admission_fixture(tmp_path):
    root = tmp_path / "fixture"
    arm_dir = root / "treatment"
    arm_dir.mkdir(parents=True)

    cfg = asdict(
        config_for_arm("treatment", "cllora-active-freeze-utility-veto-topk-2")
    )
    (arm_dir / "config.json").write_text(
        json.dumps(
            {
                "treatment_variant": "cllora-active-freeze-utility-veto-topk-2",
                "ablation_config": cfg,
            }
        ),
        encoding="utf-8",
    )
    (arm_dir / "run_status.json").write_text(
        json.dumps({"treatment_variant": "cllora-active-freeze-utility-veto-topk-2"}),
        encoding="utf-8",
    )
    (arm_dir / "instrumentation.json").write_text(
        json.dumps(
            {
                "reassign": {
                    "calls_with_frozen_basis": 3,
                    "max_frozen_cols": 2,
                    "removed_norm_sum": 0.03,
                    "removed_norm_per_frozen_call": 0.01,
                    "calls_with_active_reassign_enabled": 3,
                    "calls_with_active_reassign_path": 3,
                    "active_adjustment_norm_sum": 0.02,
                    "active_adjustment_norm_max": 0.01,
                },
                "utility_admission": {
                    "freeze_calls_with_utility_probe": 2,
                    "candidates_considered_total": 4,
                    "candidates_admitted_total": 1,
                    "candidates_vetoed_total": 3,
                    "zero_admission_freeze_calls": 1,
                    "mean_utility_delta_sum": 0.4,
                    "min_old_task_delta_min": 0.0,
                    "per_old_task_veto_counts": {"1": 2},
                },
            }
        ),
        encoding="utf-8",
    )

    payload = validate_active_treatment(
        root,
        "treatment",
        expected_variant="cllora-active-freeze-utility-veto-topk-2",
        expected_frozen_basis_variance_threshold=0.05,
        expected_frozen_basis_top_k_per_freeze=2,
        expected_admission_policy="utility_veto",
        expect_utility_probe=True,
    )

    assert payload["valid"] is True
    assert payload["utility_admission_summary"]["candidates_admitted_total"] == 1
    assert payload["utility_checks"]["utility_probe_enabled"] is True
