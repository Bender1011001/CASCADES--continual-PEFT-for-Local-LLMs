# Symmetric Bipolar Attention Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a gate-staged validation scaffold for Symmetric Bipolar Attention so CASCADES can validate, falsify, or bound the sinh-cosh signed-attention idea before any larger-suite escalation.

**Architecture:** Add standalone SBA math utilities and CPU tests first, then artifact-writing gate scripts, then opt-in model patching after model load, then tiny-model and paired reduced-training gates. SBA remains default-off, evidence-driven, and blocked by deterministic artifacts, finite-value checks, same-manifest paired runs, and the existing 7500 MB VRAM convention.

**Tech Stack:** Python 3.11+, PyTorch, pytest, Transformers, BitsAndBytes, PEFT, JSON artifacts, Windows `cmd.exe` command syntax, RTX 4060 Ti 8GB gate convention.

---

## Current-documentation scope

This plan is the only implementation artifact created in this task. The current worker must not implement Python code, run tests, run CUDA checks, or launch GPU workloads. All commands below are for the future implementation worker after switching to code mode and receiving the required approvals for GPU stages.

## File structure map

### Core math and CPU tests

- Modify `cascades/math_ops.py`: add the default-off SBA v1 formula identifier, mask normalization helper, readable reference sinh-cosh attention, stable optimized positive/negative channel implementation, value aggregation helper, diagnostics, and Jacobian measurement helpers.
- Modify `tests/test_math.py`: add CPU-only SBA tests for formula equivalence, safe masking, all-masked rows, signed L1 bound, finite outputs, finite gradients, and measured Jacobian diagnostics.

### Gate artifact utilities and CPU math gate

- Create `experiments/sba_validation/__init__.py`: make the SBA validation directory importable for tests and scripts.
- Create `experiments/sba_validation/artifact_utils.py`: centralize git revision lookup, JSON writing, finite-number scanning, artifact validation, and gate result construction.
- Create `experiments/sba_validation/run_cpu_math_gate.py`: run deterministic CPU cases, write `cpu_math_gate.json` and `cpu_math_gate.log`, and fail if formula equivalence, masking, signed L1, finite math, or Jacobian measurement requirements are unmet.
- Create `tests/test_sba_validation_artifacts.py`: CPU-only tests for artifact validators, including non-finite rejection, missing required field rejection, and CPU gate pass/fail payload shape.

### Synthetic CUDA gate

- Create `experiments/sba_validation/run_cuda_synthetic_gate.py`: run approved synthetic CUDA stress cases only after CPU gate passes and explicit GPU approval is granted; record device, shapes, dtypes, mask density, seed, finite forward/backward status, signed L1, masked leakage, timing, peak VRAM, and artifacts.

### Selectable model patch and tiny patched-model gate

- Modify `cascades/config.py`: add default-off SBA fields to `AblationConfig` so configs serialize whether the patch was requested and which formula variant was used.
- Create `cascades/sba_patch.py`: implement a reversible, explicit patch helper that replaces supported no-cache attention module forwards with SBA value aggregation, plus a patch report proving installed module count and names. Unsupported architecture aborts instead of silently labeling baseline as SBA.
- Create `tests/test_sba_patch.py`: CPU-only tests with toy modules proving default-off behavior, patch installation accounting, unsupported-module failure, and no silent fallback.
- Modify `train.py`: add default-off `enable_sba_patch` and `sba_patch_report_path` parameters, call the patch helper immediately after `AutoModelForCausalLM.from_pretrained()`, and preserve all existing defaults.
- Create `experiments/sba_validation/run_tiny_model_gate.py`: run the explicit tiny patched-model forward/backward gate only after CUDA synthetic gate passes and GPU approval is granted.

### Paired reduced training and comparison

- Create `experiments/sba_validation/run_paired_reduced_sba.py`: direct paired runner modeled on `experiments/cf_cycle_1/run_nullspace_ablation.py`, with control and SBA treatment directories, same manifest, same seed, same model ID, same rank, same max length, same epochs, same sleep behavior, 7500 MB threshold, and treatment patch-active evidence.
- Create `experiments/sba_validation/compare_paired_reduced_sba.py`: compare accepted control and treatment artifacts only after both arms are completed, finite, same-manifest, same-seed, same-revision, and under 7500 MB.
- Create `experiments/sba_validation/validate_sba_artifacts.py`: reusable CLI validator for CPU math, CUDA synthetic, tiny model, and paired reduced gate artifacts.
- Modify `experiments/cf_cycle_1/run_nullspace_ablation.py`: only if the paired runner reuses its helpers directly, expose a narrow `task_manifest_for_suite()` helper without changing existing CF-cycle behavior. Prefer not modifying this file if `run_paired_reduced_sba.py` can copy the manifest constants safely.

### Reporting and context

- Create `experiments/sba_validation/README.md`: describe gate order, artifact contracts, approval boundaries, and exact commands.
- Create `experiments/sba_validation/write_result_critic_packet.py`: generate `experiments/sba_validation/RESULT_CRITIC_PACKET.md` from gate artifacts and enforce larger-suite escalation blocking unless all prior gates pass and the user explicitly approves escalation.
- Modify `CONTEXT.md`: record the completed implementation, verification evidence, artifacts, escalation decision, and the fact that catastrophic forgetting remains unsolved.

## Gate ladder and hard pass/fail summary

1. **CPU math gate:** pass only if reference and optimized formulas match within tolerance, masked positions are inert, signed L1 row bound is at most `1 + tolerance`, outputs and gradients are finite, and the Jacobian claim is directly measured as either supported or falsified.
2. **CUDA synthetic gate:** pass only if forward outputs and backward gradients are finite, signed L1 and masked leakage bounds hold under CUDA tolerance, benchmark artifacts are durable, and peak VRAM is below or equal to `7500` MB.
3. **Patched model integration gate:** pass only if intended modules are patched, no silent fallback occurs, tiny forward loss is finite, backward gradients are finite, and peak VRAM is below or equal to `7500` MB.
4. **Paired reduced training gate:** pass only if control and treatment use the same manifest, seed, model ID, rank, max length, epochs, hardware threshold, and code revision; both arms complete; both arms have finite metrics; both remain under `7500` MB; treatment proves SBA patch active; comparison writes only after both arm gates pass.
5. **Larger-suite escalation gate:** pass only if all prior gates pass, result critic accepts the evidence quality, treatment shows useful retention or stability without unacceptable average-accuracy regression, and the user explicitly approves the larger run.

---

## Implementation tasks

### Task 1: Add RED CPU tests for SBA math invariants

**Files:**
- Modify: `tests/test_math.py`

- [ ] **Step 1: Import the planned SBA utilities**

Add these names to the existing import block in `tests/test_math.py`:

```python
    SBA_FORMULA_VARIANT,
    sba_reference_attention,
    sba_optimized_attention,
    sba_value_attention,
    sba_attention_diagnostics,
    sba_jacobian_frobenius_norm,
    softmax_jacobian_frobenius_norm,
```

- [ ] **Step 2: Add exact CPU invariant tests**

Append this block to `tests/test_math.py`:

```python
# ---------------------------------------------------------------------------
# Symmetric Bipolar Attention v1 validation invariants
# ---------------------------------------------------------------------------

class TestSymmetricBipolarAttention:
    def test_formula_variant_is_pinned(self):
        assert SBA_FORMULA_VARIANT == "sba-v1-valid-position-sinh-cosh-shared-denominator"

    def test_reference_formula_hand_checked_values(self):
        logits = torch.tensor([[[0.0, math.log(2.0), -math.log(2.0)]]], dtype=torch.float64)
        weights = sba_reference_attention(logits)
        expected = torch.tensor([[[0.0, 0.21428571428571427, -0.21428571428571427]]], dtype=torch.float64)
        assert torch.allclose(weights, expected, atol=1e-12, rtol=0.0)

    def test_optimized_matches_reference_over_shapes_and_masks(self):
        generator = torch.Generator().manual_seed(20260518)
        shapes = [(2, 3), (2, 4, 5), (2, 2, 3, 7)]
        for shape in shapes:
            logits = torch.randn(*shape, generator=generator, dtype=torch.float64) * 3.0
            valid_mask = torch.rand(*shape, generator=generator) > 0.25
            valid_mask[..., 0] = True
            reference = sba_reference_attention(logits, valid_mask)
            optimized = sba_optimized_attention(logits, valid_mask)
            assert torch.allclose(optimized, reference, atol=1e-10, rtol=1e-10)

    def test_safe_masking_and_all_masked_rows(self):
        logits = torch.tensor([[[50.0, -50.0, 0.0], [1.0, 2.0, 3.0]]], dtype=torch.float64)
        valid_mask = torch.tensor([[[True, False, True], [False, False, False]]])
        values = torch.tensor([[[1.0, 0.0], [1000.0, 1000.0], [0.0, 1.0]]], dtype=torch.float64)
        weights, attended = sba_value_attention(logits, values, valid_mask)
        assert weights[0, 0, 1].item() == 0.0
        assert torch.all(weights[0, 1] == 0.0)
        assert torch.all(attended[0, 1] == 0.0)
        assert torch.isfinite(weights).all().item()
        assert torch.isfinite(attended).all().item()
        assert abs(attended[0, 0, 1].item()) <= 1.0

    def test_signed_l1_bound_and_finite_gradients(self):
        logits = torch.tensor(
            [[[15.0, -12.0, 0.5, -0.25], [0.0, 0.0, 0.0, 0.0]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        valid_mask = torch.tensor([[[True, True, True, False], [True, True, True, True]]])
        weights = sba_optimized_attention(logits, valid_mask)
        diagnostics = sba_attention_diagnostics(weights, valid_mask)
        assert diagnostics["finite"] is True
        assert diagnostics["max_masked_leakage"] == 0.0
        assert diagnostics["max_signed_l1"] <= 1.0 + 1e-12
        loss = (weights.square().sum() + weights.sum())
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all().item()

    def test_jacobian_claim_is_measured_not_assumed(self):
        logits = torch.tensor([[0.25, -0.75, 1.25]], dtype=torch.float64)
        valid_mask = torch.tensor([[True, True, True]])
        sba_norm = sba_jacobian_frobenius_norm(logits, valid_mask)
        softmax_norm = softmax_jacobian_frobenius_norm(logits, valid_mask)
        assert math.isfinite(sba_norm)
        assert math.isfinite(softmax_norm)
        assert sba_norm >= 0.0
        assert softmax_norm >= 0.0
```

- [ ] **Step 3: Run the RED tests**

Run:

```cmd
python -m pytest tests\test_math.py -k SymmetricBipolarAttention -q
```

Expected: exit code `1`, with failures showing the SBA imports are missing. The output must contain this text:

```text
ImportError
```

If pytest reports collection failure using `cannot import name`, that is also the correct RED state.

### Task 2: Implement standalone SBA math utilities

**Files:**
- Modify: `cascades/math_ops.py`
- Modify: `tests/test_math.py`

- [ ] **Step 1: Add SBA functions to `cascades/math_ops.py`**

Insert this block after the imports in `cascades/math_ops.py`:

```python
SBA_FORMULA_VARIANT = "sba-v1-valid-position-sinh-cosh-shared-denominator"


def _sba_valid_mask(logits: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Return a Boolean mask broadcastable to logits, where True means key is valid."""
    if valid_mask is None:
        return torch.ones_like(logits, dtype=torch.bool)
    mask = valid_mask.to(device=logits.device)
    if mask.dtype != torch.bool:
        mask = mask != 0
    while mask.dim() < logits.dim():
        mask = mask.unsqueeze(-2)
    return torch.broadcast_to(mask, logits.shape)


def sba_reference_attention(
    logits: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Readable valid-position sinh-cosh normalized signed attention."""
    valid = _sba_valid_mask(logits, valid_mask)
    zeros = torch.zeros_like(logits)
    safe_logits = torch.where(valid, logits, zeros)
    numerator = torch.where(valid, torch.sinh(safe_logits), zeros)
    denominator = torch.where(valid, torch.cosh(safe_logits), zeros).sum(dim=-1, keepdim=True)
    has_valid = valid.any(dim=-1, keepdim=True)
    safe_denominator = denominator.clamp_min(torch.finfo(logits.dtype).tiny)
    weights = torch.where(has_valid, numerator / safe_denominator, zeros)
    return torch.where(valid, weights, zeros)


def sba_optimized_attention(
    logits: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Stable positive/negative evidence-channel implementation of SBA v1."""
    valid = _sba_valid_mask(logits, valid_mask)
    zeros = torch.zeros_like(logits)
    dtype_floor = torch.finfo(logits.dtype).min
    floor = torch.full_like(logits, dtype_floor)
    pos_logits = torch.where(valid, logits, floor)
    neg_logits = torch.where(valid, -logits, floor)
    max_pos = pos_logits.max(dim=-1, keepdim=True).values
    max_neg = neg_logits.max(dim=-1, keepdim=True).values
    has_valid = valid.any(dim=-1, keepdim=True)
    row_max = torch.maximum(max_pos, max_neg)
    row_max = torch.where(has_valid, row_max, torch.zeros_like(row_max))
    e_pos = torch.where(valid, torch.exp(pos_logits - row_max), zeros)
    e_neg = torch.where(valid, torch.exp(neg_logits - row_max), zeros)
    denominator = (e_pos + e_neg).sum(dim=-1, keepdim=True)
    safe_denominator = denominator.clamp_min(torch.finfo(logits.dtype).tiny)
    weights = torch.where(has_valid, (e_pos - e_neg) / safe_denominator, zeros)
    return torch.where(valid, weights, zeros)


def sba_value_attention(
    logits: torch.Tensor,
    values: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return SBA attention weights and the corresponding value aggregation."""
    weights = sba_optimized_attention(logits, valid_mask)
    attended = torch.matmul(weights, values)
    return weights, attended


def sba_attention_diagnostics(
    weights: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, object]:
    """Compute finite, signed-sum, signed-L1, and masked-leakage diagnostics."""
    valid = _sba_valid_mask(weights, valid_mask)
    masked_weights = torch.where(valid, torch.zeros_like(weights), weights)
    row_signed_sum = weights.sum(dim=-1)
    row_abs_sum = weights.abs().sum(dim=-1)
    return {
        "finite": bool(torch.isfinite(weights).all().item()),
        "row_signed_sum": row_signed_sum.detach(),
        "row_abs_sum": row_abs_sum.detach(),
        "max_signed_l1": float(row_abs_sum.max().item()) if row_abs_sum.numel() else 0.0,
        "max_masked_leakage": float(masked_weights.abs().max().item()) if masked_weights.numel() else 0.0,
    }


def sba_jacobian_frobenius_norm(
    logits: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> float:
    """Measure the Frobenius norm of the SBA attention Jacobian for small CPU tensors."""
    base = logits.detach().clone().requires_grad_(True)

    def fn(flat_logits: torch.Tensor) -> torch.Tensor:
        reshaped = flat_logits.reshape_as(base)
        return sba_optimized_attention(reshaped, valid_mask).reshape(-1)

    jacobian = torch.autograd.functional.jacobian(fn, base.reshape(-1), create_graph=False)
    return float(torch.linalg.matrix_norm(jacobian).item())


def softmax_jacobian_frobenius_norm(
    logits: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> float:
    """Measure a masked-softmax Jacobian norm for comparison with SBA diagnostics."""
    base = logits.detach().clone().requires_grad_(True)
    valid = _sba_valid_mask(base, valid_mask)
    floor = torch.full_like(base, torch.finfo(base.dtype).min)

    def fn(flat_logits: torch.Tensor) -> torch.Tensor:
        reshaped = flat_logits.reshape_as(base)
        masked = torch.where(valid, reshaped, floor)
        return torch.softmax(masked, dim=-1).reshape(-1)

    jacobian = torch.autograd.functional.jacobian(fn, base.reshape(-1), create_graph=False)
    return float(torch.linalg.matrix_norm(jacobian).item())
```

- [ ] **Step 2: Run the SBA math tests**

Run:

```cmd
python -m pytest tests\test_math.py -k SymmetricBipolarAttention -q
```

Expected: exit code `0`, with this summary shape:

```text
7 passed
```

- [ ] **Step 3: Run the full math test file**

Run:

```cmd
python -m pytest tests\test_math.py -q
```

Expected: exit code `0`, with all tests in `tests\test_math.py` passing and no `FAILED` entries.

- [ ] **Step 4: Commit math utilities and tests**

Run:

```cmd
git add cascades\math_ops.py tests\test_math.py && git commit -m "feat: add symmetric bipolar attention math gate"
```

Expected: exit code `0`, with git reporting a commit that changes `cascades/math_ops.py` and `tests/test_math.py`.

### Task 3: Add reusable gate artifact utilities and tests

**Files:**
- Create: `experiments/sba_validation/__init__.py`
- Create: `experiments/sba_validation/artifact_utils.py`
- Create: `tests/test_sba_validation_artifacts.py`

- [ ] **Step 1: Create the package marker**

Create `experiments/sba_validation/__init__.py` with this content:

```python
"""Symmetric Bipolar Attention validation gate scripts and artifact helpers."""
```

- [ ] **Step 2: Write RED tests for artifact validation**

Create `tests/test_sba_validation_artifacts.py` with this content:

```python
import math

from experiments.sba_validation.artifact_utils import (
    collect_nonfinite_paths,
    gate_payload,
    validate_required_fields,
)


def test_collect_nonfinite_paths_finds_nested_nan_and_inf():
    payload = {"outer": {"nan": float("nan"), "values": [1.0, float("inf")]}}
    failures = collect_nonfinite_paths(payload, "root")
    assert "root.outer.nan" in failures
    assert "root.outer.values[1]" in failures


def test_validate_required_fields_reports_missing_field():
    failures = validate_required_fields({"valid": True}, ["valid", "seed"], "artifact")
    assert failures == ["artifact missing required field: seed"]


def test_gate_payload_rejects_nonfinite_numbers():
    payload = gate_payload(
        gate="unit",
        command="python -m pytest",
        seed=123,
        formula_variant="sba-v1-valid-position-sinh-cosh-shared-denominator",
        mask_policy="valid_positions_only",
        dtype_policy="float64_cpu",
        checks={"finite": True, "bad_value": math.inf},
        failures=[],
        log_path="experiments/sba_validation/unit.log",
    )
    assert payload["valid"] is False
    assert "checks.bad_value" in payload["failure_reasons"][0]
```

- [ ] **Step 3: Run the RED artifact tests**

Run:

```cmd
python -m pytest tests\test_sba_validation_artifacts.py -q
```

Expected: exit code `1`, with output containing:

```text
ModuleNotFoundError
```

- [ ] **Step 4: Implement artifact utilities**

Create `experiments/sba_validation/artifact_utils.py` with this content:

```python
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def collect_nonfinite_paths(value: Any, path: str) -> list[str]:
    failures: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            failures.extend(collect_nonfinite_paths(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            failures.extend(collect_nonfinite_paths(item, f"{path}[{index}]"))
    elif isinstance(value, (int, float)) and not math.isfinite(float(value)):
        failures.append(path)
    return failures


def validate_required_fields(
    payload: dict[str, Any],
    required: Iterable[str],
    label: str,
) -> list[str]:
    failures: list[str] = []
    for field in required:
        if field not in payload:
            failures.append(f"{label} missing required field: {field}")
    return failures


def gate_payload(
    gate: str,
    command: str,
    seed: int,
    formula_variant: str,
    mask_policy: str,
    dtype_policy: str,
    checks: dict[str, Any],
    failures: list[str],
    log_path: str,
    device: dict[str, Any] | None = None,
    peak_vram_mb: float | None = None,
) -> dict[str, Any]:
    nonfinite_paths = collect_nonfinite_paths(checks, "checks")
    all_failures = list(failures) + [f"non-finite numeric value at {p}" for p in nonfinite_paths]
    payload: dict[str, Any] = {
        "gate": gate,
        "valid": len(all_failures) == 0,
        "failure_reasons": all_failures,
        "git_revision": git_revision(),
        "command": command,
        "seed": seed,
        "device": device or {"type": "cpu"},
        "formula_variant": formula_variant,
        "mask_policy": mask_policy,
        "dtype_policy": dtype_policy,
        "checks": checks,
        "peak_vram_mb": peak_vram_mb,
        "log_path": log_path,
    }
    return payload
```

- [ ] **Step 5: Run artifact tests**

Run:

```cmd
python -m pytest tests\test_sba_validation_artifacts.py -q
```

Expected: exit code `0`, with this summary:

```text
3 passed
```

- [ ] **Step 6: Commit artifact utility foundation**

Run:

```cmd
git add experiments\sba_validation\__init__.py experiments\sba_validation\artifact_utils.py tests\test_sba_validation_artifacts.py && git commit -m "feat: add sba validation artifact utilities"
```

Expected: exit code `0`, with git reporting a focused commit for the SBA validation utility files.

### Task 4: Add the CPU math gate artifact writer

**Files:**
- Create: `experiments/sba_validation/run_cpu_math_gate.py`
- Modify: `tests/test_sba_validation_artifacts.py`

- [ ] **Step 1: Add RED tests for CPU gate artifact shape**

Append this test to `tests/test_sba_validation_artifacts.py`:

```python
from experiments.sba_validation.run_cpu_math_gate import run_cpu_math_gate


def test_cpu_math_gate_payload_is_valid_and_records_jacobian_status(tmp_path):
    artifact_path = tmp_path / "cpu_math_gate.json"
    log_path = tmp_path / "cpu_math_gate.log"
    payload = run_cpu_math_gate(artifact_path=artifact_path, log_path=log_path)
    assert payload["valid"] is True
    assert payload["checks"]["formula_equivalence_max_diff"] <= 1e-8
    assert payload["checks"]["max_masked_leakage"] == 0.0
    assert payload["checks"]["max_signed_l1"] <= 1.0 + 1e-8
    assert payload["checks"]["finite_outputs"] is True
    assert payload["checks"]["finite_gradients"] is True
    assert payload["checks"]["jacobian_claim_status"] in {"supported", "falsified"}
    assert artifact_path.exists()
    assert log_path.exists()
```

- [ ] **Step 2: Run the RED CPU gate test**

Run:

```cmd
python -m pytest tests\test_sba_validation_artifacts.py -k cpu_math_gate -q
```

Expected: exit code `1`, with output containing:

```text
ModuleNotFoundError
```

- [ ] **Step 3: Implement the CPU gate script**

Create `experiments/sba_validation/run_cpu_math_gate.py` with this content:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from cascades.math_ops import (
    SBA_FORMULA_VARIANT,
    sba_attention_diagnostics,
    sba_jacobian_frobenius_norm,
    sba_optimized_attention,
    sba_reference_attention,
    sba_value_attention,
    softmax_jacobian_frobenius_norm,
)
from experiments.sba_validation.artifact_utils import gate_payload, write_json


SEED = 20260518
MASK_POLICY = "valid_positions_only_zero_masked_weights"
DTYPE_POLICY = "float64_cpu_primary_float32_smoke"


def _finite_grad_check() -> bool:
    logits = torch.tensor(
        [[[15.0, -12.0, 0.5, -0.25], [0.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    valid_mask = torch.tensor([[[True, True, True, False], [True, True, True, True]]])
    weights = sba_optimized_attention(logits, valid_mask)
    loss = weights.square().sum() + weights.sum()
    loss.backward()
    return logits.grad is not None and bool(torch.isfinite(logits.grad).all().item())


def run_cpu_math_gate(
    artifact_path: Path = Path("experiments/sba_validation/cpu_math_gate.json"),
    log_path: Path = Path("experiments/sba_validation/cpu_math_gate.log"),
) -> dict[str, object]:
    torch.manual_seed(SEED)
    failures: list[str] = []
    generator = torch.Generator().manual_seed(SEED)
    equivalence_diffs: list[float] = []
    signed_l1_values: list[float] = []
    masked_leakage_values: list[float] = []
    finite_outputs = True
    for shape in [(2, 3), (2, 4, 5), (2, 2, 3, 7)]:
        logits = torch.randn(*shape, generator=generator, dtype=torch.float64) * 3.0
        valid_mask = torch.rand(*shape, generator=generator) > 0.25
        valid_mask[..., 0] = True
        reference = sba_reference_attention(logits, valid_mask)
        optimized = sba_optimized_attention(logits, valid_mask)
        equivalence_diffs.append(float((reference - optimized).abs().max().item()))
        finite_outputs = finite_outputs and bool(torch.isfinite(optimized).all().item())
        diagnostics = sba_attention_diagnostics(optimized, valid_mask)
        signed_l1_values.append(float(diagnostics["max_signed_l1"]))
        masked_leakage_values.append(float(diagnostics["max_masked_leakage"]))
    all_masked_logits = torch.tensor([[[1.0, -2.0, 3.0]]], dtype=torch.float64)
    all_masked = torch.zeros_like(all_masked_logits, dtype=torch.bool)
    all_masked_weights = sba_optimized_attention(all_masked_logits, all_masked)
    values = torch.randn(1, 3, 2, generator=generator, dtype=torch.float64)
    weights, attended = sba_value_attention(all_masked_logits, values, all_masked)
    finite_outputs = finite_outputs and bool(torch.isfinite(weights).all().item())
    finite_outputs = finite_outputs and bool(torch.isfinite(attended).all().item())
    if not torch.all(all_masked_weights == 0.0).item():
        failures.append("all-masked row produced nonzero SBA weights")
    max_equivalence = max(equivalence_diffs, default=0.0)
    max_signed_l1 = max(signed_l1_values, default=0.0)
    max_masked_leakage = max(masked_leakage_values, default=0.0)
    if max_equivalence > 1e-8:
        failures.append(f"formula equivalence max diff {max_equivalence} exceeds 1e-8")
    if max_masked_leakage > 0.0:
        failures.append(f"masked leakage {max_masked_leakage} exceeds 0.0")
    if max_signed_l1 > 1.0 + 1e-8:
        failures.append(f"signed L1 {max_signed_l1} exceeds 1 + 1e-8")
    finite_gradients = _finite_grad_check()
    if not finite_outputs:
        failures.append("SBA outputs or value aggregation are non-finite")
    if not finite_gradients:
        failures.append("SBA gradients are non-finite")
    jacobian_logits = torch.tensor([[0.25, -0.75, 1.25]], dtype=torch.float64)
    jacobian_mask = torch.tensor([[True, True, True]])
    sba_jacobian = sba_jacobian_frobenius_norm(jacobian_logits, jacobian_mask)
    softmax_jacobian = softmax_jacobian_frobenius_norm(jacobian_logits, jacobian_mask)
    jacobian_claim_status = "supported" if sba_jacobian <= softmax_jacobian else "falsified"
    checks = {
        "formula_equivalence_max_diff": max_equivalence,
        "max_masked_leakage": max_masked_leakage,
        "max_signed_l1": max_signed_l1,
        "finite_outputs": finite_outputs,
        "finite_gradients": finite_gradients,
        "jacobian_claim": "sba_jacobian_frobenius_norm <= softmax_jacobian_frobenius_norm",
        "jacobian_claim_status": jacobian_claim_status,
        "sba_jacobian_frobenius_norm": sba_jacobian,
        "softmax_jacobian_frobenius_norm": softmax_jacobian,
    }
    payload = gate_payload(
        gate="cpu_math",
        command="python experiments\\sba_validation\\run_cpu_math_gate.py",
        seed=SEED,
        formula_variant=SBA_FORMULA_VARIANT,
        mask_policy=MASK_POLICY,
        dtype_policy=DTYPE_POLICY,
        checks=checks,
        failures=failures,
        log_path=str(log_path),
    )
    write_json(artifact_path, payload)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "CPU math gate completed\n"
        f"valid={payload['valid']}\n"
        f"failure_reasons={payload['failure_reasons']}\n"
        f"checks={checks}\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SBA CPU math validation gate")
    parser.add_argument("--out", default="experiments/sba_validation/cpu_math_gate.json")
    parser.add_argument("--log", default="experiments/sba_validation/cpu_math_gate.log")
    args = parser.parse_args()
    payload = run_cpu_math_gate(Path(args.out), Path(args.log))
    if not payload["valid"]:
        print(payload["failure_reasons"])
        raise SystemExit(2)
    print(f"CPU math gate valid: {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run CPU gate unit test**

Run:

```cmd
python -m pytest tests\test_sba_validation_artifacts.py -k cpu_math_gate -q
```

Expected: exit code `0`, with this summary:

```text
1 passed
```

- [ ] **Step 5: Run CPU math gate script**

Run:

```cmd
python experiments\sba_validation\run_cpu_math_gate.py --out experiments\sba_validation\cpu_math_gate.json --log experiments\sba_validation\cpu_math_gate.log
```

Expected: exit code `0`, with stdout exactly shaped like:

```text
CPU math gate valid: experiments\sba_validation\cpu_math_gate.json
```

Expected artifact hard criteria in `experiments\sba_validation\cpu_math_gate.json`:

```json
{
  "gate": "cpu_math",
  "valid": true,
  "formula_variant": "sba-v1-valid-position-sinh-cosh-shared-denominator",
  "mask_policy": "valid_positions_only_zero_masked_weights"
}
```

- [ ] **Step 6: Commit CPU gate**

Run:

```cmd
git add experiments\sba_validation\run_cpu_math_gate.py experiments\sba_validation\cpu_math_gate.json experiments\sba_validation\cpu_math_gate.log tests\test_sba_validation_artifacts.py && git commit -m "feat: add sba cpu math gate artifact"
```

Expected: exit code `0`, with a commit containing the CPU gate script, test update, and durable CPU artifacts.

### Task 5: Add synthetic CUDA stress gate script, but do not run it until approved

**Files:**
- Create: `experiments/sba_validation/run_cuda_synthetic_gate.py`
- Modify: `tests/test_sba_validation_artifacts.py`

- [ ] **Step 1: Add CPU-only import and command-shape test**

Append this test to `tests/test_sba_validation_artifacts.py`:

```python
from experiments.sba_validation.run_cuda_synthetic_gate import CUDA_SYNTHETIC_CASES, build_case_label


def test_cuda_synthetic_cases_are_bounded_and_labeled():
    labels = [build_case_label(case) for case in CUDA_SYNTHETIC_CASES]
    assert labels == [
        "b1_h2_q16_k16_d32_float32_mask0.00",
        "b1_h2_q32_k32_d64_float32_mask0.25",
        "b1_h4_q64_k64_d64_float16_mask0.50",
    ]
```

- [ ] **Step 2: Run the RED test**

Run:

```cmd
python -m pytest tests\test_sba_validation_artifacts.py -k cuda_synthetic_cases -q
```

Expected: exit code `1`, with output containing:

```text
ModuleNotFoundError
```

- [ ] **Step 3: Create the CUDA synthetic gate script**

Create `experiments/sba_validation/run_cuda_synthetic_gate.py` with this content:

```python
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from cascades.math_ops import SBA_FORMULA_VARIANT, sba_attention_diagnostics, sba_optimized_attention
from experiments.sba_validation.artifact_utils import gate_payload, write_json


CUDA_SYNTHETIC_CASES = [
    {"batch": 1, "heads": 2, "queries": 16, "keys": 16, "dim": 32, "dtype": "float32", "mask_density": 0.0},
    {"batch": 1, "heads": 2, "queries": 32, "keys": 32, "dim": 64, "dtype": "float32", "mask_density": 0.25},
    {"batch": 1, "heads": 4, "queries": 64, "keys": 64, "dim": 64, "dtype": "float16", "mask_density": 0.50},
]


def build_case_label(case: dict[str, object]) -> str:
    return (
        f"b{case['batch']}_h{case['heads']}_q{case['queries']}_k{case['keys']}"
        f"_d{case['dim']}_{case['dtype']}_mask{float(case['mask_density']):.2f}"
    )


def _dtype(name: str) -> torch.dtype:
    return torch.float16 if name == "float16" else torch.float32


def run_cuda_synthetic_gate(
    artifact_path: Path = Path("experiments/sba_validation/cuda_synthetic_gate.json"),
    log_path: Path = Path("experiments/sba_validation/cuda_synthetic_gate.log"),
    vram_threshold_mb: float = 7500.0,
) -> dict[str, object]:
    seed = 20260518
    torch.manual_seed(seed)
    failures: list[str] = []
    if not torch.cuda.is_available():
        failures.append("CUDA is unavailable")
        payload = gate_payload(
            gate="cuda_synthetic",
            command="python experiments\\sba_validation\\run_cuda_synthetic_gate.py --vram-threshold-mb 7500",
            seed=seed,
            formula_variant=SBA_FORMULA_VARIANT,
            mask_policy="valid_positions_only_zero_masked_weights",
            dtype_policy="float32_and_float16_cuda",
            checks={"cases": []},
            failures=failures,
            log_path=str(log_path),
            device={"type": "cuda", "available": False},
        )
        write_json(artifact_path, payload)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("CUDA unavailable\n", encoding="utf-8")
        return payload
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    case_results: list[dict[str, object]] = []
    for case in CUDA_SYNTHETIC_CASES:
        dtype = _dtype(str(case["dtype"]))
        label = build_case_label(case)
        shape = (int(case["batch"]), int(case["heads"]), int(case["queries"]), int(case["keys"]))
        logits = torch.randn(*shape, device=device, dtype=dtype).requires_grad_(True)
        mask = torch.rand(*shape, device=device) >= float(case["mask_density"])
        mask[..., 0] = True
        start = time.perf_counter()
        weights = sba_optimized_attention(logits, mask)
        loss = weights.float().square().mean()
        loss.backward()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        diagnostics = sba_attention_diagnostics(weights.detach().float(), mask)
        grad_finite = logits.grad is not None and bool(torch.isfinite(logits.grad).all().item())
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        case_result = {
            "label": label,
            "shape": list(shape),
            "dtype": str(case["dtype"]),
            "mask_density": float(case["mask_density"]),
            "forward_finite": bool(torch.isfinite(weights).all().item()),
            "backward_finite": grad_finite,
            "max_signed_l1": diagnostics["max_signed_l1"],
            "max_masked_leakage": diagnostics["max_masked_leakage"],
            "elapsed_ms": elapsed_ms,
            "peak_vram_mb": peak_mb,
        }
        case_results.append(case_result)
        if not case_result["forward_finite"]:
            failures.append(f"{label} forward output is non-finite")
        if not case_result["backward_finite"]:
            failures.append(f"{label} backward gradient is non-finite")
        if float(case_result["max_signed_l1"]) > 1.0 + 2e-3:
            failures.append(f"{label} signed L1 bound exceeded CUDA tolerance")
        if float(case_result["max_masked_leakage"]) > 1e-6:
            failures.append(f"{label} masked leakage exceeded 1e-6")
        if peak_mb > vram_threshold_mb:
            failures.append(f"{label} peak VRAM {peak_mb:.0f} MB exceeded {vram_threshold_mb:.0f} MB")
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    checks = {"cases": case_results, "vram_threshold_mb": vram_threshold_mb}
    payload = gate_payload(
        gate="cuda_synthetic",
        command="python experiments\\sba_validation\\run_cuda_synthetic_gate.py --vram-threshold-mb 7500",
        seed=seed,
        formula_variant=SBA_FORMULA_VARIANT,
        mask_policy="valid_positions_only_zero_masked_weights",
        dtype_policy="float32_and_float16_cuda",
        checks=checks,
        failures=failures,
        log_path=str(log_path),
        device={"type": "cuda", "available": True, "name": torch.cuda.get_device_name(device)},
        peak_vram_mb=peak_vram_mb,
    )
    write_json(artifact_path, payload)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(f"CUDA synthetic gate valid={payload['valid']}\nchecks={checks}\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SBA synthetic CUDA stress gate")
    parser.add_argument("--out", default="experiments/sba_validation/cuda_synthetic_gate.json")
    parser.add_argument("--log", default="experiments/sba_validation/cuda_synthetic_gate.log")
    parser.add_argument("--vram-threshold-mb", type=float, default=7500.0)
    args = parser.parse_args()
    payload = run_cuda_synthetic_gate(Path(args.out), Path(args.log), args.vram_threshold_mb)
    if not payload["valid"]:
        print(payload["failure_reasons"])
        raise SystemExit(2)
    print(f"CUDA synthetic gate valid: {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run CPU-only import test and syntax check**

Run:

```cmd
python -m pytest tests\test_sba_validation_artifacts.py -k cuda_synthetic_cases -q && python -m py_compile experiments\sba_validation\run_cuda_synthetic_gate.py
```

Expected: exit code `0`, with pytest reporting:

```text
1 passed
```

Expected `py_compile`: no stdout and no stderr.

- [ ] **Step 5: Run CUDA gate only after explicit GPU approval**

Run only after CPU math gate artifact has `valid=true` and the user approves GPU work:

```cmd
python experiments\sba_validation\run_cuda_synthetic_gate.py --out experiments\sba_validation\cuda_synthetic_gate.json --log experiments\sba_validation\cuda_synthetic_gate.log --vram-threshold-mb 7500
```

Expected pass output:

```text
CUDA synthetic gate valid: experiments\sba_validation\cuda_synthetic_gate.json
```

Hard fail criteria: exit code `2`, `valid=false`, any non-finite forward or backward value, masked leakage above `1e-6`, signed L1 above `1.002`, missing benchmark artifacts, CUDA unavailable after approval, or peak VRAM above `7500` MB.

- [ ] **Step 6: Commit synthetic CUDA gate after CPU-only checks, and include CUDA artifacts only if the approved run was executed**

If only CPU checks were run:

```cmd
git add experiments\sba_validation\run_cuda_synthetic_gate.py tests\test_sba_validation_artifacts.py && git commit -m "feat: add sba synthetic cuda gate"
```

If the approved CUDA run was also executed and passed:

```cmd
git add experiments\sba_validation\run_cuda_synthetic_gate.py experiments\sba_validation\cuda_synthetic_gate.json experiments\sba_validation\cuda_synthetic_gate.log tests\test_sba_validation_artifacts.py && git commit -m "feat: add sba synthetic cuda gate"
```

Expected: exit code `0`, with git reporting a focused commit.

### Task 6: Add default-off patch config and patch helper tests

**Files:**
- Modify: `cascades/config.py`
- Create: `cascades/sba_patch.py`
- Create: `tests/test_sba_patch.py`

- [ ] **Step 1: Write RED patch tests**

Create `tests/test_sba_patch.py` with this content:

```python
import pytest
import torch
import torch.nn as nn

from cascades.config import AblationConfig
from cascades.sba_patch import SbaPatchError, apply_sba_patch, collect_patch_report


class ToyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.sba_enabled = False
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 4, bias=False)
        self.o_proj = nn.Linear(4, 4, bias=False)
        self.num_heads = 1
        self.head_dim = 4
        self.hidden_size = 4

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        return self.o_proj(self.v_proj(hidden_states)), None, None


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([ToyAttention(), ToyAttention()])


def test_default_config_keeps_sba_disabled():
    cfg = AblationConfig()
    assert cfg.enable_symmetric_bipolar_attention is False
    assert cfg.sba_formula_variant == "sba-v1-valid-position-sinh-cosh-shared-denominator"


def test_apply_patch_marks_toy_attention_modules():
    model = ToyModel()
    report = apply_sba_patch(model, module_class_names={"ToyAttention"})
    assert report["requested"] is True
    assert report["patched_module_count"] == 2
    assert report["patched_module_names"] == ["layers.0", "layers.1"]
    assert model.layers[0].sba_enabled is True
    assert model.layers[1].sba_enabled is True


def test_patched_toy_attention_uses_signed_weights():
    model = ToyModel()
    apply_sba_patch(model, module_class_names={"ToyAttention"})
    hidden = torch.tensor([[[1.0, 0.0, -1.0, 0.5], [0.0, 1.0, 0.25, -0.5]]])
    output, attention_weights, _ = model.layers[0](hidden, output_attentions=True)
    assert output.shape == hidden.shape
    assert attention_weights is not None
    assert torch.isfinite(output).all().item()
    assert attention_weights.abs().sum(dim=-1).max().item() <= 1.0 + 1e-6


def test_unsupported_model_raises_instead_of_silent_fallback():
    model = nn.Linear(4, 4)
    with pytest.raises(SbaPatchError, match="no supported attention modules"):
        apply_sba_patch(model, module_class_names={"ToyAttention"})


def test_collect_patch_report_detects_enabled_modules():
    model = ToyModel()
    apply_sba_patch(model, module_class_names={"ToyAttention"})
    report = collect_patch_report(model)
    assert report["patched_module_count"] == 2
    assert report["patched_module_names"] == ["layers.0", "layers.1"]
```

- [ ] **Step 2: Run RED patch tests**

Run:

```cmd
python -m pytest tests\test_sba_patch.py -q
```

Expected: exit code `1`, with output containing:

```text
ModuleNotFoundError
```

- [ ] **Step 3: Add default-off config fields**

Add these fields to `AblationConfig` in `cascades/config.py` after `frozen_basis_utility_metric`:

```python
    # Symmetric Bipolar Attention validation patch. Default-off; not production attention.
    enable_symmetric_bipolar_attention: bool = False
    sba_formula_variant: str = "sba-v1-valid-position-sinh-cosh-shared-denominator"
```

- [ ] **Step 4: Implement the patch helper**

Create `cascades/sba_patch.py` with this content:

```python
from __future__ import annotations

import math
from types import MethodType
from typing import Iterable

import torch
import torch.nn as nn

from cascades.math_ops import SBA_FORMULA_VARIANT, sba_value_attention


DEFAULT_SUPPORTED_ATTENTION_CLASS_NAMES = {
    "Qwen2Attention",
    "Qwen2SdpaAttention",
    "Qwen3Attention",
    "Qwen3SdpaAttention",
}


class SbaPatchError(RuntimeError):
    """Raised when an SBA patch request cannot be installed safely."""


def _module_class_name(module: nn.Module) -> str:
    return module.__class__.__name__


def collect_patch_report(model: nn.Module) -> dict[str, object]:
    names = [name for name, module in model.named_modules() if getattr(module, "sba_enabled", False)]
    return {
        "requested": bool(names),
        "formula_variant": SBA_FORMULA_VARIANT,
        "patched_module_count": len(names),
        "patched_module_names": names,
    }


def _normalize_attention_mask(attention_mask, logits: torch.Tensor) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    mask = attention_mask.to(device=logits.device)
    if mask.dtype == torch.bool:
        valid = mask
    else:
        valid = mask >= 0 if mask.dim() == logits.dim() else mask != 0
    while valid.dim() < logits.dim():
        valid = valid.unsqueeze(1)
    return torch.broadcast_to(valid, logits.shape)


def _install_sba_forward(module: nn.Module) -> None:
    original_forward = module.forward

    def sba_forward(self, hidden_states, attention_mask=None, output_attentions=False, **kwargs):
        batch, query_len, _ = hidden_states.shape
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        num_heads = int(getattr(self, "num_heads", 1))
        head_dim = int(getattr(self, "head_dim", query.shape[-1] // num_heads))
        query = query.view(batch, query_len, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch, query_len, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch, query_len, num_heads, head_dim).transpose(1, 2)
        logits = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(head_dim)
        valid_mask = _normalize_attention_mask(attention_mask, logits)
        weights, attended = sba_value_attention(logits, value, valid_mask)
        attended = attended.transpose(1, 2).contiguous().view(batch, query_len, num_heads * head_dim)
        output = self.o_proj(attended)
        return output, weights if output_attentions else None, None

    setattr(module, "sba_original_forward", original_forward)
    module.forward = MethodType(sba_forward, module)


def apply_sba_patch(
    model: nn.Module,
    module_class_names: Iterable[str] | None = None,
) -> dict[str, object]:
    """Mark supported attention modules for SBA and reject unsupported silent fallback."""
    supported = set(module_class_names or DEFAULT_SUPPORTED_ATTENTION_CLASS_NAMES)
    patched_names: list[str] = []
    for name, module in model.named_modules():
        if _module_class_name(module) in supported:
            setattr(module, "sba_enabled", True)
            setattr(module, "sba_formula_variant", SBA_FORMULA_VARIANT)
            _install_sba_forward(module)
            patched_names.append(name)
    if not patched_names:
        raise SbaPatchError(
            "SBA patch requested but no supported attention modules were found; "
            "aborting to prevent silent baseline fallback"
        )
    return {
        "requested": True,
        "formula_variant": SBA_FORMULA_VARIANT,
        "supported_class_names": sorted(supported),
        "patched_module_count": len(patched_names),
        "patched_module_names": patched_names,
    }
```

- [ ] **Step 5: Run patch tests**

Run:

```cmd
python -m pytest tests\test_sba_patch.py -q
```

Expected: exit code `0`, with this summary:

```text
6 passed
```

- [ ] **Step 6: Commit patch helper foundation**

Run:

```cmd
git add cascades\config.py cascades\sba_patch.py tests\test_sba_patch.py && git commit -m "feat: add default-off sba patch helper"
```

Expected: exit code `0`, with git reporting a focused commit.

### Task 7: Wire optional patching after model load without changing defaults

**Files:**
- Modify: `train.py`
- Modify: `tests/test_sba_patch.py`

- [ ] **Step 1: Add a RED source-level wiring test**

Append this test to `tests/test_sba_patch.py`:

```python
from pathlib import Path


def test_train_wires_sba_patch_after_model_load():
    source = Path("train.py").read_text(encoding="utf-8")
    load_index = source.index("AutoModelForCausalLM.from_pretrained")
    patch_index = source.index("apply_sba_patch")
    inject_index = source.index("inject_cascades(")
    assert load_index < patch_index < inject_index
    assert "enable_sba_patch: bool = False" in source
    assert "sba_patch_report_path: str | None = None" in source
```

- [ ] **Step 2: Run RED wiring test**

Run:

```cmd
python -m pytest tests\test_sba_patch.py -k train_wires -q
```

Expected: exit code `1`, with output containing:

```text
ValueError: substring not found
```

- [ ] **Step 3: Modify `train.py` imports and signature**

Add this import near the CASCADES imports:

```python
from cascades.sba_patch import SbaPatchError, apply_sba_patch
from experiments.sba_validation.artifact_utils import write_json
```

Add these parameters to `train_cascades()` after `vram_threshold_mb`:

```python
    enable_sba_patch: bool = False,
    sba_patch_report_path: str | None = None,
```

Add these docstring lines after the `vram_threshold_mb` line:

```python
        enable_sba_patch: Default-off SBA validation patch flag.
        sba_patch_report_path: Optional JSON path proving installed SBA patch modules.
```

- [ ] **Step 4: Apply patch immediately after model load**

Insert this block immediately after the `AutoModelForCausalLM.from_pretrained()` call and before `prepare_model_for_kbit_training`:

```python
    if enable_sba_patch:
        try:
            patch_report = apply_sba_patch(model)
        except SbaPatchError:
            raise
        if sba_patch_report_path is not None:
            write_json(Path(sba_patch_report_path), patch_report)
        print(
            "SBA patch installed: "
            f"{patch_report['patched_module_count']} attention modules"
        )
```

Add `Path` to the import block:

```python
from pathlib import Path
```

- [ ] **Step 5: Run CPU wiring tests and syntax check**

Run:

```cmd
python -m pytest tests\test_sba_patch.py -q && python -m py_compile train.py cascades\sba_patch.py
```

Expected: exit code `0`, with pytest summary:

```text
6 passed
```

Expected `py_compile`: no stdout and no stderr.

- [ ] **Step 6: Commit default-off train wiring**

Run:

```cmd
git add train.py tests\test_sba_patch.py && git commit -m "feat: wire default-off sba model patch"
```

Expected: exit code `0`, with git reporting a focused commit.

### Task 8: Add tiny patched-model forward/backward gate

**Files:**
- Create: `experiments/sba_validation/run_tiny_model_gate.py`
- Modify: `tests/test_sba_validation_artifacts.py`

- [ ] **Step 1: Add CPU-only command constant test**

Append this test to `tests/test_sba_validation_artifacts.py`:

```python
from experiments.sba_validation.run_tiny_model_gate import TINY_MODEL_GATE_COMMAND


def test_tiny_model_gate_command_uses_sba_patch_and_threshold():
    assert "--enable-sba-patch" in TINY_MODEL_GATE_COMMAND
    assert "--vram-threshold-mb 7500" in TINY_MODEL_GATE_COMMAND
```

- [ ] **Step 2: Run RED test**

Run:

```cmd
python -m pytest tests\test_sba_validation_artifacts.py -k tiny_model_gate_command -q
```

Expected: exit code `1`, with output containing:

```text
ModuleNotFoundError
```

- [ ] **Step 3: Create tiny model gate script**

Create `experiments/sba_validation/run_tiny_model_gate.py` with this content:

```python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from cascades.math_ops import SBA_FORMULA_VARIANT
from cascades.sba_patch import apply_sba_patch
from experiments.sba_validation.artifact_utils import gate_payload, write_json


DEFAULT_MODEL_ID = "p-e-w/Qwen3-4B-Instruct-2507-heretic"
TINY_MODEL_GATE_COMMAND = (
    "python experiments\\sba_validation\\run_tiny_model_gate.py "
    "--model-id p-e-w/Qwen3-4B-Instruct-2507-heretic "
    "--enable-sba-patch --vram-threshold-mb 7500"
)


def run_tiny_model_gate(
    model_id: str,
    artifact_path: Path,
    log_path: Path,
    vram_threshold_mb: float,
) -> dict[str, object]:
    seed = 20260518
    torch.manual_seed(seed)
    failures: list[str] = []
    if not torch.cuda.is_available():
        failures.append("CUDA is unavailable for tiny patched-model gate")
        payload = gate_payload(
            gate="tiny_model",
            command=TINY_MODEL_GATE_COMMAND,
            seed=seed,
            formula_variant=SBA_FORMULA_VARIANT,
            mask_policy="model_attention_mask_valid_tokens",
            dtype_policy="nf4_bfloat16_compute",
            checks={"model_id": model_id},
            failures=failures,
            log_path=str(log_path),
            device={"type": "cuda", "available": False},
        )
        write_json(artifact_path, payload)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("CUDA unavailable\n", encoding="utf-8")
        return payload
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
    patch_report = apply_sba_patch(model)
    encoded = tokenizer("SBA validation tiny forward backward check", return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    labels = encoded["input_ids"].clone()
    model.train()
    outputs = model(**encoded, labels=labels)
    loss = outputs.loss
    forward_finite = bool(torch.isfinite(loss).all().item())
    loss.backward()
    grad_finite_values: list[bool] = []
    for parameter in model.parameters():
        if parameter.grad is not None:
            grad_finite_values.append(bool(torch.isfinite(parameter.grad).all().item()))
    backward_finite = bool(grad_finite_values) and all(grad_finite_values)
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    if patch_report["patched_module_count"] <= 0:
        failures.append("no SBA patched modules were installed")
    if not forward_finite:
        failures.append("tiny patched-model loss is non-finite")
    if not backward_finite:
        failures.append("tiny patched-model gradients are non-finite")
    if peak_vram_mb > vram_threshold_mb:
        failures.append(f"tiny patched-model peak VRAM {peak_vram_mb:.0f} MB exceeded {vram_threshold_mb:.0f} MB")
    checks = {
        "model_id": model_id,
        "patched_module_count": patch_report["patched_module_count"],
        "patched_module_names": patch_report["patched_module_names"],
        "input_shape": list(encoded["input_ids"].shape),
        "finite_loss": forward_finite,
        "finite_gradients": backward_finite,
        "no_silent_fallback": patch_report["patched_module_count"] > 0,
        "vram_threshold_mb": vram_threshold_mb,
    }
    payload = gate_payload(
        gate="tiny_model",
        command=TINY_MODEL_GATE_COMMAND,
        seed=seed,
        formula_variant=SBA_FORMULA_VARIANT,
        mask_policy="model_attention_mask_valid_tokens",
        dtype_policy="nf4_bfloat16_compute",
        checks=checks,
        failures=failures,
        log_path=str(log_path),
        device={"type": "cuda", "available": True, "name": torch.cuda.get_device_name(device)},
        peak_vram_mb=peak_vram_mb,
    )
    write_json(artifact_path, payload)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(f"Tiny model gate valid={payload['valid']}\nchecks={checks}\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SBA tiny patched-model gate")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--enable-sba-patch", action="store_true", required=True)
    parser.add_argument("--out", default="experiments/sba_validation/tiny_model_gate.json")
    parser.add_argument("--log", default="experiments/sba_validation/tiny_model_gate.log")
    parser.add_argument("--vram-threshold-mb", type=float, default=7500.0)
    args = parser.parse_args()
    payload = run_tiny_model_gate(Path(args.model_id).as_posix() if False else args.model_id, Path(args.out), Path(args.log), args.vram_threshold_mb)
    if not payload["valid"]:
        print(payload["failure_reasons"])
        raise SystemExit(2)
    print(f"Tiny model gate valid: {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run CPU-only script tests and syntax check**

Run:

```cmd
python -m pytest tests\test_sba_validation_artifacts.py -k tiny_model_gate_command -q && python -m py_compile experiments\sba_validation\run_tiny_model_gate.py
```

Expected: exit code `0`, pytest summary:

```text
1 passed
```

Expected `py_compile`: no stdout and no stderr.

- [ ] **Step 5: Run tiny model gate only after CUDA synthetic gate passes and GPU approval is explicit**

Run:

```cmd
python experiments\sba_validation\run_tiny_model_gate.py --model-id p-e-w/Qwen3-4B-Instruct-2507-heretic --enable-sba-patch --out experiments\sba_validation\tiny_model_gate.json --log experiments\sba_validation\tiny_model_gate.log --vram-threshold-mb 7500
```

Expected pass output:

```text
Tiny model gate valid: experiments\sba_validation\tiny_model_gate.json
```

Hard fail criteria: exit code `2`, missing patch report, patched module count `0`, non-finite loss, non-finite gradients, silent baseline fallback, or peak VRAM above `7500` MB.

- [ ] **Step 6: Commit tiny model gate**

If GPU gate was not approved yet, commit only script and tests:

```cmd
git add experiments\sba_validation\run_tiny_model_gate.py tests\test_sba_validation_artifacts.py && git commit -m "feat: add sba tiny model gate"
```

If approved GPU gate ran and passed, include artifacts:

```cmd
git add experiments\sba_validation\run_tiny_model_gate.py experiments\sba_validation\tiny_model_gate.json experiments\sba_validation\tiny_model_gate.log tests\test_sba_validation_artifacts.py && git commit -m "feat: add sba tiny model gate"
```

Expected: exit code `0`, with git reporting a focused commit.

### Task 9: Add paired reduced SBA runner and comparison gate

**Files:**
- Create: `experiments/sba_validation/run_paired_reduced_sba.py`
- Create: `experiments/sba_validation/compare_paired_reduced_sba.py`
- Create: `experiments/sba_validation/validate_sba_artifacts.py`
- Modify: `tests/test_sba_validation_artifacts.py`

- [ ] **Step 1: Add RED tests for paired scope constants and comparison validation**

Append these tests to `tests/test_sba_validation_artifacts.py`:

```python
from experiments.sba_validation.compare_paired_reduced_sba import compare_paired_artifacts
from experiments.sba_validation.run_paired_reduced_sba import PAIRED_REDUCED_CONFIG, REASONING3_TASKS


def test_paired_reduced_config_is_bounded():
    assert REASONING3_TASKS == [
        "data/task0_gsm8k_cot.jsonl",
        "data/task1_arc_cot.jsonl",
        "data/task2_csqa_cot.jsonl",
    ]
    assert PAIRED_REDUCED_CONFIG["seed"] == 43
    assert PAIRED_REDUCED_CONFIG["rank"] == 4
    assert PAIRED_REDUCED_CONFIG["max_length"] == 256
    assert PAIRED_REDUCED_CONFIG["epochs"] == 2
    assert PAIRED_REDUCED_CONFIG["vram_threshold_mb"] == 7500.0


def test_compare_rejects_mismatched_manifest(tmp_path):
    root = tmp_path / "paired_reduced"
    for arm in ["control", "treatment"]:
        arm_dir = root / arm
        arm_dir.mkdir(parents=True)
        (arm_dir / "run_status.json").write_text('{"status":"completed","seed":43,"git_revision":"abc"}', encoding="utf-8")
        (arm_dir / "metrics.json").write_text('{"avg_acc":0.5,"bwt":0.0,"peak_vram_mb":1000.0,"old_task_deltas":[0.0,0.0]}', encoding="utf-8")
        manifest = [{"task_index":0,"path":"data/task0_gsm8k_cot.jsonl"}]
        if arm == "treatment":
            manifest = [{"task_index":0,"path":"data/task1_arc_cot.jsonl"}]
        (arm_dir / "task_manifest.json").write_text(__import__("json").dumps(manifest), encoding="utf-8")
        (arm_dir / "instrumentation.json").write_text('{"sba_patch":{"patched_module_count":1,"patched_module_names":["model.layers.0.self_attn"]}}', encoding="utf-8")
    result = compare_paired_artifacts(root, fail_invalid=False)
    assert result["valid"] is False
    assert "task manifests differ across arms" in result["failures"]
```

- [ ] **Step 2: Run RED paired tests**

Run:

```cmd
python -m pytest tests\test_sba_validation_artifacts.py -k "paired_reduced_config or mismatched_manifest" -q
```

Expected: exit code `1`, with output containing:

```text
ModuleNotFoundError
```

- [ ] **Step 3: Create the paired runner**

Create `experiments/sba_validation/run_paired_reduced_sba.py` with this content:

```python
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cascades.data as data
from experiments.sba_validation.artifact_utils import git_revision, write_json


REASONING3_TASKS = [
    "data/task0_gsm8k_cot.jsonl",
    "data/task1_arc_cot.jsonl",
    "data/task2_csqa_cot.jsonl",
]
PAIRED_REDUCED_CONFIG = {
    "model_id": "p-e-w/Qwen3-4B-Instruct-2507-heretic",
    "seed": 43,
    "rank": 4,
    "max_length": 256,
    "epochs": 2,
    "vram_threshold_mb": 7500.0,
    "task_suite": "reasoning3",
}


def apply_reasoning3_manifest() -> list[str]:
    data.TASK_FILES = list(REASONING3_TASKS)
    data.TASK_NAMES = {i: f"Task {i} ({Path(path).stem})" for i, path in enumerate(REASONING3_TASKS)}
    data.NUM_TASKS = len(REASONING3_TASKS)
    return list(REASONING3_TASKS)


def task_manifest() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, file_name in enumerate(REASONING3_TASKS):
        path = ROOT / file_name
        rows.append({
            "task_index": index,
            "path": file_name,
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else None,
        })
    return rows


def run_arm(arm: str, output_root: Path, enable_sba_patch: bool) -> None:
    import train

    apply_reasoning3_manifest()
    train.NUM_TASKS = len(REASONING3_TASKS)
    arm_dir = output_root / arm
    if arm_dir.exists():
        shutil.rmtree(arm_dir)
    arm_dir.mkdir(parents=True)
    write_json(arm_dir / "task_manifest.json", task_manifest())
    config = dict(PAIRED_REDUCED_CONFIG)
    config.update({"arm": arm, "enable_sba_patch": enable_sba_patch, "git_revision": git_revision()})
    write_json(arm_dir / "config.json", config)
    patch_report_path = arm_dir / "sba_patch_report.json"
    status = "completed"
    reason = None
    try:
        matrix = train.train_cascades(
            seed=PAIRED_REDUCED_CONFIG["seed"],
            model_id=PAIRED_REDUCED_CONFIG["model_id"],
            output_prefix=str(arm_dir / "cascades"),
            epochs=PAIRED_REDUCED_CONFIG["epochs"],
            rank=PAIRED_REDUCED_CONFIG["rank"],
            max_length=PAIRED_REDUCED_CONFIG["max_length"],
            enable_sleep=True,
            abort_on_nonfinite=True,
            vram_threshold_mb=PAIRED_REDUCED_CONFIG["vram_threshold_mb"],
            enable_sba_patch=enable_sba_patch,
            sba_patch_report_path=str(patch_report_path) if enable_sba_patch else None,
        )
    except Exception as exc:
        status = "failed"
        reason = str(exc)
        matrix = None
    if matrix is not None:
        import numpy as np
        from cascades.metrics import average_accuracy, backward_transfer

        np.save(arm_dir / "accuracy_matrix.npy", matrix)
        metrics = {
            "arm": arm,
            "avg_acc": float(average_accuracy(matrix)),
            "bwt": float(backward_transfer(matrix)),
            "old_task_deltas": [float(matrix[-1, i] - matrix[i, i]) for i in range(matrix.shape[0] - 1)],
            "peak_vram_mb": float(__import__("torch").cuda.max_memory_allocated() / (1024**2)) if __import__("torch").cuda.is_available() else 0.0,
            "seed": PAIRED_REDUCED_CONFIG["seed"],
            "rank": PAIRED_REDUCED_CONFIG["rank"],
            "max_length": PAIRED_REDUCED_CONFIG["max_length"],
            "epochs": PAIRED_REDUCED_CONFIG["epochs"],
            "git_revision": git_revision(),
        }
        write_json(arm_dir / "metrics.json", metrics)
    sba_patch = json.loads(patch_report_path.read_text(encoding="utf-8")) if patch_report_path.exists() else {"patched_module_count": 0, "patched_module_names": []}
    write_json(arm_dir / "instrumentation.json", {"sba_patch": sba_patch})
    write_json(arm_dir / "run_status.json", {"status": status, "reason": reason, "seed": PAIRED_REDUCED_CONFIG["seed"], "git_revision": git_revision()})
    if status != "completed":
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired reduced SBA validation")
    parser.add_argument("--arm", choices=["control", "treatment"], required=True)
    parser.add_argument("--output-root", default="experiments/sba_validation/paired_reduced")
    args = parser.parse_args()
    run_arm(args.arm, ROOT / args.output_root, enable_sba_patch=args.arm == "treatment")
    print(f"SBA paired reduced {args.arm} completed")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create comparison gate**

Create `experiments/sba_validation/compare_paired_reduced_sba.py` with this content:

```python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


REQUIRED_ARM_FILES = ["run_status.json", "metrics.json", "task_manifest.json", "instrumentation.json", "config.json"]


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _nonfinite(value: Any, path: str) -> list[str]:
    failures: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            failures.extend(_nonfinite(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            failures.extend(_nonfinite(item, f"{path}[{index}]"))
    elif isinstance(value, (int, float)) and not math.isfinite(float(value)):
        failures.append(f"non-finite numeric value at {path}")
    return failures


def compare_paired_artifacts(root: Path, vram_threshold_mb: float = 7500.0, fail_invalid: bool = True) -> dict[str, Any]:
    failures: list[str] = []
    arms: dict[str, dict[str, Any]] = {}
    for arm in ["control", "treatment"]:
        arm_dir = root / arm
        payloads: dict[str, Any] = {}
        for name in REQUIRED_ARM_FILES:
            path = arm_dir / name
            if not path.exists():
                failures.append(f"{arm} missing required artifact: {name}")
            else:
                payloads[name] = _load(path)
        if len(payloads) == len(REQUIRED_ARM_FILES):
            arms[arm] = payloads
    if set(arms) == {"control", "treatment"}:
        for arm, payloads in arms.items():
            if payloads["run_status.json"].get("status") != "completed":
                failures.append(f"{arm} status is not completed")
            failures.extend(_nonfinite(payloads["metrics.json"], f"{arm}.metrics"))
            if float(payloads["metrics.json"].get("peak_vram_mb", math.inf)) > vram_threshold_mb:
                failures.append(f"{arm} peak_vram_mb exceeds {vram_threshold_mb:.0f}")
        if arms["control"]["task_manifest.json"] != arms["treatment"]["task_manifest.json"]:
            failures.append("task manifests differ across arms")
        for field in ["seed", "rank", "max_length", "epochs", "model_id"]:
            if arms["control"]["config.json"].get(field) != arms["treatment"]["config.json"].get(field):
                failures.append(f"paired config field differs: {field}")
        if arms["control"]["config.json"].get("git_revision") != arms["treatment"]["config.json"].get("git_revision"):
            failures.append("git revisions differ across arms")
        sba_patch = arms["treatment"]["instrumentation.json"].get("sba_patch", {})
        if int(sba_patch.get("patched_module_count", 0)) <= 0:
            failures.append("treatment SBA patch inactive")
    if failures:
        result = {"valid": False, "failures": failures}
        if fail_invalid:
            print(json.dumps(result, indent=2, allow_nan=False))
            raise SystemExit(2)
        return result
    control_metrics = arms["control"]["metrics.json"]
    treatment_metrics = arms["treatment"]["metrics.json"]
    delta_bwt_points = (treatment_metrics["bwt"] - control_metrics["bwt"]) * 100.0
    delta_avg_acc_points = (treatment_metrics["avg_acc"] - control_metrics["avg_acc"]) * 100.0
    old_task_delta_gaps_points = [
        (treatment - control) * 100.0
        for treatment, control in zip(treatment_metrics["old_task_deltas"], control_metrics["old_task_deltas"])
    ]
    return {
        "valid": True,
        "failures": [],
        "delta_bwt_points": delta_bwt_points,
        "delta_avg_acc_points": delta_avg_acc_points,
        "old_task_delta_gaps_points": old_task_delta_gaps_points,
        "control_peak_vram_mb": control_metrics["peak_vram_mb"],
        "treatment_peak_vram_mb": treatment_metrics["peak_vram_mb"],
        "vram_threshold_mb": vram_threshold_mb,
        "sba_patch_active": True,
        "continue": delta_bwt_points >= 1.5 and delta_avg_acc_points >= -2.0 and min(old_task_delta_gaps_points, default=0.0) >= -3.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare paired reduced SBA artifacts")
    parser.add_argument("--root", default="experiments/sba_validation/paired_reduced")
    parser.add_argument("--out", default="experiments/sba_validation/paired_reduced/comparison.json")
    parser.add_argument("--vram-threshold-mb", type=float, default=7500.0)
    args = parser.parse_args()
    result = compare_paired_artifacts(Path(args.root), args.vram_threshold_mb, fail_invalid=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Create general artifact validator CLI**

Create `experiments/sba_validation/validate_sba_artifacts.py` with this content:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.sba_validation.artifact_utils import collect_nonfinite_paths, validate_required_fields


REQUIRED_GATE_FIELDS = [
    "gate",
    "valid",
    "failure_reasons",
    "git_revision",
    "command",
    "seed",
    "formula_variant",
    "mask_policy",
    "dtype_policy",
    "checks",
    "log_path",
]


def validate_gate_artifact(path: Path) -> dict[str, object]:
    failures: list[str] = []
    if not path.exists():
        return {"valid": False, "failures": [f"missing artifact: {path}"]}
    payload = json.loads(path.read_text(encoding="utf-8"))
    failures.extend(validate_required_fields(payload, REQUIRED_GATE_FIELDS, str(path)))
    failures.extend([f"non-finite numeric value at {p}" for p in collect_nonfinite_paths(payload, str(path))])
    if payload.get("valid") is not True:
        failures.append(f"artifact valid field is not true: {path}")
    return {"valid": len(failures) == 0, "failures": failures, "artifact": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate SBA gate artifact JSON")
    parser.add_argument("artifact")
    args = parser.parse_args()
    result = validate_gate_artifact(Path(args.artifact))
    print(json.dumps(result, indent=2, allow_nan=False))
    if not result["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run CPU tests and syntax checks**

Run:

```cmd
python -m pytest tests\test_sba_validation_artifacts.py -k "paired_reduced_config or mismatched_manifest" -q && python -m py_compile experiments\sba_validation\run_paired_reduced_sba.py experiments\sba_validation\compare_paired_reduced_sba.py experiments\sba_validation\validate_sba_artifacts.py
```

Expected: exit code `0`, pytest summary:

```text
2 passed
```

Expected `py_compile`: no stdout and no stderr.

- [ ] **Step 7: Run paired reduced training only after CPU, CUDA synthetic, and tiny model gates pass plus explicit GPU approval**

Create output root:

```cmd
if not exist experiments\sba_validation\paired_reduced mkdir experiments\sba_validation\paired_reduced
```

Run fresh control:

```cmd
python experiments\sba_validation\run_paired_reduced_sba.py --arm control --output-root experiments\sba_validation\paired_reduced > experiments\sba_validation\paired_reduced\control.log 2>&1
```

Expected control gate: exit code `0`, `experiments\sba_validation\paired_reduced\control\run_status.json` has `status` equal to `completed`, all numeric metrics finite, and `peak_vram_mb <= 7500`.

Run fresh SBA treatment only after control passes:

```cmd
python experiments\sba_validation\run_paired_reduced_sba.py --arm treatment --output-root experiments\sba_validation\paired_reduced > experiments\sba_validation\paired_reduced\treatment.log 2>&1
```

Expected treatment gate: exit code `0`, `experiments\sba_validation\paired_reduced\treatment\run_status.json` has `status` equal to `completed`, metrics finite, `peak_vram_mb <= 7500`, and `instrumentation.json` contains `sba_patch.patched_module_count > 0`.

Compare only after both arms pass:

```cmd
python experiments\sba_validation\compare_paired_reduced_sba.py --root experiments\sba_validation\paired_reduced --out experiments\sba_validation\paired_reduced\comparison.json --vram-threshold-mb 7500
```

Expected pass output: JSON with `valid=true`, no failures, same manifest accepted, same seed accepted, treatment patch active, both peaks below or equal to `7500`, and a boolean `continue` decision. A `continue=false` result is valid evidence and is not a harness failure if all gates pass.

- [ ] **Step 8: Commit paired runner and comparison**

If GPU paired run was not approved yet:

```cmd
git add experiments\sba_validation\run_paired_reduced_sba.py experiments\sba_validation\compare_paired_reduced_sba.py experiments\sba_validation\validate_sba_artifacts.py tests\test_sba_validation_artifacts.py && git commit -m "feat: add sba paired reduced validation runner"
```

If approved paired run completed and passed gate validation:

```cmd
git add experiments\sba_validation\run_paired_reduced_sba.py experiments\sba_validation\compare_paired_reduced_sba.py experiments\sba_validation\validate_sba_artifacts.py experiments\sba_validation\paired_reduced tests\test_sba_validation_artifacts.py && git commit -m "feat: add sba paired reduced validation runner"
```

Expected: exit code `0`, with git reporting a focused commit.

### Task 10: Add reporting, result packet, and escalation decision guard

**Files:**
- Create: `experiments/sba_validation/README.md`
- Create: `experiments/sba_validation/write_result_critic_packet.py`
- Create: `experiments/sba_validation/RESULT_CRITIC_PACKET.md` after gate execution

- [ ] **Step 1: Create SBA validation README**

Create `experiments/sba_validation/README.md` with this content:

```markdown
# Symmetric Bipolar Attention Validation

This directory contains the bounded evidence ladder for SBA v1: `sba-v1-valid-position-sinh-cosh-shared-denominator`.

## Gate order

1. CPU math gate: `cpu_math_gate.json` and `cpu_math_gate.log`.
2. Synthetic CUDA gate: `cuda_synthetic_gate.json` and `cuda_synthetic_gate.log`, only after CPU gate passes and GPU approval is explicit.
3. Tiny patched-model gate: `tiny_model_gate.json` and `tiny_model_gate.log`, only after synthetic CUDA passes and GPU approval is explicit.
4. Paired reduced training gate: `paired_reduced/control`, `paired_reduced/treatment`, and `paired_reduced/comparison.json`, only after all earlier gates pass and GPU approval is explicit.
5. Larger-suite escalation: blocked unless all prior gates pass, the result critic accepts evidence quality, and the user explicitly approves escalation.

## Non-goals

- SBA is not the default attention path.
- SBA is not a catastrophic-forgetting solution claim.
- Full current4, v10, Digital Twin, generative subset, broad settings fuzzer, and larger-suite experiments remain blocked until escalation is approved.
```

- [ ] **Step 2: Create result packet writer**

Create `experiments/sba_validation/write_result_critic_packet.py` with this content:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"valid": False, "failure_reasons": [f"missing {path}"]}


def write_packet(root: Path, out: Path) -> str:
    cpu = _load(root / "cpu_math_gate.json")
    cuda = _load(root / "cuda_synthetic_gate.json")
    tiny = _load(root / "tiny_model_gate.json")
    comparison = _load(root / "paired_reduced" / "comparison.json")
    all_prior_gates_passed = all(payload.get("valid") is True for payload in [cpu, cuda, tiny, comparison])
    escalation_allowed = False
    lines = [
        "# SBA Validation Result Critic Packet",
        "",
        "## Gate verdicts",
        f"- CPU math gate valid: {cpu.get('valid')}",
        f"- Synthetic CUDA gate valid: {cuda.get('valid')}",
        f"- Tiny patched-model gate valid: {tiny.get('valid')}",
        f"- Paired reduced comparison valid: {comparison.get('valid')}",
        "",
        "## Hard criteria",
        f"- Formula equivalence max diff: {cpu.get('checks', {}).get('formula_equivalence_max_diff')}",
        f"- Masked leakage max: {cpu.get('checks', {}).get('max_masked_leakage')}",
        f"- Signed L1 max: {cpu.get('checks', {}).get('max_signed_l1')}",
        f"- Jacobian claim status: {cpu.get('checks', {}).get('jacobian_claim_status')}",
        f"- CUDA peak VRAM MB: {cuda.get('peak_vram_mb')}",
        f"- Tiny-model peak VRAM MB: {tiny.get('peak_vram_mb')}",
        f"- Paired treatment peak VRAM MB: {comparison.get('treatment_peak_vram_mb')}",
        f"- Paired delta BWT points: {comparison.get('delta_bwt_points')}",
        f"- Paired delta average accuracy points: {comparison.get('delta_avg_acc_points')}",
        "",
        "## Escalation decision",
        f"- All prior gates passed: {all_prior_gates_passed}",
        f"- Larger-suite escalation allowed now: {escalation_allowed}",
        "- Reason: larger-suite escalation still requires explicit user approval after result critic review even when all gates pass.",
        "",
        "## Required next decision",
        "Route this packet to result criticism. Do not claim catastrophic forgetting is solved.",
    ]
    text = "\n".join(lines) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Write SBA validation result critic packet")
    parser.add_argument("--root", default="experiments/sba_validation")
    parser.add_argument("--out", default="experiments/sba_validation/RESULT_CRITIC_PACKET.md")
    args = parser.parse_args()
    write_packet(Path(args.root), Path(args.out))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run syntax check**

Run:

```cmd
python -m py_compile experiments\sba_validation\write_result_critic_packet.py
```

Expected: exit code `0`, no stdout, and no stderr.

- [ ] **Step 4: Write packet after gate artifacts exist**

Run after all available gate artifacts are written:

```cmd
python experiments\sba_validation\write_result_critic_packet.py --root experiments\sba_validation --out experiments\sba_validation\RESULT_CRITIC_PACKET.md
```

Expected stdout:

```text
Wrote experiments\sba_validation\RESULT_CRITIC_PACKET.md
```

Hard reporting criteria: packet lists CPU, CUDA, tiny-model, and paired-reduced verdicts; records Jacobian claim status as `supported` or `falsified`; records VRAM peaks; blocks larger-suite escalation unless all gates pass and explicit approval is later granted.

- [ ] **Step 5: Commit reporting files**

Run:

```cmd
git add experiments\sba_validation\README.md experiments\sba_validation\write_result_critic_packet.py experiments\sba_validation\RESULT_CRITIC_PACKET.md && git commit -m "docs: add sba validation reporting packet"
```

Expected: exit code `0`, with git reporting a focused commit.

### Task 11: Run final CPU verification and preserve gate boundaries

**Files:**
- Verify: `cascades/math_ops.py`
- Verify: `cascades/config.py`
- Verify: `cascades/sba_patch.py`
- Verify: `train.py`
- Verify: `experiments/sba_validation/*.py`
- Verify: `tests/test_math.py`
- Verify: `tests/test_sba_patch.py`
- Verify: `tests/test_sba_validation_artifacts.py`

- [ ] **Step 1: Run bytecode compilation for touched Python files**

Run:

```cmd
python -m py_compile cascades\math_ops.py cascades\config.py cascades\sba_patch.py train.py experiments\sba_validation\artifact_utils.py experiments\sba_validation\run_cpu_math_gate.py experiments\sba_validation\run_cuda_synthetic_gate.py experiments\sba_validation\run_tiny_model_gate.py experiments\sba_validation\run_paired_reduced_sba.py experiments\sba_validation\compare_paired_reduced_sba.py experiments\sba_validation\validate_sba_artifacts.py experiments\sba_validation\write_result_critic_packet.py
```

Expected: exit code `0`, no stdout, and no stderr.

- [ ] **Step 2: Run CPU-only tests**

Run:

```cmd
python -m pytest tests\test_math.py tests\test_sba_patch.py tests\test_sba_validation_artifacts.py -q
```

Expected: exit code `0`, all selected tests pass, and no `FAILED` entries.

- [ ] **Step 3: Validate CPU gate artifact**

Run:

```cmd
python experiments\sba_validation\validate_sba_artifacts.py experiments\sba_validation\cpu_math_gate.json
```

Expected: exit code `0`, stdout JSON with:

```json
{
  "valid": true,
  "failures": []
}
```

- [ ] **Step 4: Verify GPU gates were not run without approval**

If no explicit approval exists for GPU gates, run no GPU commands. Record in the handoff that `cuda_synthetic_gate.json`, `tiny_model_gate.json`, and paired reduced training are pending approval. This is a valid implementation stop point after CPU artifacts and code scaffolding pass.

- [ ] **Step 5: Commit final CPU verification updates if files changed**

Run:

```cmd
git status --short
```

Expected before committing: only SBA validation files, tests, `train.py`, `cascades/math_ops.py`, `cascades/config.py`, `cascades/sba_patch.py`, and intended artifacts are listed. If unrelated files appear, stop and ask for review.

If only intended files appear:

```cmd
git add cascades\math_ops.py cascades\config.py cascades\sba_patch.py train.py experiments\sba_validation tests\test_math.py tests\test_sba_patch.py tests\test_sba_validation_artifacts.py && git commit -m "test: verify sba validation scaffold"
```

Expected: exit code `0`, with git reporting a focused commit or `nothing to commit` if every prior task already committed.

### Task 12: Update project context and hand off to result criticism or GPU approval decision

**Files:**
- Modify: `CONTEXT.md`

- [ ] **Step 1: Add context entry after implementation**

Add this entry near the top of `CONTEXT.md` under `## Recent Work`:

```markdown
### 2026-05-18 — Symmetric Bipolar Attention validation implementation scaffold
- Implemented the default-off SBA validation scaffold from `docs/superpowers/plans/2026-05-18-symmetric-bipolar-attention-validation.md` because the approved design requires a staged evidence ladder before any architecture replacement or larger-suite escalation.
- CPU evidence: `experiments/sba_validation/cpu_math_gate.json` validates formula equivalence, safe masking, signed L1 bound, finite outputs and gradients, and records the Jacobian claim as supported or falsified. CPU tests for `tests/test_math.py`, `tests/test_sba_patch.py`, and `tests/test_sba_validation_artifacts.py` pass.
- Gate boundary: synthetic CUDA, tiny patched-model, and paired reduced training gates require explicit GPU approval and must remain blocked unless prior gate artifacts are valid.
- Decision: SBA remains default-off and not a catastrophic-forgetting solution claim. Larger-suite escalation remains blocked pending result critic review and explicit user approval.
```

- [ ] **Step 2: Run markdown-only inspection**

Run:

```cmd
git diff -- CONTEXT.md
```

Expected: diff contains only the new SBA implementation scaffold entry and any frontmatter status update chosen by the implementer.

- [ ] **Step 3: Commit context update**

Run:

```cmd
git add CONTEXT.md && git commit -m "docs: record sba validation scaffold status"
```

Expected: exit code `0`, with git reporting a focused context commit.

---

## Final self-review checklist for the future implementer

- [ ] Spec coverage: every approved gate is represented by a task: CPU math, CUDA synthetic, patched model integration, paired reduced training, larger-suite escalation decision, reporting artifacts, and final context update.
- [ ] Hard criteria coverage: formula equivalence, safe masking, signed L1 bound, finite outputs and gradients, Jacobian measurement, synthetic CUDA stability, tiny patched-model forward/backward, same-manifest paired reduced training, under-7500 MB VRAM, finite metrics, and durable artifacts are each enforced by a task or gate.
- [ ] Banned phrase scan: the implementation files and result packet use concrete behavior, concrete commands, and concrete failure strings.
- [ ] Type and signature consistency: `enable_symmetric_bipolar_attention`, `sba_formula_variant`, `enable_sba_patch`, `sba_patch_report_path`, `SBA_FORMULA_VARIANT`, `sba_optimized_attention()`, and `apply_sba_patch()` use the same names across tests, config, patch helper, training, and gate scripts.
- [ ] Default behavior: no existing CASCADES training call changes behavior unless `enable_sba_patch=True` or the dedicated SBA paired runner treatment arm is used.
- [ ] Approval boundary: GPU commands are present for future execution but must not run until all earlier gates pass and the user approves the specific GPU stage.

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-18-symmetric-bipolar-attention-validation.md`. Two execution options:

1. **Subagent-Driven (recommended)** - Dispatch a fresh subagent per task, review between tasks, and preserve strict gate boundaries.
2. **Inline Execution** - Execute tasks in this session using `superpowers:executing-plans`, batching only CPU-safe steps until GPU approval is explicit.

Recommended next mode: `code`, starting at Task 1 and stopping after CPU scaffolding unless the user explicitly approves GPU gate execution.
