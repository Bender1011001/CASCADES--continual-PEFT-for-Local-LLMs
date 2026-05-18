# CF-cycle-2 Reasoning3 Guardrails Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Repair hard runtime and comparison guardrails so a reduced-memory `reasoning3` null-space feasibility ablation can produce finite, comparable control/treatment evidence without using the invalid `current4` artifact set.

**Architecture:** This cycle is code-first, then experiment-design. Add opt-in hard guardrails in the training path, make the direct wrapper persist failed-run status instead of letting invalid runs masquerade as evidence, and make comparison reject missing, non-finite, over-threshold, or projection-inactive arms.

**Tech Stack:** Python, PyTorch, NumPy, pytest, Windows cmd, RTX 4060 Ti 8GB, the existing CASCADES direct wrapper.

---

## Cycle Question

Can the existing direct null-space ablation wrapper be made safe enough to run a reduced-memory `reasoning3` control/treatment feasibility ablation at rank 4 and max length 256, such that both arms either complete with finite metrics below 7500 MB peak VRAM or fail fast with durable failed-run evidence?

## Handoff Target Decision

- **Next mode:** `code`
- **Reason:** CF-cycle-1 already produced the relevant algorithmic hypothesis and experiment design seed, but the current code only warns on VRAM threshold breaches, has no hard non-finite loss abort, and the comparison script accepts invalid control metrics. A new hypothesis-generation step would not produce actionable evidence until these guardrails are implemented.
- **Return path after code:** hand back to `llm-experiment-designer` for CPU-light verification review and explicit reduced-memory GPU launch commands. Do not launch GPU work from code mode unless explicitly approved.

## Known Evidence

- CF-cycle-1 report classifies the `current4` control run as diagnostic only because it violated the audit checkpoint, breached the 7500 MB VRAM guardrail, reached `loss=nan`, and lacks completed metrics/instrumentation.
- The lighter task suite is `reasoning3`, defined as GSM8K, ARC, and CSQA, excluding the large Digital Twin task.
- The direct wrapper mutates task-suite globals before importing training code and is the intended wrapper for this ablation, but it does not currently enforce hard finite-loss or hard VRAM aborts.
- The comparison script currently enforces peak VRAM only for treatment and does not reject non-finite metrics.

## Evidence Sources for This Cycle

- `experiments/cf_cycle_1/REPORT.md`
- `experiments/cf_cycle_1/result_critic_packet.md`
- `experiments/cf_cycle_1/run_nullspace_ablation.py`
- `experiments/cf_cycle_1/compare_nullspace_ablation.py`
- `experiments/cf_cycle_1/harness_audit.log`
- `train.py`
- `cascades/vram_monitor.py`
- `tests/test_data.py`
- New guardrail unit tests under `tests/`

## Preconditions Before Any GPU Run

1. CPU-light syntax and tests pass after code changes.
2. Training aborts immediately on non-finite training loss or non-finite evaluation loss when the wrapper enables guardrails.
3. Training aborts or the wrapper marks the run failed if any observed peak VRAM exceeds 7500 MB, including checkpoints that happen before training resets peak CUDA stats.
4. Wrapper writes durable failed-run evidence such as `run_status.json` when a guardrail abort happens, and failed runs must not be treated as completed comparable arms.
5. Comparison validates both arms, rejects missing files, rejects non-finite metrics, rejects either arm above 7500 MB, and rejects treatment with inactive projection evidence.
6. Treatment projection evidence remains mandatory: frozen bases become non-empty, `_cllora_reassign()` receives a frozen basis, and `removed_norm_sum` is positive.

## Success, Failure, and Inconclusive Criteria

- **Development success:** guardrail implementation passes CPU-light tests and produces deterministic rejection behavior for synthetic invalid metrics/instrumentation.
- **Feasibility success after approved GPU work:** both reduced-memory `reasoning3` arms complete with finite `metrics.json`, finite `instrumentation.json`, peak VRAM at or below 7500 MB, matching task manifests, and treatment projection evidence active.
- **Algorithmic treatment success after feasibility only:** treatment BWT is at least +1.5 points better than control, final average accuracy is no more than 2 points worse, and no old-task delta gap is worse than -3 points.
- **Failure:** any arm hits a hard non-finite loss abort, exceeds 7500 MB, fails to write required artifacts, or comparison rejects the run.
- **Inconclusive:** code guardrails pass but GPU is unavailable or interrupted before both arms complete; treatment completes but projection evidence is inactive, which is a harness failure rather than a null-space falsification.

## File Structure

- Modify `train.py`: add opt-in hard guardrail checks to `train_cascades()` without changing default behavior for normal training calls.
- Modify `experiments/cf_cycle_1/run_nullspace_ablation.py`: add CLI guardrail options, pass guardrail settings into training, track all VRAM checkpoints, catch guardrail failures, and persist `run_status.json`.
- Modify `experiments/cf_cycle_1/compare_nullspace_ablation.py`: validate both arms before comparison and exit nonzero on invalid evidence.
- Create `tests/test_cf_cycle_2_guardrails.py`: CPU-only tests for finite checking, VRAM status validation, failed-run status persistence, and comparison rejection of invalid arms.
- Create or update `experiments/cf_cycle_2/README.md`: document code-first preconditions and approved reduced-memory commands after guardrail verification.
- Update `CONTEXT.md`: record what changed and why at the end of the code task.

### Task 1: Add CPU-only guardrail tests

**Files:**
- Create: `tests/test_cf_cycle_2_guardrails.py`
- Read: `experiments/cf_cycle_1/run_nullspace_ablation.py`
- Read: `experiments/cf_cycle_1/compare_nullspace_ablation.py`

- [ ] **Step 1: Create tests for finite-value validation and arm rejection**

Create `tests/test_cf_cycle_2_guardrails.py` with tests that build temporary `metrics.json` and `instrumentation.json` files for control and treatment. Include at least these cases:

```python
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


def populate_valid_run(root: Path, control_peak: float = 7000.0, treatment_peak: float = 7000.0) -> None:
    write_json(root / "control" / "metrics.json", finite_metrics("control", control_peak))
    write_json(root / "control" / "instrumentation.json", active_instrumentation())
    write_json(root / "treatment" / "metrics.json", finite_metrics("treatment", treatment_peak))
    write_json(root / "treatment" / "instrumentation.json", active_instrumentation())


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


def test_comparison_rejects_inactive_projection(tmp_path: Path) -> None:
    populate_valid_run(tmp_path)
    instr = active_instrumentation()
    instr["reassign"]["calls_with_frozen_basis"] = 0
    instr["reassign"]["removed_norm_sum"] = 0.0
    write_json(tmp_path / "treatment" / "instrumentation.json", instr)

    with pytest.raises(SystemExit) as excinfo:
        compare.compare_runs(tmp_path, vram_threshold_mb=7500.0, fail_invalid=True)

    assert excinfo.value.code == 2
```

- [ ] **Step 2: Run tests and verify they fail before implementation**

Run: `python -m pytest tests/test_cf_cycle_2_guardrails.py -q`

Expected before implementation: tests fail because `compare_runs()` does not exist or invalid evidence is not rejected.

### Task 2: Add opt-in hard guardrails to training

**Files:**
- Modify: `train.py`

- [ ] **Step 1: Add explicit guardrail exception helpers**

Add helpers near the main training function:

```python
class TrainingGuardrailViolation(RuntimeError):
    """Raised when an opt-in training safety guardrail is violated."""


def _raise_if_nonfinite_tensor(value: torch.Tensor, label: str) -> None:
    if not torch.isfinite(value.detach()).all().item():
        raise TrainingGuardrailViolation(f"non-finite {label}")


def _raise_if_vram_over_threshold(threshold_mb: float | None, device: str, label: str) -> None:
    if threshold_mb is None or not torch.cuda.is_available():
        return
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    if peak_mb > threshold_mb:
        raise TrainingGuardrailViolation(
            f"peak VRAM {peak_mb:.0f} MB exceeded threshold {threshold_mb:.0f} MB at {label}"
        )
```

- [ ] **Step 2: Extend `train_cascades()` signature with default-off guardrails**

Change the signature to include:

```python
    abort_on_nonfinite: bool = False,
    vram_threshold_mb: float | None = None,
```

Keep defaults backward compatible.

- [ ] **Step 3: Check training loss before backward**

Immediately after `loss = outputs.loss`, add:

```python
                if abort_on_nonfinite:
                    _raise_if_nonfinite_tensor(loss, f"training loss task={t} epoch={ep + 1} batch={num_batches + 1}")
```

- [ ] **Step 4: Check VRAM during training**

After `optimizer.step()` and after existing `log_vram()` checkpoints, call:

```python
                _raise_if_vram_over_threshold(vram_threshold_mb, device, f"task={t} epoch={ep + 1} batch={num_batches}")
```

Also call the helper after model load, after adapter injection, after first backward logging, after sleep, and after evaluation cache clearing.

- [ ] **Step 5: Check evaluation loss before `math.exp()`**

Immediately after `out = model(...)`, add:

```python
                    if abort_on_nonfinite:
                        _raise_if_nonfinite_tensor(out.loss, f"eval loss train_task={t} eval_task={eval_t} batch={n_batches + 1}")
```

After computing `avg_loss`, add:

```python
                if abort_on_nonfinite and not math.isfinite(avg_loss):
                    raise TrainingGuardrailViolation(f"non-finite average eval loss train_task={t} eval_task={eval_t}")
```

- [ ] **Step 6: Run syntax checks**

Run: `python -m py_compile train.py`

Expected: exits 0.

### Task 3: Make the direct wrapper persist pass/fail status

**Files:**
- Modify: `experiments/cf_cycle_1/run_nullspace_ablation.py`

- [ ] **Step 1: Add CLI guardrail flags**

Add arguments:

```python
    parser.add_argument("--vram-threshold-mb", type=float, default=7500.0)
    parser.add_argument("--allow-nonfinite", action="store_true")
    parser.add_argument("--allow-vram-over-threshold", action="store_true")
```

- [ ] **Step 2: Pass guardrails into training**

Update the `train.train_cascades(...)` call to include:

```python
            abort_on_nonfinite=not args.allow_nonfinite,
            vram_threshold_mb=None if args.allow_vram_over_threshold else args.vram_threshold_mb,
```

- [ ] **Step 3: Track the true maximum observed VRAM**

Update wrapper instrumentation so `stats["vram"]` records every `log_vram()` checkpoint and add a helper that computes the maximum across checkpoint `max_allocated_mb` values and final CUDA peak. Use this maximum for persisted `metrics["peak_vram_mb"]`.

- [ ] **Step 4: Catch guardrail failures and write `run_status.json`**

Wrap the training call so a guardrail failure writes a JSON file like:

```json
{
  "arm": "control",
  "status": "failed_guardrail",
  "reason": "non-finite training loss task=2 epoch=1 batch=37",
  "task_suite": "reasoning3",
  "seed": 42,
  "rank": 4,
  "max_length": 256,
  "vram_threshold_mb": 7500.0
}
```

Then re-raise or exit nonzero so automation cannot mistake the run for a success.

- [ ] **Step 5: Write successful `run_status.json`**

After metrics and instrumentation are written, write:

```json
{
  "arm": "treatment",
  "status": "completed",
  "task_suite": "reasoning3",
  "seed": 42,
  "rank": 4,
  "max_length": 256,
  "vram_threshold_mb": 7500.0
}
```

- [ ] **Step 6: Run syntax checks**

Run: `python -m py_compile experiments\cf_cycle_1\run_nullspace_ablation.py`

Expected: exits 0.

### Task 4: Make comparison reject invalid evidence

**Files:**
- Modify: `experiments/cf_cycle_1/compare_nullspace_ablation.py`

- [ ] **Step 1: Refactor comparison into testable functions**

Add `compare_runs(root: Path, vram_threshold_mb: float = 7500.0, fail_invalid: bool = True) -> dict` and keep `main()` as a CLI wrapper.

- [ ] **Step 2: Validate required artifacts before arithmetic**

Require these files for both arms:

```text
metrics.json
instrumentation.json
run_status.json
```

Reject if any are missing, if `run_status.json` is not `completed`, or if JSON cannot be parsed.

- [ ] **Step 3: Reject non-finite metrics recursively**

Implement a recursive helper that rejects `nan`, `inf`, and `-inf` in every numeric field in `metrics.json` and in numeric instrumentation values used by comparison.

- [ ] **Step 4: Enforce VRAM threshold for both arms**

Reject when either `control["peak_vram_mb"]` or `treatment["peak_vram_mb"]` is greater than `vram_threshold_mb`.

- [ ] **Step 5: Preserve projection evidence requirements**

Reject treatment unless `calls_with_frozen_basis > 0`, `removed_norm_sum > 0.0`, and at least one freeze event has nonzero frozen columns.

- [ ] **Step 6: Emit invalid comparison JSON and exit 2**

When invalid, print JSON containing `valid: false` and `failures: [...]`, then exit 2. When valid, print comparison JSON with `valid: true` and the existing deltas plus `continue`.

- [ ] **Step 7: Run tests**

Run: `python -m pytest tests/test_cf_cycle_2_guardrails.py -q`

Expected: all guardrail tests pass.

### Task 5: Document the reduced-memory CF-cycle-2 runbook

**Files:**
- Create: `experiments/cf_cycle_2/README.md`
- Modify: `CONTEXT.md`

- [ ] **Step 1: Create CF-cycle-2 runbook**

Document that GPU commands are blocked until CPU-light verification passes. Include commands for later explicit approval only:

```cmd
if not exist experiments\cf_cycle_2\nullspace_ablation mkdir experiments\cf_cycle_2\nullspace_ablation
python experiments\cf_cycle_1\run_nullspace_ablation.py --arm control --task-suite reasoning3 --seed 42 --epochs 2 --rank 4 --max-length 256 --output-root experiments\cf_cycle_2\nullspace_ablation --vram-threshold-mb 7500 > experiments\cf_cycle_2\nullspace_ablation\control.log 2>&1
python experiments\cf_cycle_1\run_nullspace_ablation.py --arm treatment --task-suite reasoning3 --seed 42 --epochs 2 --rank 4 --max-length 256 --output-root experiments\cf_cycle_2\nullspace_ablation --vram-threshold-mb 7500 > experiments\cf_cycle_2\nullspace_ablation\treatment.log 2>&1
python experiments\cf_cycle_1\compare_nullspace_ablation.py --root experiments\cf_cycle_2\nullspace_ablation --vram-threshold-mb 7500 > experiments\cf_cycle_2\nullspace_ablation\comparison.json
```

- [ ] **Step 2: Add context log**

Append a short entry to `CONTEXT.md` that states guardrails were repaired for CF-cycle-2 and why the loop is using reduced-memory `reasoning3` before any return to `current4`.

### Task 6: Final CPU-light verification for code handoff

**Files:**
- Verify: `train.py`
- Verify: `experiments/cf_cycle_1/run_nullspace_ablation.py`
- Verify: `experiments/cf_cycle_1/compare_nullspace_ablation.py`
- Verify: `tests/test_data.py`
- Verify: `tests/test_cf_cycle_2_guardrails.py`

- [ ] **Step 1: Run syntax verification**

Run:

```cmd
python -m py_compile train.py experiments\cf_cycle_1\run_nullspace_ablation.py experiments\cf_cycle_1\compare_nullspace_ablation.py
```

Expected: exits 0.

- [ ] **Step 2: Run relevant tests**

Run:

```cmd
python -m pytest tests/test_data.py tests/test_cf_cycle_2_guardrails.py -q
```

Expected: all selected tests pass.

- [ ] **Step 3: Required code-mode output**

Return a concise packet with files changed, verification commands and outputs, whether GPU remains blocked or explicitly ready for experiment-design approval, and a warning not to use the invalid CF-cycle-1 `current4` artifacts.

## Next Handoff Packet for Code Mode

**Objective:** Implement CF-cycle-2 guardrail repair so the reduced-memory `reasoning3` ablation can produce either finite comparable evidence or durable failed-run evidence.

**Constraints:** Do not launch GPU treatment or reuse the invalid CF-cycle-1 `current4` artifact set. Keep defaults backward compatible for normal training. Stay inside Roo approvals and file restrictions.

**Required behavior:** hard abort on non-finite training/eval loss; hard abort or failed-run marking when any observed peak VRAM is above 7500 MB; comparison rejects missing, failed, non-finite, over-threshold, or projection-inactive arms; treatment projection evidence remains mandatory.

**Required output:** patched files, CPU verification evidence, and a clear ready/not-ready verdict for the experiment designer. Do not claim catastrophic forgetting is solved.
