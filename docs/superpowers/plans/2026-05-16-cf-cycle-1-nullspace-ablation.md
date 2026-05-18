# CF-Cycle-1 Frozen Null-Space Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the first CF-cycle-1 null-space retention experiment executable, comparable, and evidence-rich enough to decide whether frozen null-space protection reduces backward-transfer variance and old-task loss.

**Architecture:** Run a CPU-light harness audit before GPU time, then execute a one-seed two-arm controlled ablation with the same task suite, order, seed, model, rank, max length, epochs, sleep setting, and non-target ablation flags. Add a small wrapper around [`train_cascades()`](train.py:171) to persist config/task manifests, metrics, VRAM, wall time, and projection/basis instrumentation without modifying core adapter math.

**Tech Stack:** Python, PyTorch, Transformers, pytest, Windows `cmd.exe`, JSON/CSV/NPY artifacts.

---

## Read-only evidence already collected by experiment designer

### CPU audit command 1: task definition tests

Command:

```cmd
python -m pytest tests/test_data.py -q
```

Observed output summary:

```text
collected 11 items
tests\test_data.py F....F..... [100%]
FAILED tests/test_data.py::TestTaskDefinitions::test_correct_number_of_tasks
AssertionError: CASCADES evaluates 3 sequential continual learning tasks
assert 4 == 3
FAILED tests/test_data.py::TestTaskDefinitions::test_task_files_have_expected_names
AssertionError: Task 0 file should contain 'task0' in name: data/task2_csqa_cot.jsonl
assert 'task0' in 'task2_csqa_cot.jsonl'
2 failed, 9 passed in 2.32s
```

Interpretation: the no-GPU precondition is not satisfied. [`tests/test_data.py`](tests/test_data.py:20) asserts a 3-task suite while [`TASK_FILES`](cascades/data.py:35) currently defines a 4-task suite and intentionally reorders tasks.

### CPU audit command 2: signature and runner pass-through

Command:

```cmd
python -c "import inspect, json; import cascades.data as d; import train; import research_runner; from experiment_matrix import DEFAULT_TASK_FILES; print('TASK_FILES=', json.dumps(d.TASK_FILES)); print('NUM_TASKS=', d.NUM_TASKS); print('DEFAULT_TASK_FILES=', json.dumps(DEFAULT_TASK_FILES)); print('train_cascades_sig=', inspect.signature(train.train_cascades)); src=inspect.getsource(research_runner.ExperimentRunner._run_cascades); print('run_cascades_has_rank_kwarg=', '\"rank\"' in src or '\'rank\'' in src); print('run_cascades_has_max_length_kwarg=', 'max_length' in src); print('run_cascades_pops_task_files=', 'task_files = overrides.pop(\"task_files\", None)' in src); print('run_cascades_calls_train_cascades=', 'accuracy_matrix = train_cascades(**kwargs)' in src)"
```

Observed output:

```text
TASK_FILES= ["data/task2_csqa_cot.jsonl", "data/task1_arc_cot.jsonl", "data/task0_gsm8k_cot.jsonl", "data/task3_digital_twin.jsonl"]
NUM_TASKS= 4
DEFAULT_TASK_FILES= ["data/task0_gsm8k_cot.jsonl", "data/task1_arc_cot.jsonl", "data/task2_csqa_cot.jsonl"]
train_cascades_sig= (seed: 'int' = 42, dmole_threshold: 'float' = 0.22, model_id: 'str' = 'p-e-w/Qwen3-4B-Instruct-2507-heretic', output_prefix: 'str' = 'cascades_v10', lr_liquid: 'float' = 0.002, lr_gate: 'float' = 0.0005, lr_funlora: 'float' = 5e-05, lr_riemannian: 'float' = 0.005, epochs: 'int' = 2, rank: 'int' = 8, max_length: 'int' = 384, num_samples: 'int' = 150, eval_em: 'bool' = False, enable_sleep: 'bool' = True, config: 'AblationConfig' = AblationConfig(...)) -> 'np.ndarray'
run_cascades_has_rank_kwarg= False
run_cascades_has_max_length_kwarg= False
run_cascades_pops_task_files= True
run_cascades_calls_train_cascades= True
```

Interpretation: [`research_runner.ExperimentRunner._run_cascades()`](research_runner.py:366) should not be used for the controlled ablation until fixed or bypassed. It drops [`task_files`](research_runner.py:376), does not forward `rank`, and does not forward `max_length`, even though [`train_cascades()`](train.py:171) accepts those parameters.

### CPU audit command 3: task manifest and revision

Command:

```cmd
python -c "import json, pathlib, subprocess; from cascades.data import TASK_FILES, NUM_TASKS; rows=[{'task_index':i,'path':p,'exists':pathlib.Path(p).exists(),'examples':(sum(1 for line in pathlib.Path(p).open(encoding='utf-8') if line.strip()) if pathlib.Path(p).exists() else None),'bytes':(pathlib.Path(p).stat().st_size if pathlib.Path(p).exists() else None)} for i,p in enumerate(TASK_FILES)]; print('git_revision=', subprocess.check_output(['git','rev-parse','--short','HEAD'], text=True).strip()); print('num_tasks=', NUM_TASKS); print('task_manifest=', json.dumps(rows, indent=2))"
```

Observed output:

```text
git_revision= 7c4e01d
num_tasks= 4
task_manifest= [
  {"task_index": 0, "path": "data/task2_csqa_cot.jsonl", "exists": true, "examples": 150, "bytes": 85370},
  {"task_index": 1, "path": "data/task1_arc_cot.jsonl", "exists": true, "examples": 150, "bytes": 122051},
  {"task_index": 2, "path": "data/task0_gsm8k_cot.jsonl", "exists": true, "examples": 150, "bytes": 88086},
  {"task_index": 3, "path": "data/task3_digital_twin.jsonl", "exists": true, "examples": 159, "bytes": 28358013}
]
```

Interpretation: the effective current code-state suite is 4 tasks in this order: CommonsenseQA, ARC, GSM8K, Digital Twin. The Digital Twin file is much larger by bytes but has 159 JSONL examples, so the run is still bounded by examples and `max_length`.

### CPU audit command 4: `num_samples` usage

Command:

```cmd
python -c "from pathlib import Path; import re; text=Path('train.py').read_text(encoding='utf-8-sig'); print([m.start() for m in re.finditer('num_samples', text)])"
```

Expected output after implementation audit:

```text
[<positions only in signature/docstring>]
```

Observed by source search: [`num_samples`](train.py:183) appears in the signature and docstring, but no call to [`prepare_data()`](cascades/data.py:80) passes a sample limit. It must not be part of the experiment contract unless code mode wires it through.

---

## Experiment design decisions

1. **No GPU until harness mismatch is acknowledged in artifacts.** The failing [`tests/test_data.py`](tests/test_data.py:20) audit triggers the stop/debug rule. The GPU run below is designed but should not be launched through [`research_runner.py`](research_runner.py:1).
2. **Official CF-cycle-1 controlled ablation should use the current 4-task suite first.** Rationale: this is the actual effective path of [`train_cascades()`](train.py:171) today, includes the project’s Digital Twin forgetting target, and avoids silently comparing a 3-task historical benchmark to a 4-task current project. Historical 3-task reproduction remains a later comparability run after task-suite selection is first-class.
3. **The original 3-task reasoning suite must be kept available as `reasoning3`, not silently conflated with `current4`.** The wrapper should expose `--task-suite current4` and `--task-suite reasoning3` so later critics can request historical comparability without editing globals by hand.
4. **Use a direct wrapper, not [`research_runner.ExperimentRunner._run_cascades()`](research_runner.py:366), for CF-cycle-1.** The runner does not currently forward `rank`, `max_length`, or `task_files` overrides, so it can generate false evidence.
5. **Keep [`enable_cllora_reassign`](cascades/config.py:32) disabled in both arms.** [`_cllora_reassign()`](cascades/adapters.py:176) still applies strict frozen-basis projection before returning when [`enable_cllora_reassign`](cascades/config.py:32) is false, so the treatment isolates strict frozen protection from active-sketch energy reassignment.

---

## Experiment 1: CPU-light harness comparability audit

**Purpose:** Confirm the experiment will run the intended tasks, flags, lengths, ranks, and logging path before GPU time.

**Artifacts to persist:**

- `experiments/cf_cycle_1/harness_audit.log`
- `experiments/cf_cycle_1/audit_snapshot.json`
- `experiments/cf_cycle_1/task_manifest_current4.json`
- `experiments/cf_cycle_1/task_manifest_reasoning3.json`

**Commands after code-mode implementation:**

```cmd
if not exist experiments\cf_cycle_1 mkdir experiments\cf_cycle_1
python experiments\cf_cycle_1\harness_audit.py > experiments\cf_cycle_1\harness_audit.log 2>&1
python -m pytest tests/test_data.py -q > experiments\cf_cycle_1\pytest_test_data.log 2>&1
```

**Pass criteria:**

- `audit_snapshot.json` records git revision, Python version, CUDA availability, current suite, reasoning3 suite, [`train_cascades()`](train.py:171) signature, [`research_runner.ExperimentRunner._run_cascades()`](research_runner.py:366) override behavior, and [`num_samples`](train.py:183) status.
- [`tests/test_data.py`](tests/test_data.py:20) either passes after task-suite-aware test fixes, or its failure is explicitly recorded as a known docs/test drift blocker.
- No GPU command runs unless `audit_snapshot.json` says `gpu_preconditions.ok` is true.

**Falsifies or weakens hypothesis 2 if:**

- No task/config drift is found and all pass-through paths are verified, yet BWT variance remains large under fixed one-seed controls.

**Supports hypothesis 2 if:**

- Task count/order drift, dropped overrides, unused sample controls, or missing metric artifacts are observed. The current read-only evidence already supports this.

---

## Experiment 2: one-seed two-arm frozen null-space ablation

**Purpose:** Test whether strict frozen null-space projection improves retention under identical current 4-task conditions.

**Shared run contract:**

- Model: `p-e-w/Qwen3-4B-Instruct-2507-heretic`
- Task suite: `current4`
- Task order: `data/task2_csqa_cot.jsonl`, `data/task1_arc_cot.jsonl`, `data/task0_gsm8k_cot.jsonl`, `data/task3_digital_twin.jsonl`
- Seed: `42`
- Rank: `8`
- Max length: `384`
- Epochs per task: `2`
- [`enable_sleep`](train.py:185): `True`
- Proxy metrics are primary; generative exact match is secondary and should be skipped in the first GPU pass unless the wrapper exposes a fixed 5-example subset.

**Control arm config:**

```python
from cascades.config import AblationConfig

CONTROL_CONFIG = AblationConfig(
    enable_paca=True,
    enable_deal=True,
    enable_gainlora_gate=True,
    enable_coso_nullspace=False,
    enable_cllora_reassign=False,
    enable_svc=True,
    enable_dmole_select=True,
    enable_funlora=True,
    gqa_ratio=1.0,
    ear_gamma=1e-4,
    enable_soft_ear=True,
    enable_principal_expansion=True,
    cfg_lambda=1.5,
    enable_ambient_dedup=True,
)
```

**Treatment arm config:**

```python
from cascades.config import AblationConfig

TREATMENT_CONFIG = AblationConfig(
    enable_paca=True,
    enable_deal=True,
    enable_gainlora_gate=True,
    enable_coso_nullspace=True,
    enable_cllora_reassign=False,
    enable_svc=True,
    enable_dmole_select=True,
    enable_funlora=True,
    gqa_ratio=1.0,
    ear_gamma=1e-4,
    enable_soft_ear=True,
    enable_principal_expansion=True,
    cfg_lambda=1.5,
    enable_ambient_dedup=True,
)
```

**Commands after code-mode implementation:**

```cmd
if not exist experiments\cf_cycle_1\nullspace_ablation mkdir experiments\cf_cycle_1\nullspace_ablation
python experiments\cf_cycle_1\run_nullspace_ablation.py --arm control --task-suite current4 --seed 42 --epochs 2 --rank 8 --max-length 384 --output-root experiments\cf_cycle_1\nullspace_ablation > experiments\cf_cycle_1\nullspace_ablation\control.log 2>&1
python experiments\cf_cycle_1\run_nullspace_ablation.py --arm treatment --task-suite current4 --seed 42 --epochs 2 --rank 8 --max-length 384 --output-root experiments\cf_cycle_1\nullspace_ablation > experiments\cf_cycle_1\nullspace_ablation\treatment.log 2>&1
python experiments\cf_cycle_1\compare_nullspace_ablation.py --root experiments\cf_cycle_1\nullspace_ablation > experiments\cf_cycle_1\nullspace_ablation\comparison.json
```

**Artifacts to persist per arm:**

- `experiments/cf_cycle_1/nullspace_ablation/<arm>/config.json`
- `experiments/cf_cycle_1/nullspace_ablation/<arm>/task_manifest.json`
- `experiments/cf_cycle_1/nullspace_ablation/<arm>/cascades_results.csv`
- `experiments/cf_cycle_1/nullspace_ablation/<arm>/accuracy_matrix.npy`
- `experiments/cf_cycle_1/nullspace_ablation/<arm>/metrics.json`
- `experiments/cf_cycle_1/nullspace_ablation/<arm>/instrumentation.json`
- `experiments/cf_cycle_1/nullspace_ablation/<arm>/cascades_weights.pt`
- `experiments/cf_cycle_1/nullspace_ablation/<arm>.log`

**Required metrics in `metrics.json`:**

```json
{
  "arm": "treatment",
  "task_suite": "current4",
  "seed": 42,
  "rank": 8,
  "max_length": 384,
  "epochs": 2,
  "avg_acc": 0.0,
  "bwt": 0.0,
  "final_accs": [0.0, 0.0, 0.0, 0.0],
  "diagonal_accs": [0.0, 0.0, 0.0, 0.0],
  "old_task_deltas": [0.0, 0.0, 0.0],
  "peak_vram_mb": 0.0,
  "wall_time_s": 0.0,
  "git_revision": "7c4e01d"
}
```

**Required projection/basis evidence in `instrumentation.json`:**

```json
{
  "ear_events": [
    {"call_index": 1, "adapters": 1, "ear_initialized_before": 0, "ear_initialized_after": 1, "max_q_null_u_cols": 4, "max_q_null_v_cols": 4}
  ],
  "freeze_events": [
    {"task_boundary_call": 1, "u_cols_before": 0, "u_cols_after": 1, "v_cols_before": 0, "v_cols_after": 1, "adapter_rank": 8}
  ],
  "reassign": {
    "calls_total": 0,
    "calls_with_null_sketch": 0,
    "calls_with_frozen_basis": 0,
    "max_frozen_cols": 0,
    "removed_norm_sum": 0.0
  },
  "vram": [
    {"tag": "after_model_load", "allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}
  ]
}
```

**Continue rule:**

- Treatment BWT minus control BWT is at least `+0.015`.
- Treatment final average proxy ACC is not more than `0.02` below control.
- No treatment old-task delta is worse than the corresponding control old-task delta by more than `0.03`.
- Treatment peak VRAM is below `7500` MB.
- Treatment instrumentation shows non-empty frozen basis after at least one task boundary.
- Treatment instrumentation shows at least one [`_cllora_reassign()`](cascades/adapters.py:176) call with a non-empty frozen basis and positive `removed_norm_sum`.

**Stop/debug rule:**

- Treatment BWT is equal to or worse than control.
- [`ear_initialized`](cascades/injection.py:263) never becomes true in treatment.
- [`frozen_null_basis`](cascades/adapters.py:291) remains empty after a task boundary in treatment.
- Treatment diagonal accuracy on the current/new task drops more than `0.03` versus control.
- Final average proxy ACC drops more than `0.02` to `0.03` versus control.
- Task/config mismatch remains unresolved or unrecorded.
- Peak VRAM exceeds `7500` MB.

**Falsification mapping:**

- Hypothesis 1 is falsified only if treatment projection is proven active and treatment does not improve BWT or old-task deltas versus control. If projection gates are inactive, the result is an implementation/harness failure, not an algorithmic falsification.
- Hypothesis 2 is supported by the current audit because task-count drift and runner override drops were observed. It is weakened only after code-mode fixes pass and variance remains large under logged comparable runs.
- Hypothesis 3 is supported if treatment improves BWT but loses more than `0.03` diagonal/new-task proxy ACC or basis occupancy grows aggressively toward the cap. It is weakened if BWT improves without plasticity loss and occupancy stays modest.
- Hypothesis 4 is inconclusive in the first GPU pass unless the optional fixed generative subset is added.

---

## Code-mode implementation tasks

### Task 1: Add CPU-light harness audit script

**Files:**

- Create: `experiments/cf_cycle_1/harness_audit.py`
- Create: `experiments/cf_cycle_1/README.md`
- Test: `tests/test_data.py`

- [ ] **Step 1: Create the experiment directory**

Run:

```cmd
if not exist experiments\cf_cycle_1 mkdir experiments\cf_cycle_1
```

Expected: directory exists and no output is required.

- [ ] **Step 2: Write `experiments/cf_cycle_1/harness_audit.py`**

Use this complete script:

```python
from __future__ import annotations

import inspect
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

import torch

import cascades.data as data
import experiment_matrix
import research_runner
import train


CURRENT4 = list(data.TASK_FILES)
REASONING3 = list(experiment_matrix.DEFAULT_TASK_FILES)


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def count_examples(path: str) -> int | None:
    p = Path(path)
    if not p.exists():
        return None
    return sum(1 for line in p.open("r", encoding="utf-8") if line.strip())


def task_manifest(files: list[str]) -> list[dict]:
    rows = []
    for i, file_name in enumerate(files):
        p = Path(file_name)
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


def runner_forwarding_snapshot() -> dict:
    src = inspect.getsource(research_runner.ExperimentRunner._run_cascades)
    return {
        "forwards_rank": '"rank"' in src or "'rank'" in src,
        "forwards_max_length": "max_length" in src,
        "pops_task_files": 'task_files = overrides.pop("task_files", None)' in src,
        "calls_train_cascades_kwargs": "accuracy_matrix = train_cascades(**kwargs)" in src,
    }


def num_samples_snapshot() -> dict:
    text = Path("train.py").read_text(encoding="utf-8-sig")
    positions = [m.start() for m in re.finditer("num_samples", text)]
    return {
        "occurrence_count": len(positions),
        "positions": positions,
        "wired_to_prepare_data": "num_samples" in "\n".join(
            line for line in text.splitlines() if "prepare_data(" in line
        ),
    }


def main() -> None:
    out_dir = Path("experiments/cf_cycle_1")
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot = {
        "git_revision": git_revision(),
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "current_num_tasks": data.NUM_TASKS,
        "current_task_files": CURRENT4,
        "reasoning3_task_files": REASONING3,
        "current4_manifest": task_manifest(CURRENT4),
        "reasoning3_manifest": task_manifest(REASONING3),
        "train_cascades_signature": str(inspect.signature(train.train_cascades)),
        "runner_forwarding": runner_forwarding_snapshot(),
        "num_samples": num_samples_snapshot(),
        "gpu_preconditions": {
            "ok": False,
            "reason": "Set true only after task-suite drift is acknowledged and CF-cycle-1 wrapper artifacts are present.",
        },
    }

    (out_dir / "audit_snapshot.json").write_text(
        json.dumps(snapshot, indent=2), encoding="utf-8"
    )
    (out_dir / "task_manifest_current4.json").write_text(
        json.dumps(snapshot["current4_manifest"], indent=2), encoding="utf-8"
    )
    (out_dir / "task_manifest_reasoning3.json").write_text(
        json.dumps(snapshot["reasoning3_manifest"], indent=2), encoding="utf-8"
    )
    print(json.dumps(snapshot, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the audit**

Run:

```cmd
python experiments\cf_cycle_1\harness_audit.py > experiments\cf_cycle_1\harness_audit.log 2>&1
```

Expected: `experiments/cf_cycle_1/audit_snapshot.json`, `experiments/cf_cycle_1/task_manifest_current4.json`, and `experiments/cf_cycle_1/task_manifest_reasoning3.json` exist.

- [ ] **Step 4: Run the existing data tests and preserve output**

Run:

```cmd
python -m pytest tests/test_data.py -q > experiments\cf_cycle_1\pytest_test_data.log 2>&1
```

Expected before data-test repair: `pytest_test_data.log` records the known 3-vs-4 task mismatch. Expected after data-test repair: all tests pass.

- [ ] **Step 5: Commit audit scaffold**

Run:

```cmd
git add experiments/cf_cycle_1/harness_audit.py experiments/cf_cycle_1/README.md docs/superpowers/plans/2026-05-16-cf-cycle-1-nullspace-ablation.md
git commit -m "exp: add cf-cycle-1 harness audit plan"
```

Expected: commit succeeds after review.

### Task 2: Add controlled ablation runner with instrumentation

**Files:**

- Create: `experiments/cf_cycle_1/run_nullspace_ablation.py`
- Create: `experiments/cf_cycle_1/compare_nullspace_ablation.py`
- Modify: no core training files unless the wrapper cannot capture the required signal

- [ ] **Step 1: Write `experiments/cf_cycle_1/run_nullspace_ablation.py`**

Use this complete script:

```python
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

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
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def count_examples(path: str) -> int | None:
    p = Path(path)
    if not p.exists():
        return None
    return sum(1 for line in p.open("r", encoding="utf-8") if line.strip())


def task_manifest(files: list[str]) -> list[dict]:
    rows = []
    for i, file_name in enumerate(files):
        p = Path(file_name)
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


def config_for_arm(arm: str) -> AblationConfig:
    return AblationConfig(
        enable_paca=True,
        enable_deal=True,
        enable_gainlora_gate=True,
        enable_coso_nullspace=(arm == "treatment"),
        enable_cllora_reassign=False,
        enable_svc=True,
        enable_dmole_select=True,
        enable_funlora=True,
        gqa_ratio=1.0,
        ear_gamma=1e-4,
        enable_soft_ear=True,
        enable_principal_expansion=True,
        cfg_lambda=1.5,
        enable_ambient_dedup=True,
    )


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


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
        },
        "vram": [],
    }

    orig_reassign = adapters_mod._cllora_reassign
    orig_freeze = adapters_mod.CASCADESAdapter.freeze_current_subspace
    orig_batched_null = train_module.batched_null_space_extraction
    orig_log_vram = train_module.log_vram

    def wrapped_reassign(grad, null_sketch, config, frozen_basis=None):
        frozen_cols = int(frozen_basis.shape[1]) if frozen_basis is not None else 0
        stats["reassign"]["calls_total"] += 1
        if null_sketch is not None:
            stats["reassign"]["calls_with_null_sketch"] += 1
        before = grad.detach().clone()
        out = orig_reassign(grad, null_sketch, config, frozen_basis=frozen_basis)
        if frozen_cols > 0:
            stats["reassign"]["calls_with_frozen_basis"] += 1
            stats["reassign"]["max_frozen_cols"] = max(
                stats["reassign"]["max_frozen_cols"], frozen_cols
            )
            stats["reassign"]["removed_norm_sum"] += float((before - out).norm().item())
        return out

    def wrapped_freeze(self):
        u_before = int(getattr(self, "frozen_null_basis").shape[1])
        v_before = int(getattr(self, "frozen_null_basis_V").shape[1])
        rank = int(self.U_shared.shape[1])
        orig_freeze(self)
        u_after = int(getattr(self, "frozen_null_basis").shape[1])
        v_after = int(getattr(self, "frozen_null_basis_V").shape[1])
        stats["freeze_events"].append(
            {
                "task_boundary_call": len(stats["freeze_events"]) + 1,
                "u_cols_before": u_before,
                "u_cols_after": u_after,
                "v_cols_before": v_before,
                "v_cols_after": v_after,
                "adapter_rank": rank,
                "out_features": int(self.U_shared.shape[0]),
                "in_features": int(self.V_shared.shape[1]),
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
        if torch.cuda.is_available():
            stats["vram"].append(
                {
                    "tag": tag,
                    "allocated_mb": torch.cuda.memory_allocated() / (1024**2),
                    "reserved_mb": torch.cuda.memory_reserved() / (1024**2),
                    "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
                }
            )
        return orig_log_vram(tag, device)

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
    cfg = config_for_arm(arm)
    out_dir = Path(args.output_root) / arm
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
        )
    finally:
        restore()

    wall_time = time.time() - start
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
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

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arms = ["control", "treatment"] if args.arm == "both" else [args.arm]
    for arm in arms:
        run_arm(args, arm)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `experiments/cf_cycle_1/compare_nullspace_ablation.py`**

Use this complete script:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CF-cycle-1 null-space ablation arms")
    parser.add_argument("--root", default="experiments/cf_cycle_1/nullspace_ablation")
    args = parser.parse_args()
    root = Path(args.root)
    control = load_json(root / "control" / "metrics.json")
    treatment = load_json(root / "treatment" / "metrics.json")
    treatment_instr = load_json(root / "treatment" / "instrumentation.json")

    old_delta_gaps = [
        t - c
        for t, c in zip(treatment["old_task_deltas"], control["old_task_deltas"])
    ]
    comparison = {
        "delta_bwt_points": (treatment["bwt"] - control["bwt"]) * 100.0,
        "delta_avg_acc_points": (treatment["avg_acc"] - control["avg_acc"]) * 100.0,
        "old_task_delta_gaps_points": [x * 100.0 for x in old_delta_gaps],
        "control_peak_vram_mb": control["peak_vram_mb"],
        "treatment_peak_vram_mb": treatment["peak_vram_mb"],
        "projection_active": (
            treatment_instr["reassign"]["calls_with_frozen_basis"] > 0
            and treatment_instr["reassign"]["removed_norm_sum"] > 0.0
        ),
        "frozen_basis_nonempty": any(
            event["u_cols_after"] > 0 or event["v_cols_after"] > 0
            for event in treatment_instr["freeze_events"]
        ),
    }
    comparison["continue"] = (
        comparison["delta_bwt_points"] >= 1.5
        and comparison["delta_avg_acc_points"] >= -2.0
        and min(comparison["old_task_delta_gaps_points"], default=0.0) >= -3.0
        and comparison["treatment_peak_vram_mb"] < 7500.0
        and comparison["projection_active"]
        and comparison["frozen_basis_nonempty"]
    )
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Syntax-check the scripts without GPU**

Run:

```cmd
python -m py_compile experiments\cf_cycle_1\harness_audit.py experiments\cf_cycle_1\run_nullspace_ablation.py experiments\cf_cycle_1\compare_nullspace_ablation.py
```

Expected: command exits `0` with no output.

- [ ] **Step 4: Run harness audit before GPU**

Run:

```cmd
python experiments\cf_cycle_1\harness_audit.py > experiments\cf_cycle_1\harness_audit.log 2>&1
```

Expected: `experiments/cf_cycle_1/audit_snapshot.json` exists and records `current_task_files`, `reasoning3_task_files`, runner pass-through status, and current GPU availability.

- [ ] **Step 5: Run the GPU control arm**

Run:

```cmd
python experiments\cf_cycle_1\run_nullspace_ablation.py --arm control --task-suite current4 --seed 42 --epochs 2 --rank 8 --max-length 384 --output-root experiments\cf_cycle_1\nullspace_ablation > experiments\cf_cycle_1\nullspace_ablation\control.log 2>&1
```

Expected: `experiments/cf_cycle_1/nullspace_ablation/control/metrics.json` exists. The control log should show [`enable_coso_nullspace`](cascades/config.py:30) as `False` and no frozen-basis projection events.

- [ ] **Step 6: Run the GPU treatment arm**

Run:

```cmd
python experiments\cf_cycle_1\run_nullspace_ablation.py --arm treatment --task-suite current4 --seed 42 --epochs 2 --rank 8 --max-length 384 --output-root experiments\cf_cycle_1\nullspace_ablation > experiments\cf_cycle_1\nullspace_ablation\treatment.log 2>&1
```

Expected: `experiments/cf_cycle_1/nullspace_ablation/treatment/metrics.json` and `experiments/cf_cycle_1/nullspace_ablation/treatment/instrumentation.json` exist. The treatment instrumentation should show non-empty frozen bases after boundaries and projection calls with non-empty frozen bases.

- [ ] **Step 7: Compare arms**

Run:

```cmd
python experiments\cf_cycle_1\compare_nullspace_ablation.py --root experiments\cf_cycle_1\nullspace_ablation > experiments\cf_cycle_1\nullspace_ablation\comparison.json
```

Expected: `comparison.json` contains `delta_bwt_points`, `delta_avg_acc_points`, `old_task_delta_gaps_points`, `projection_active`, `frozen_basis_nonempty`, and `continue`.

- [ ] **Step 8: Commit executable experiment artifacts**

Run:

```cmd
git add experiments/cf_cycle_1/harness_audit.py experiments/cf_cycle_1/run_nullspace_ablation.py experiments/cf_cycle_1/compare_nullspace_ablation.py experiments/cf_cycle_1/README.md
git commit -m "exp: add cf-cycle-1 nullspace ablation runner"
```

Expected: commit succeeds after review.

### Task 3: Repair task-suite tests or explicitly document current4 as official

**Files:**

- Modify: `tests/test_data.py`
- Modify: `README.md`
- Optional Modify: `experiment_matrix.py`

- [ ] **Step 1: Update `tests/test_data.py` expectations for current code state**

Change [`TestTaskDefinitions.test_correct_number_of_tasks`](tests/test_data.py:20) to assert the current 4-task suite if current4 remains official:

```python
def test_correct_number_of_tasks(self):
    assert NUM_TASKS == 4, "CASCADES current4 evaluates 4 sequential continual learning tasks"
```

Change [`TestTaskDefinitions.test_task_files_have_expected_names`](tests/test_data.py:37) to validate distinct task files without assuming task index matches filename:

```python
def test_task_files_have_expected_names(self):
    """Task files should be JSONL files with stable task identifiers in their names."""
    basenames = [os.path.basename(path) for path in TASK_FILES]
    assert len(set(basenames)) == NUM_TASKS, f"Task files are not unique: {basenames}"
    for basename in basenames:
        assert basename.startswith("task"), f"Task file should start with 'task': {basename}"
        assert basename.endswith(".jsonl"), f"Task file should be JSONL: {basename}"
```

- [ ] **Step 2: Run data tests**

Run:

```cmd
python -m pytest tests/test_data.py -q
```

Expected: all tests in [`tests/test_data.py`](tests/test_data.py:1) pass.

- [ ] **Step 3: Update README task-suite section**

Modify [`README.md`](README.md:129) to state that historical saved results used the original 3-task reasoning suite, while current CF-cycle-1 controlled ablations use current4 with Digital Twin as task 3.

Required text:

```markdown
Current CF-cycle-1 task suite (`current4`, used by `cascades/data.py`) is:
1. `data/task2_csqa_cot.jsonl` — CommonsenseQA
2. `data/task1_arc_cot.jsonl` — ARC Science
3. `data/task0_gsm8k_cot.jsonl` — GSM8K Math
4. `data/task3_digital_twin.jsonl` — Digital Twin

Historical v9/v10 BWT values in saved CSVs and earlier README text may refer to the original 3-task reasoning suite. Do not compare BWT across suites without recording the task manifest.
```

- [ ] **Step 4: Commit docs/test repair**

Run:

```cmd
git add tests/test_data.py README.md
git commit -m "test: document current4 task suite for cf-cycle-1"
```

Expected: commit succeeds after review.

---

## Optional secondary generative validation

Do not enable [`--eval_em`](train.py:527) in the first control/treatment run unless the GPU budget remains safe after proxy metrics. Current [`train_cascades()`](train.py:171) runs up to 25 generative samples per task when [`eval_em`](train.py:184) is true and does not return the generated EM payload to the caller, so it is not a cheap fixed subset yet.

If proxy criteria pass and a secondary check is needed, add a later code-mode task to expose `generative_eval_samples=5` and fixed sample indices in [`train_cascades()`](train.py:171), then rerun only the winning arm and control with the exact same saved weights or a deterministic replay.

---

## Handoff packet to next mode

**Experiments run:**

- CPU audit: [`tests/test_data.py`](tests/test_data.py:20) via `python -m pytest tests/test_data.py -q`.
- CPU audit: signature/runner pass-through introspection via `python -c ...`.
- CPU audit: task manifest/revision via `python -c ...`.
- CPU audit: source search for [`num_samples`](train.py:183), [`prepare_data()`](cascades/data.py:80), [`enable_coso_nullspace`](cascades/config.py:30), and [`enable_cllora_reassign`](cascades/config.py:32).

**Observed facts:**

- [`tests/test_data.py`](tests/test_data.py:20) fails because it expects 3 tasks but [`NUM_TASKS`](cascades/data.py:49) is 4.
- [`tests/test_data.py`](tests/test_data.py:37) assumes task index appears in filename, but current [`TASK_FILES`](cascades/data.py:35) intentionally starts with `task2_csqa_cot.jsonl`.
- [`research_runner.ExperimentRunner._run_cascades()`](research_runner.py:366) drops task files and does not forward `rank` or `max_length`.
- [`num_samples`](train.py:183) is not wired to [`prepare_data()`](cascades/data.py:80).
- [`_cllora_reassign()`](cascades/adapters.py:176) applies frozen-basis projection before checking [`enable_cllora_reassign`](cascades/config.py:32), so strict frozen protection can be isolated by setting [`enable_coso_nullspace`](cascades/config.py:30) true and [`enable_cllora_reassign`](cascades/config.py:32) false.

**Interpretations:**

- Hypothesis 2 is currently supported: harness/config drift is real enough to block unqualified GPU evidence.
- Hypothesis 1 remains untested on GPU because projection activation evidence is not yet persisted.
- Code-mode changes are required for reliable experiment execution and evidence capture.

**Checks not run and why:**

- GPU ablation was not run because the CPU audit found task/config mismatch and missing projection instrumentation.
- Generative exact match was not run because current [`eval_em`](train.py:184) is not a tiny fixed subset and does not return structured EM metrics to the wrapper.

**Recommended next mode:** `code` to implement the audit/runner scripts and task-suite documentation repair, then return to `llm-experiment-designer` or `llm-result-critic` with artifacts.
