# CF-cycle-1 Frozen Null-Space Ablation

This directory contains the executable artifacts for the first CF-cycle-1 null-space retention experiment. The goal is to make the planned one-seed, two-arm ablation auditable before spending GPU time.

## Why this path exists

CPU-light probes found harness drift that can make old GPU results non-comparable:

- The current official task suite is `current4` from `cascades/data.py`: CommonsenseQA -> ARC Science -> GSM8K Math -> Digital Twin.
- Historical v9/v10 saved CSVs and older README text may refer to the original `reasoning3` suite: GSM8K -> ARC -> CommonsenseQA.
- `research_runner.ExperimentRunner._run_cascades()` currently drops `task_files` and does not forward `rank` or `max_length`, so CF-cycle-1 uses direct wrapper scripts instead of `research_runner.py`.
- `train_cascades(num_samples=...)` is not currently wired through to `prepare_data()`, so `num_samples` is not part of this experiment contract.

## Scripts

- `harness_audit.py` writes CPU-light diagnostics and task manifests.
- `run_nullspace_ablation.py` runs either `control`, `treatment`, or `both` arms through a direct wrapper around `train_cascades()`.
- `compare_nullspace_ablation.py` compares persisted arm metrics and projection/basis instrumentation.

## CPU-light verification only

Run these commands before any GPU work:

```cmd
python -m py_compile experiments\cf_cycle_1\harness_audit.py experiments\cf_cycle_1\run_nullspace_ablation.py experiments\cf_cycle_1\compare_nullspace_ablation.py
python experiments\cf_cycle_1\harness_audit.py > experiments\cf_cycle_1\harness_audit.log 2>&1
python -m pytest tests/test_data.py -q
```

The audit command persists:

- `audit_snapshot.json`
- `task_manifest_current4.json`
- `task_manifest_reasoning3.json`
- `harness_audit.log`

## GPU commands for later explicit launch

Do not run these unless GPU execution is explicitly requested after reviewing the audit artifacts:

```cmd
if not exist experiments\cf_cycle_1\nullspace_ablation mkdir experiments\cf_cycle_1\nullspace_ablation
python experiments\cf_cycle_1\run_nullspace_ablation.py --arm control --task-suite current4 --seed 42 --epochs 2 --rank 8 --max-length 384 --output-root experiments\cf_cycle_1\nullspace_ablation > experiments\cf_cycle_1\nullspace_ablation\control.log 2>&1
python experiments\cf_cycle_1\run_nullspace_ablation.py --arm treatment --task-suite current4 --seed 42 --epochs 2 --rank 8 --max-length 384 --output-root experiments\cf_cycle_1\nullspace_ablation > experiments\cf_cycle_1\nullspace_ablation\treatment.log 2>&1
python experiments\cf_cycle_1\compare_nullspace_ablation.py --root experiments\cf_cycle_1\nullspace_ablation > experiments\cf_cycle_1\nullspace_ablation\comparison.json
```

## Arms

Both arms keep `enable_cllora_reassign=False` to isolate strict frozen-basis projection from active-sketch reassignment.

- `control`: `enable_coso_nullspace=False`, `enable_cllora_reassign=False`
- `treatment`: `enable_coso_nullspace=True`, `enable_cllora_reassign=False`

Shared defaults: seed `42`, rank `8`, max length `384`, epochs `2`, sleep enabled, proxy metrics primary, generative exact match disabled.

## Required per-arm GPU artifacts

When launched later, each arm should persist:

- `config.json`
- `task_manifest.json`
- `cascades_results.csv`
- `accuracy_matrix.npy`
- `metrics.json`
- `instrumentation.json`
- `cascades_weights.pt`

## Interpretation guardrails

The treatment only tests the algorithmic hypothesis if instrumentation proves projection was active: frozen bases must become non-empty after a task boundary, `_cllora_reassign()` must be called with a frozen basis, and `removed_norm_sum` must be positive. If projection gates are inactive, the run is a harness failure rather than a null-space falsification.
