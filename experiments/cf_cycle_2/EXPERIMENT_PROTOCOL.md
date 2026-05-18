# CF-cycle-2 Reasoning3 Null-Space Feasibility Protocol

## Verdict

The reduced-memory `reasoning3` GPU launch is ready for explicit experiment-design approval after CPU-light verification. Launch control first, inspect its status and artifacts, and only then launch treatment. Do not use the invalid CF-cycle-1 `current4` artifacts as algorithmic evidence.

## CPU-light evidence collected before approval

From the repository root, the experiment-design review reran:

```cmd
python -m py_compile train.py experiments\cf_cycle_1\run_nullspace_ablation.py experiments\cf_cycle_1\compare_nullspace_ablation.py && python -m pytest tests/test_data.py tests/test_cf_cycle_2_guardrails.py -q
```

Observed result: exit code 0, 16 tests passed in 0.90 seconds. Only a pytest-asyncio deprecation warning was printed; it is unrelated to the guardrail behavior.

The review also ran:

```cmd
python experiments\cf_cycle_1\run_nullspace_ablation.py --help && python experiments\cf_cycle_1\compare_nullspace_ablation.py --help
```

Observed result: exit code 0. The runner help showed `--task-suite {current4,reasoning3}`, `--vram-threshold-mb`, `--allow-nonfinite`, and `--allow-vram-over-threshold`; the comparison help showed `--root` and `--vram-threshold-mb`.

No GPU command was run during this review.

## Approved launch sequence

Create the output directory once:

```cmd
if not exist experiments\cf_cycle_2\nullspace_ablation mkdir experiments\cf_cycle_2\nullspace_ablation
```

### Step 1: control arm only

```cmd
python experiments\cf_cycle_1\run_nullspace_ablation.py --arm control --task-suite reasoning3 --seed 42 --epochs 2 --rank 4 --max-length 256 --output-root experiments\cf_cycle_2\nullspace_ablation --vram-threshold-mb 7500 > experiments\cf_cycle_2\nullspace_ablation\control.log 2>&1
```

Continue to treatment only if the control command exits 0 and the control artifacts meet all control-gate criteria below.

### Step 2: treatment arm, conditional on valid control

```cmd
python experiments\cf_cycle_1\run_nullspace_ablation.py --arm treatment --task-suite reasoning3 --seed 42 --epochs 2 --rank 4 --max-length 256 --output-root experiments\cf_cycle_2\nullspace_ablation --vram-threshold-mb 7500 > experiments\cf_cycle_2\nullspace_ablation\treatment.log 2>&1
```

Continue to comparison only if the treatment command exits 0 and the treatment artifacts meet all treatment-gate criteria below.

### Step 3: comparison, conditional on valid control and treatment

```cmd
python experiments\cf_cycle_1\compare_nullspace_ablation.py --root experiments\cf_cycle_2\nullspace_ablation --vram-threshold-mb 7500 > experiments\cf_cycle_2\nullspace_ablation\comparison.json
```

The comparison command must exit 0 and write valid JSON with `valid: true` before algorithmic metrics are interpreted.

## Expected artifacts

Top-level output directory:

- `experiments/cf_cycle_2/nullspace_ablation/control.log`
- `experiments/cf_cycle_2/nullspace_ablation/treatment.log`
- `experiments/cf_cycle_2/nullspace_ablation/comparison.json`

Each arm directory should contain:

- `config.json`
- `task_manifest.json`
- `accuracy_matrix.npy`
- `cascades_results.csv`
- `cascades_weights.pt`
- `metrics.json`
- `instrumentation.json`
- `run_status.json`

The treatment `instrumentation.json` must include active projection evidence: non-empty frozen basis events, `_cllora_reassign()` calls with a frozen basis, and positive `removed_norm_sum`.

## Control-gate criteria

The control arm is valid only if all of these are true:

1. The command exits 0.
2. `run_status.json` has `status` equal to `completed`.
3. `metrics.json` and `instrumentation.json` contain only finite numeric values.
4. `metrics.json` has `task_suite` equal to `reasoning3`, `seed` equal to 42, `epochs` equal to 2, `rank` equal to 4, and `max_length` equal to 256.
5. `task_manifest.json` lists exactly the three `reasoning3` tasks: GSM8K, ARC, and CSQA, with no Digital Twin task.
6. `peak_vram_mb` is finite and at or below 7500 MB.

If any control-gate criterion fails, stop. Do not launch treatment. Preserve `control.log`, the control arm directory, and any nonzero exit status as failed-run evidence, then hand to result critic and likely debug/code for guardrail or memory investigation.

## Treatment-gate criteria

The treatment arm is valid only if all control-gate-style criteria are true for treatment and these additional projection checks pass:

1. At least one `freeze_events` entry has `u_cols_after > 0` or `v_cols_after > 0`.
2. `reassign.calls_with_frozen_basis > 0`.
3. `reassign.removed_norm_sum > 0.0`.

If treatment exits nonzero due to a guardrail, stop and preserve the treatment artifacts as failed-run evidence. If treatment completes but projection evidence is inactive, treat this as a harness/projection failure rather than null-space falsification; stop and hand to code/debug with the preserved artifacts.

## Comparison and interpretation criteria

Feasibility success requires comparison output with:

1. `valid: true`.
2. Both arms completed.
3. Both arms finite.
4. Both arms at or below 7500 MB peak VRAM.
5. Matching `reasoning3` task manifests.
6. Active treatment projection evidence.

Algorithmic treatment success can be considered only after feasibility success and requires:

1. Treatment BWT at least control BWT plus 0.015, reported by comparison as `delta_bwt_points >= 1.5`.
2. Treatment final average ACC no more than 0.02 worse than control, reported as `delta_avg_acc_points >= -2.0`.
3. No old-task delta gap worse than -0.03, reported as every `old_task_delta_gaps_points` value at least -3.0.

If `comparison.json` has `valid: true` but `continue: false`, the reduced-memory feasibility run is valid evidence but does not support algorithmic escalation. Do not claim catastrophic forgetting is solved.

## Stop/continue decision

Recommended launch policy: control-only first, then treatment sequentially only if control completes and passes the gate. This is safer than launching both arms via `--arm both` because it prevents treatment GPU time from being spent after an invalid control and keeps failed-control evidence clean.

## Handoff status map

- Guardrail readiness for reduced-memory `reasoning3`: supported by CPU-light syntax, tests, and CLI help checks.
- Frozen null-space catastrophic-forgetting mitigation effect: inconclusive until approved GPU control and treatment runs complete and compare successfully.
- CF-cycle-1 `current4` artifact reuse: falsified/blocked as algorithmic evidence by the prior report; remains diagnostic only.
- Inactive treatment projection evidence: would indicate harness/projection failure, not algorithmic null-space falsification.
