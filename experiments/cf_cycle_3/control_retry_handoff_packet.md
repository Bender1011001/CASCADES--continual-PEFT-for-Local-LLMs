# CF-cycle-3 Experiment Designer Handoff — reduced-memory reasoning3 control retry

## 1. Hypothesis tested

The fixed data loader removes the known all-ignored-label failure mode from the reduced-memory `reasoning3` control arm, so the control can complete with finite artifacts while staying under the 7500 MB VRAM threshold.

## 2. Experiment performed

1. Reviewed CF-cycle-3 root-cause evidence and existing fixed-loader outputs.
2. Re-ran CPU-light preflight checks for the data-loader fix and full `reasoning3` task-suite label validity.
3. Launched **control only** into a new output root to preserve the failed CF-cycle-2 artifacts.
4. Monitored the GPU run through completion.
5. Ran a post-run control-gate validator over required artifacts, run status, metrics, instrumentation, manifest, VRAM, and matrix finiteness.

Treatment was **not launched** in this step.

## 3. Files changed

- Added `experiments/cf_cycle_3/reasoning3_prepare_data_preflight.py` to scan every reduced-memory `reasoning3` batch after the data-loader fix.
- Added `experiments/cf_cycle_3/validate_control_gate.py` to validate the retried control artifacts against the control gate.
- Added this handoff packet at `experiments/cf_cycle_3/control_retry_handoff_packet.md`.

## 4. Commands/checks run

### CPU-light regression and old static probe sanity check

```cmd
python -m pytest tests/test_data.py -q && python -m py_compile cascades\data.py experiments\cf_cycle_3\batch93_probe.py && python experiments\cf_cycle_3\batch93_probe.py --start-batch 93 --end-batch 93 --out experiments/cf_cycle_3/batch93_static_probe.preflight.json
```

Observed: exit 0; `tests/test_data.py` reported 14 passed. Important caveat: `batch93_probe.py` intentionally encodes the **old prompt-first truncation** diagnostic logic, so its preflight output still reports old-style `batch_93_all_labels_masked=true`; it was useful only as a sanity check that the historical root-cause sample mapping still points to JSONL line 10, not as validation of the fixed `prepare_data()` path.

### Full fixed-loader `reasoning3` preflight

```cmd
python -m py_compile experiments\cf_cycle_3\reasoning3_prepare_data_preflight.py && python experiments\cf_cycle_3\reasoning3_prepare_data_preflight.py --fail-on-invalid
```

Observed: exit 0; wrote `experiments/cf_cycle_3/reasoning3_prepare_data_preflight.json`.

### Control-only retry command

```cmd
if not exist experiments\cf_cycle_3\nullspace_ablation_retry mkdir experiments\cf_cycle_3\nullspace_ablation_retry && python experiments\cf_cycle_1\run_nullspace_ablation.py --arm control --task-suite reasoning3 --seed 42 --epochs 2 --rank 4 --max-length 256 --output-root experiments\cf_cycle_3\nullspace_ablation_retry --vram-threshold-mb 7500 > experiments\cf_cycle_3\nullspace_ablation_retry\control.log 2>&1
```

Observed: control completed; no treatment was run.

### Post-run gate validation

```cmd
python -m py_compile experiments\cf_cycle_3\reasoning3_prepare_data_preflight.py experiments\cf_cycle_3\validate_control_gate.py && python experiments\cf_cycle_3\validate_control_gate.py > experiments\cf_cycle_3\control_gate_validation.json && type experiments\cf_cycle_3\control_gate_validation.json
```

Observed: exit 0; wrote `experiments/cf_cycle_3/control_gate_validation.json` with `valid=true`.

## 5. Raw or summarized results

### Preflight evidence

- `experiments/cf_cycle_3/batch93_static_probe.verify.json` from the code handoff: batch 93 valid labels = 1, `batch_93_all_labels_masked=false`, zero-label batches empty, min valid labels = 1.
- `experiments/cf_cycle_3/reasoning3_prepare_data_preflight.json`: all three `reasoning3` tasks valid under seed 42 / max length 256; each had zero zero-label batches; max sequence length was 256 for every task.

### Control retry artifacts

- New output root: `experiments/cf_cycle_3/nullspace_ablation_retry`.
- Preserved failed CF-cycle-2 output root: `experiments/cf_cycle_2/nullspace_ablation/control` was not overwritten.
- Control status: `experiments/cf_cycle_3/nullspace_ablation_retry/control/run_status.json` reports `status=completed`, `task_suite=reasoning3`, `seed=42`, `rank=4`, `max_length=256`, `epochs=2`, `allow_nonfinite=false`, `allow_vram_over_threshold=false`, `peak_vram_mb=6848.810546875`, `wall_time_s=1295.886130809784`.
- Control metrics: `experiments/cf_cycle_3/nullspace_ablation_retry/control/metrics.json` reports `avg_acc=0.5652729740720018`, `bwt=-0.03614532252823527`, `final_accs=[0.36523208501204096, 0.5494726965413683, 0.7811141406625961]`, and `old_task_deltas=[-0.0645747844901221, -0.007715860566348431]`.
- Gate validation: `experiments/cf_cycle_3/control_gate_validation.json` reports `valid=true`, no missing artifacts, no parameter mismatches, expected manifest paths, finite metrics, finite instrumentation, finite accuracy matrix shape `[3, 3]`, and `under_vram_threshold=true`.

## 6. Result label

**Confirmed for the bounded control-gate question:** the fixed data loader is sufficient for the reduced-memory `reasoning3` control arm to complete finite and under the configured VRAM threshold in this one seed / one suite / one run.

**Not tested:** treatment projection behavior and null-space mitigation effect. Catastrophic forgetting is not solved by this evidence.

## 7. Confidence level

**Medium-high** that the previously diagnosed all-ignored-label control failure is fixed for this reduced-memory `reasoning3` control protocol.

**Low/unknown** for treatment behavior and algorithmic forgetting claims because treatment has not been launched after this valid control.

## 8. Recommended next action

Route to `llm-result-critic` to review the control retry evidence. If the critic accepts the control gate, the next executable step is a **treatment-only** retry in the same new output root, followed by comparison only if the treatment completes and passes treatment-gate criteria. Do not run treatment and comparison blindly together.

Suggested treatment command after critic approval only:

```cmd
python experiments\cf_cycle_1\run_nullspace_ablation.py --arm treatment --task-suite reasoning3 --seed 42 --epochs 2 --rank 4 --max-length 256 --output-root experiments\cf_cycle_3\nullspace_ablation_retry --vram-threshold-mb 7500 > experiments\cf_cycle_3\nullspace_ablation_retry\treatment.log 2>&1
```

Suggested comparison command after valid treatment only:

```cmd
python experiments\cf_cycle_1\compare_nullspace_ablation.py --root experiments\cf_cycle_3\nullspace_ablation_retry --vram-threshold-mb 7500 > experiments\cf_cycle_3\nullspace_ablation_retry\comparison.json
```

If the critic rejects any control-gate evidence, route to `debug` with `experiments/cf_cycle_3/control_gate_validation.json`, `experiments/cf_cycle_3/nullspace_ablation_retry/control.log`, and the control artifact directory.

## 9. Checks not run

- Treatment arm was intentionally not run in this step.
- Comparison was intentionally not run because no treatment artifact exists yet.
- No multi-seed or longer-suite generalization check was run.
- No claim is made about solving catastrophic forgetting.

## Handoff target

`llm-result-critic`
