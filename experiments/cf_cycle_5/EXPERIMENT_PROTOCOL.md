# CF-cycle-5 Experiment Protocol — CL-LoRA-active reduced reasoning3 treatment

Date: 2026-05-17

handoff_target: code, then llm-experiment-designer for bounded execution, then llm-result-critic after evidence exists

## 1. Current objective

Run one bounded reduced-memory `reasoning3` treatment-redesign check for catastrophic-forgetting mitigation: test whether active CL-LoRA reassignment plus soft EAR, layered on strict frozen null-space projection, improves BWT enough to justify further testing on the RTX 4060 Ti 8GB path.

Non-goals: no full `current4`, v10 default escalation, Digital Twin, generative subset, broad settings fuzzer, threshold tuning after results, or catastrophic-forgetting solution claim.

## 2. Hypothesis tested

Prior frozen-only treatment underperformed because `config_for_arm()` in `experiments/cf_cycle_1/run_nullspace_ablation.py` enabled strict frozen projection for treatment but disabled active CL-LoRA reassignment for both arms. The candidate treatment should keep strict projection active and turn on active-sketch reassignment plus soft EAR:

- treatment: `enable_coso_nullspace=True`, `enable_cllora_reassign=True`, `enable_soft_ear=True`, `ear_gamma=1e-4`.
- control: unchanged seed-43 reduced `reasoning3` control can be reused provisionally from CF-cycle-4; paired rerun is cleaner only if a reviewer rejects cross-revision control reuse.

## 3. Current evidence state

### Confirmed

- CF-cycle-4 seed 43 produced valid reduced `reasoning3` control/treatment artifacts under 7500 MB, active frozen projection, and `continue=false`; `delta_bwt_points=-0.008929480489536235`.
- CF-cycle-3 seed 42 produced valid reduced `reasoning3` control/treatment artifacts under 7500 MB, active frozen projection, and `continue=false`; `delta_bwt_points=0.36167033073388477`.
- Current runner treatment is frozen-only because treatment enables `enable_coso_nullspace` while both arms serialize `enable_cllora_reassign=False`.
- `AblationConfig` defaults `enable_cllora_reassign=True`, so the ablation wrapper deviates from production defaults.
- `_cllora_reassign()` applies strict frozen projection before the active CL-LoRA gate; disabling `enable_cllora_reassign` bypasses the active-sketch and soft-EAR path but does not bypass frozen projection.

### Likely

- Isolated frozen projection is feasible but too weak, mistimed, under-capacity, or confounded. Confidence: medium.
- A one-knob active-reassignment treatment is higher expected value than full-suite escalation. Confidence: medium-high.

### Plausible

- Active CL-LoRA reassignment may add missing treatment strength while preserving the same reduced memory envelope. Confidence: medium.

### Weak signal

- Seed-43 treatment projection activity was real but modest: 3600 frozen-basis calls and removed norm sum about 3.10; normalized removed norm per frozen call was not recorded.

### Contradicted

- Frozen-only null-space treatment success is not supported across seeds 42 and 43 because both miss the +1.5 BWT-point threshold.

### Blocked

- Full `current4`, v10 default, Digital Twin, generative subset, and broad settings-fuzzer escalation remain blocked until a reduced-suite treatment gate succeeds.

## 4. Inspected surfaces and implementation constraints

### Runner config surface

In `experiments/cf_cycle_1/run_nullspace_ablation.py`, change only the experiment-selection surface needed to make the treatment artifact-distinguishable:

1. Add a backward-compatible CLI flag such as `--treatment-variant` with choices `frozen-only` and `cllora-active`, defaulting to `frozen-only`.
2. Update `config_for_arm(arm, treatment_variant="frozen-only")` so control remains unchanged and treatment selects exactly one variant.
3. For `cllora-active` treatment, serialize:
   - `enable_coso_nullspace=True`
   - `enable_cllora_reassign=True`
   - `enable_soft_ear=True`
   - `ear_gamma=1e-4`
4. Preserve all other treatment flags unless required by existing code.
5. Write the variant string into `config.json` and `run_status.json` so CF-cycle-5 artifacts cannot be confused with CF-cycle-3/4 frozen-only treatment artifacts.

Do not silently change control behavior. If the old seed-43 control is reused, copy the CF-cycle-4 control artifacts into the CF-cycle-5 output root and record provenance.

### Instrumentation

In `install_instrumentation()` in `experiments/cf_cycle_1/run_nullspace_ablation.py`, preserve existing keys and add only bounded evidence:

- Keep `calls_total`, `calls_with_null_sketch`, `calls_with_frozen_basis`, `max_frozen_cols`, and `removed_norm_sum`.
- Compute `removed_norm_sum` as strict frozen-basis removal only, not combined frozen plus active adjustment. This prevents active reassignment from falsely satisfying the frozen-projection gate.
- Add `removed_norm_per_frozen_call = removed_norm_sum / calls_with_frozen_basis` when frozen calls are positive; otherwise `0.0`.
- Add minimal active-reassignment evidence:
  - `calls_with_active_reassign_enabled`
  - `calls_with_active_reassign_path` for calls where `enable_cllora_reassign` is true and `null_sketch` is non-null
  - `active_adjustment_norm_sum`, measured as norm between the frozen-only gradient and final output
  - `active_adjustment_norm_max`

The wrapper can compute a local frozen-only projection before calling the original function, then compare that to the original function output. Do not change `_cllora_reassign()` algorithm behavior for this Rank 1 test.

### Adapter sanity check

In `cascades/adapters.py`, keep `_cllora_reassign()` algorithm unchanged for this test:

- Strict frozen projection runs first.
- Active CL-LoRA reassignment is gated by `enable_cllora_reassign` and `null_sketch`.
- Soft EAR is used when `enable_soft_ear=True`.

This experiment only verifies whether the intended gate activates and is observable.

### Freeze and loop-order awareness

Do not change `freeze_current_subspace()` basis threshold, cap, U/V behavior, or task-boundary timing. Do not change `train_cascades()` loop ordering.

Known confounder to document in the result packet: D-MoLE migration currently happens before the freeze step at task boundaries, so the frozen basis may reflect post-migration adapter membership. This is not changed in Rank 1 because loop ordering changes are a separate hypothesis.

### Comparison gate

Preserve the hard validity gate and success threshold in `compare_runs()`:

- reject missing artifacts;
- reject non-`completed` run status;
- reject non-finite metrics or instrumentation;
- reject peak VRAM above 7500 MB;
- reject inactive frozen projection or empty frozen-basis evidence;
- keep success as `delta_bwt_points >= 1.5`, `delta_avg_acc_points >= -2.0`, minimum old-task delta gap `>= -3.0`, under-threshold VRAM, active projection, and non-empty frozen basis.

Active reassignment serialization and instrumentation can be enforced by a CF-cycle-5 sidecar gate before accepting the comparison, or by a backward-compatible optional comparison check. Do not weaken existing comparison criteria.

## 5. Official reduced-suite protocol

Default official path: one treatment-only GPU run using the existing valid CF-cycle-4 seed-43 control copied into a new CF-cycle-5 output root. This is the smallest useful check because the control metrics and task manifest already passed the same reduced `reasoning3`, seed, rank, length, epoch, finite, and VRAM gates.

Use a paired control rerun only if code review or result criticism rejects provisional cross-revision control reuse.

Experiment candidate:

- suite: `reasoning3`
- seed: `43`
- rank: `4`
- max length: `256`
- epochs: `2`
- VRAM threshold: `7500 MB`
- treatment variant: `cllora-active`
- treatment flags: `enable_coso_nullspace=True`, `enable_cllora_reassign=True`, `enable_soft_ear=True`, `ear_gamma=1e-4`
- control provenance: CF-cycle-4 seed-43 control copied from `experiments/cf_cycle_4/nullspace_ablation_seed43/control`

## 6. Required CPU preflight checklist

Run from repository root before any GPU work.

### A. Bytecode compile

Compile every touched file among the protocol-relevant set, plus any new CF-cycle-5 validator if created:

```cmd
python -m py_compile train.py cascades\adapters.py cascades\config.py experiments\cf_cycle_1\run_nullspace_ablation.py experiments\cf_cycle_1\compare_nullspace_ablation.py experiments\cf_cycle_5\validate_active_treatment_gate.py
```

If no new validator file exists, omit `experiments\cf_cycle_5\validate_active_treatment_gate.py`.

### B. Regression tests

```cmd
python -m pytest tests\test_data.py tests\test_cf_cycle_2_guardrails.py -q
```

### C. Seed-43 data preflight at max length 256

```cmd
python experiments\cf_cycle_3\reasoning3_prepare_data_preflight.py --seed 43 --max-length 256 --out experiments\cf_cycle_5\cllora_active_seed43\reasoning3_prepare_data_preflight.json --fail-on-invalid
```

Expected: valid true, three `reasoning3` tasks, zero zero-label batches, max sequence length <= 256. Existing CF-cycle-4 evidence already meets this, but rerunning after code changes gives a local CF-cycle-5 artifact.

### D. Candidate config artifact-distinguishability preflight

After code-mode patches the runner surface, confirm that the candidate treatment config differs from CF-cycle-3/4 frozen-only treatment before launching GPU:

```cmd
python -c "from dataclasses import asdict; import json; from experiments.cf_cycle_1.run_nullspace_ablation import config_for_arm; c=asdict(config_for_arm('treatment', treatment_variant='cllora-active')); print(json.dumps(c, indent=2)); assert c['enable_coso_nullspace']; assert c['enable_cllora_reassign']; assert c['enable_soft_ear']; assert abs(c['ear_gamma'] - 1e-4) < 1e-12"
```

Also confirm the prior frozen-only artifact remains distinguishable:

```cmd
python -c "import json, pathlib; p=pathlib.Path('experiments/cf_cycle_4/nullspace_ablation_seed43/treatment/instrumentation.json'); print('prior_instrumentation_exists=', p.exists()); cfg=json.loads(pathlib.Path('experiments/cf_cycle_4/nullspace_ablation_seed43/treatment/metrics.json').read_text()); print('prior_seed=', cfg.get('seed'), 'prior_rank=', cfg.get('rank'))"
```

The decisive distinguishability check is the new treatment `config.json` after launch: `ablation_config.enable_cllora_reassign` must be true, whereas prior frozen-only treatment serialized it false.

## 7. GPU execution commands — only after explicit permission

### A. Prepare output root with copied provisional control

```cmd
mkdir experiments\cf_cycle_5\cllora_active_seed43
xcopy /E /I /Y experiments\cf_cycle_4\nullspace_ablation_seed43\control experiments\cf_cycle_5\cllora_active_seed43\control
```

### B. Launch treatment only

```cmd
python experiments\cf_cycle_1\run_nullspace_ablation.py --arm treatment --task-suite reasoning3 --seed 43 --epochs 2 --rank 4 --max-length 256 --output-root experiments\cf_cycle_5\cllora_active_seed43 --treatment-variant cllora-active --vram-threshold-mb 7500 > experiments\cf_cycle_5\cllora_active_seed43\treatment.log 2>&1
```

Do not use `--arm both` for the default path. Do not use `--allow-nonfinite` or `--allow-vram-over-threshold`.

### C. Validate treatment gate

```cmd
python experiments\cf_cycle_4\validate_seed43_arm_gate.py --root experiments\cf_cycle_5\cllora_active_seed43 --arm treatment --vram-threshold-mb 7500 > experiments\cf_cycle_5\cllora_active_seed43\treatment_gate.json
```

Then run the CF-cycle-5 active-treatment gate, implemented either as a small validator script or equivalent one-shot Python command, and save:

```cmd
python experiments\cf_cycle_5\validate_active_treatment_gate.py --root experiments\cf_cycle_5\cllora_active_seed43 --arm treatment --out experiments\cf_cycle_5\cllora_active_seed43\treatment_active_gate.json
```

Required active-treatment gate facts:

- `config.ablation_config.enable_coso_nullspace == true`
- `config.ablation_config.enable_cllora_reassign == true`
- `config.ablation_config.enable_soft_ear == true`
- `config.ablation_config.ear_gamma == 1e-4`
- `instrumentation.reassign.calls_with_frozen_basis > 0`
- `instrumentation.reassign.removed_norm_sum > 0`
- `instrumentation.reassign.removed_norm_per_frozen_call` finite
- `instrumentation.reassign.calls_with_active_reassign_path > 0`
- `instrumentation.reassign.active_adjustment_norm_sum` finite, with positive value treated as stronger evidence than zero

### D. Compare against copied seed-43 control

```cmd
python experiments\cf_cycle_1\compare_nullspace_ablation.py --root experiments\cf_cycle_5\cllora_active_seed43 --vram-threshold-mb 7500 > experiments\cf_cycle_5\cllora_active_seed43\comparison.json
```

## 8. Paired-control fallback, not the default path

If a reviewer rejects copied-control comparability, rerun both arms under a separate root with the same patched runner revision:

```cmd
python experiments\cf_cycle_1\run_nullspace_ablation.py --arm control --task-suite reasoning3 --seed 43 --epochs 2 --rank 4 --max-length 256 --output-root experiments\cf_cycle_5\cllora_active_seed43_paired --treatment-variant cllora-active --vram-threshold-mb 7500 > experiments\cf_cycle_5\cllora_active_seed43_paired\control.log 2>&1
python experiments\cf_cycle_1\run_nullspace_ablation.py --arm treatment --task-suite reasoning3 --seed 43 --epochs 2 --rank 4 --max-length 256 --output-root experiments\cf_cycle_5\cllora_active_seed43_paired --treatment-variant cllora-active --vram-threshold-mb 7500 > experiments\cf_cycle_5\cllora_active_seed43_paired\treatment.log 2>&1
python experiments\cf_cycle_1\compare_nullspace_ablation.py --root experiments\cf_cycle_5\cllora_active_seed43_paired --vram-threshold-mb 7500 > experiments\cf_cycle_5\cllora_active_seed43_paired\comparison.json
```

Use this fallback only after the cheaper treatment-only run is invalid, near-threshold, or criticized for control comparability.

## 9. Success gate

The treatment is successful only if all of the following are true:

1. Treatment `run_status.json` reports `status="completed"`.
2. Treatment `metrics.json`, `instrumentation.json`, and `accuracy_matrix.npy` are finite.
3. Peak VRAM is <= 7500 MB.
4. Frozen projection is active: frozen basis non-empty, calls with frozen basis > 0, and strict frozen removed norm sum > 0.
5. Active reassignment is serialized and instrumented: treatment config has `enable_cllora_reassign=True`, active path calls > 0, and active adjustment norm is finite.
6. `comparison.json` is valid and has:
   - `delta_bwt_points >= 1.5`
   - `delta_avg_acc_points >= -2.0`
   - `min(old_task_delta_gaps_points) >= -3.0`
   - `continue == true`

## 10. Verdict labels after execution

- `success`: all success-gate conditions pass.
- `weak positive`: valid artifacts, active projection, active reassignment evidence, finite metrics, under VRAM, BWT improves but `continue=false` because the +1.5 BWT-point threshold is missed.
- `insufficient`: treatment completes but active reassignment evidence is absent/zero or instrumentation cannot distinguish frozen removal from active adjustment.
- `contradicted`: treatment is valid, active reassignment is observable, but BWT does not improve or old-task gaps regress beyond the protocol threshold.
- `invalid`: missing artifacts, failed run status, non-finite values, over-threshold VRAM, inactive frozen projection, or candidate config not artifact-distinguishable.

## 11. What to report back

Report these exact facts after execution or attempted execution:

1. Whether the CL-LoRA-active treatment variant is artifact-distinguishable from CF-cycle-3/4 frozen-only treatment.
2. Files touched and why.
3. CPU preflight commands and raw pass/fail outputs.
4. GPU command actually run, or skipped/blocked reason if not run.
5. Required artifact paths: treatment `config.json`, `run_status.json`, `metrics.json`, `instrumentation.json`, `treatment_gate.json`, `treatment_active_gate.json`, and `comparison.json`.
6. Observed peak VRAM, BWT delta, average-accuracy delta, old-task delta gaps, frozen projection counters, `removed_norm_per_frozen_call`, and active reassignment counters.
7. Result label from the verdict labels above.
8. Recommended next action.

## 12. Stop condition

Stop at this protocol until code-mode implements the minimal runner/instrumentation/gate changes and CPU preflight passes. Do not launch GPU work without explicit bounded permission. Do not modify unrelated treatment logic. Do not escalate to full `current4`, v10, Digital Twin, generative subset, or broad settings fuzzer. Do not claim catastrophic forgetting is solved.
