# CF-cycle-5 Hypothesis Handoff — CL-LoRA-active treatment candidate

Date: 2026-05-17

handoff_target: llm-experiment-designer

## 1. Current objective

Start CF-cycle-5 as a bounded reduced-memory [`reasoning3`](../cf_cycle_1/run_nullspace_ablation.py:32) treatment-redesign cycle for catastrophic-forgetting mitigation.

The cycle question is: can an active CL-LoRA reassignment variant, layered on top of the already confirmed strict frozen null-space projection, produce a stronger BWT improvement without violating RTX 4060 Ti 8GB guardrails?

This is not a full [`current4`](../cf_cycle_1/run_nullspace_ablation.py:26), v10, Digital Twin, generative subset, or broad settings-fuzzer escalation cycle.

## 2. Current evidence state

### Confirmed

- Seed 43 produced valid sequential control/treatment artifacts under the 7500 MB threshold; [`comparison.json`](../cf_cycle_4/nullspace_ablation_seed43/comparison.json:1) reports valid true, projection active, frozen basis non-empty, `delta_bwt_points=-0.008929480489536235`, and `continue=false`.
- Seed 42 produced valid reduced-memory comparison after the data-loader fix; [`comparison.json`](../cf_cycle_3/nullspace_ablation_retry/comparison.json:1) reports `delta_bwt_points=0.36167033073388477`, active projection, frozen basis non-empty, and `continue=false`.
- The existing wrapper tested strict frozen projection for treatment but disabled active CL-LoRA reassignment for both arms in [`config_for_arm()`](../cf_cycle_1/run_nullspace_ablation.py:79): treatment sets `enable_coso_nullspace=True`, while `enable_cllora_reassign=False` for both control and treatment.
- The production default in [`AblationConfig`](../../cascades/config.py:15) sets [`enable_cllora_reassign`](../../cascades/config.py:32) to true.
- Strict frozen projection runs before the active CL-LoRA gate in [`_cllora_reassign()`](../../cascades/adapters.py:176), so prior runs tested frozen-basis projection but bypassed the active-sketch reassignment and soft-EAR path when [`enable_cllora_reassign`](../../cascades/adapters.py:195) was false.

### Likely

- Isolated frozen-basis projection is feasible but too weak, mistimed, under-capacity, or confounded. Confidence: medium.
- A one-knob reduced-suite treatment variant has higher expected value than full-suite escalation. Confidence: high for engineering value, medium for algorithmic payoff.

### Plausible ranked hypotheses

1. Active CL-LoRA reassignment on top of strict frozen projection is missing treatment strength. Confidence: medium; recommended first.
2. Frozen-basis capacity, threshold, or update cadence is too conservative. Confidence: medium-low; second if Rank 1 fails validly.
3. Freeze timing is suboptimal because D-MoLE migration occurs before freeze. Confidence: low-medium; third because it changes loop ordering.
4. D-MoLE migration confounds protected adapter selection. Confidence: low-medium; inspect/instrument before changing.
5. Soft EAR gamma matters after active reassignment is enabled. Confidence: low until Rank 1 produces active-reassignment artifacts.
6. Tiny staged settings fuzzer may help only after the obvious one-knob test fails validly. Confidence: low.

### Weak signal

- Seed 43 projection activity was positive but modest: [`instrumentation.json`](../cf_cycle_4/nullspace_ablation_seed43/treatment/instrumentation.json:674) reports 6900 total calls, 3600 frozen-basis calls, max frozen columns 5, and removed norm sum 3.100088352795865.
- No normalized removed-norm-per-call or active-sketch adjustment baseline exists yet.

### Contradicted

- Isolated frozen-nullspace algorithmic success is contradicted or unsupported across seeds 42 and 43 because both miss the plus 1.5 BWT-point gate.
- Lowering the threshold after seeing results is not supported.

### Blocked

- Full [`current4`](../cf_cycle_1/run_nullspace_ablation.py:26), v10, Digital Twin, and generative-subset escalation remain blocked.
- Catastrophic forgetting is not solved.

## 3. Next mode to use

Use llm-experiment-designer.

## 4. Specific task for that mode

Design one executable reduced-suite protocol for the CL-LoRA-active treatment variant. The expected result is a concrete experiment protocol and CPU preflight checklist; code changes or GPU execution should occur only if the mode determines they are necessary and allowed.

The experiment designer should inspect or specify these exact areas:

1. Runner config surface:
   - Inspect [`config_for_arm()`](../cf_cycle_1/run_nullspace_ablation.py:79).
   - Add or specify a one-knob treatment variant where treatment serializes `enable_cllora_reassign=True` while preserving `enable_coso_nullspace=True`.
   - Keep control comparable and avoid silently changing control behavior unless a paired rerun is explicitly selected.

2. Instrumentation:
   - Inspect [`install_instrumentation()`](../cf_cycle_1/run_nullspace_ablation.py:152).
   - Preserve counters for `calls_total`, `calls_with_null_sketch`, `calls_with_frozen_basis`, `max_frozen_cols`, and `removed_norm_sum`.
   - Add or compute `removed_norm_per_frozen_call`.
   - Add minimal active-reassignment evidence if feasible: count calls where `enable_cllora_reassign` is true and `null_sketch` is non-null, plus an active-sketch adjustment norm distinct from frozen-basis removal.

3. Adapter logic sanity check:
   - Inspect [`_cllora_reassign()`](../../cascades/adapters.py:176), especially the early return at [`enable_cllora_reassign`](../../cascades/adapters.py:195) and soft-EAR call at [`soft_ear()`](../../cascades/adapters.py:207).
   - Do not redesign the algorithm in this step; verify that the intended gate activates and is observable.

4. Freeze and loop-order awareness:
   - Inspect [`freeze_current_subspace()`](../../cascades/adapters.py:366) only to document current basis behavior; do not change basis threshold/capacity for this Rank 1 experiment.
   - Inspect [`train_cascades()`](../../train.py:196), D-MoLE migration at [`train.py`](../../train.py:413), and freeze at [`train.py`](../../train.py:441) only to preserve current ordering and document confounders.

5. Comparison gate:
   - Inspect [`compare_runs()`](../cf_cycle_1/compare_nullspace_ablation.py:105) and the continue criteria at [`comparison["continue"]`](../cf_cycle_1/compare_nullspace_ablation.py:164).
   - Ensure the final comparison rejects missing or failed run status, non-finite metrics, VRAM above 7500 MB, missing frozen-basis evidence, and inactive projection.

Recommended experiment candidate:

- Suite: reduced [`reasoning3`](../cf_cycle_1/run_nullspace_ablation.py:32).
- Seed: 43.
- Rank: 4.
- Max length: 256.
- Epochs: 2.
- VRAM threshold: 7500 MB.
- Treatment: `enable_coso_nullspace=True`, `enable_cllora_reassign=True`, `enable_soft_ear=True`, `ear_gamma=1e-4`.
- Control: existing valid seed-43 control may be reused as a provisional cheap check; paired rerun is cleaner if runner or instrumentation changes affect comparability.

Required pre-GPU CPU gates:

- Bytecode compile any touched files among [`train.py`](../../train.py:1), [`cascades/adapters.py`](../../cascades/adapters.py:1), [`cascades/config.py`](../../cascades/config.py:1), [`run_nullspace_ablation.py`](../cf_cycle_1/run_nullspace_ablation.py:1), and [`compare_nullspace_ablation.py`](../cf_cycle_1/compare_nullspace_ablation.py:1).
- Run [`tests/test_data.py`](../../tests/test_data.py:1) and [`tests/test_cf_cycle_2_guardrails.py`](../../tests/test_cf_cycle_2_guardrails.py:1).
- Reuse or rerun seed-43 data preflight using [`reasoning3_prepare_data_preflight.py`](../cf_cycle_3/reasoning3_prepare_data_preflight.py:1) at max length 256.
- Confirm candidate treatment config is artifact-distinguishable from CF-cycle-3/4 frozen-only treatment.

Success gate:

- Completed treatment [`run_status.json`](../cf_cycle_4/nullspace_ablation_seed43/treatment/run_status.json:1)-style artifact.
- Finite [`metrics.json`](../cf_cycle_4/nullspace_ablation_seed43/treatment/metrics.json:1)-style artifact.
- Peak VRAM at or below 7500 MB.
- Active projection: non-empty frozen basis, calls with frozen basis greater than zero, removed norm sum greater than zero.
- Active reassignment serialized and instrumented.
- Valid comparison with `delta_bwt_points >= 1.5`, `delta_avg_acc_points >= -2.0`, minimum old-task delta gap `>= -3.0`, and `continue=true` under [`compare_runs()`](../cf_cycle_1/compare_nullspace_ablation.py:105).

## 5. Stop condition

Stop after producing an executable protocol and CPU preflight checklist for the CL-LoRA-active reduced-suite experiment. Do not run GPU jobs unless explicitly executing the experiment in a mode that can do so. Do not modify unrelated treatment logic. Do not escalate to full [`current4`](../cf_cycle_1/run_nullspace_ablation.py:26), v10, Digital Twin, or a broad settings fuzzer. Do not claim catastrophic forgetting is solved.

## 6. What to report back

Report:

- Whether the CL-LoRA-active treatment variant is artifact-distinguishable from prior frozen-only treatment.
- Exact files to touch or inspect and why.
- CPU preflight checklist and expected commands.
- Planned reduced [`reasoning3`](../cf_cycle_1/run_nullspace_ablation.py:32) command line, seed 43, rank 4, max length 256, two epochs, 7500 MB threshold.
- Required artifacts: treatment config, run status, metrics, instrumentation, treatment gate, and comparison JSON.
- Verdict labels to use after execution: success, weak positive, insufficient, contradicted, or invalid.
- Explicit non-goals and blocked escalation list.

## Active cycle record

- cycle_objective: CF-cycle-5 CL-LoRA-active treatment candidate for catastrophic-forgetting mitigation.
- cycle_number_or_label: CF-cycle-5.
- current_hypothesis: frozen-only treatment underperformed because the ablation disabled active CL-LoRA reassignment/soft-EAR despite confirming strict frozen projection activity.
- current_evidence_summary: active projection feasible but below threshold across seed 42 and seed 43; active reassignment path confirmed off in prior wrapper.
- active_handoff_target: llm-experiment-designer.
- stop_or_continue_condition: continue only to an executable reduced-suite protocol and CPU preflight gates before any GPU execution.
- files_touched_by_coordinator: [`experiments/cf_cycle_5/hypothesis_handoff.md`](hypothesis_handoff.md:1) and [`CONTEXT.md`](../../CONTEXT.md:1).
- commands_run_by_coordinator: none; evidence was read from existing reports and handoffs.
