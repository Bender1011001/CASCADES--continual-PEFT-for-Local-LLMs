# CF-cycle-6 Result Critic Packet — CL-LoRA-active frozen-basis threshold 0.02 treatment-only run

## Handoff target

- Next mode: `llm-result-critic`.
- Requested critic task: assess whether the bounded seed-43 treatment-only evidence supports, weakens, or contradicts lowering the CL-LoRA-active frozen-basis admission threshold from 0.05 to 0.02.

## Hypothesis tested

Lowering [`frozen_basis_variance_threshold`](../../cascades/config.py:55) for the CL-LoRA-active treatment variant from 0.05 to 0.02 should admit a stronger frozen basis, increase useful frozen projection/reassignment evidence, and improve old-task retention versus the copied seed-43 control without breaching the 7500 MB VRAM gate.

## Experiment performed

- Scope: treatment-only reduced [`reasoning3`](../cf_cycle_1/run_nullspace_ablation.py:32) run, seed 43, rank 4, max length 256, two epochs, 7500 MB VRAM threshold.
- Variant: [`cllora-active-freeze-thresh-02`](../cf_cycle_1/run_nullspace_ablation.py:79), preserving CL-LoRA-active flags and changing only the frozen-basis threshold to 0.02.
- Control: copied valid seed-43 CF-cycle-4 control from [`experiments/cf_cycle_4/nullspace_ablation_seed43/control`](../cf_cycle_4/nullspace_ablation_seed43/control) into [`experiments/cf_cycle_6/cllora_active_freeze_thresh_02_seed43/control`](cllora_active_freeze_thresh_02_seed43/control).
- Stop condition honored: no full [`current4`](../cf_cycle_1/run_nullspace_ablation.py:26), v10, Digital Twin, generative subset, broad fuzzer, or paired fallback was launched.

## Files changed

- Added this evidence packet: [`experiments/cf_cycle_6/cllora_active_freeze_thresh_02_result_critic_packet.md`](cllora_active_freeze_thresh_02_result_critic_packet.md:1).
- Updated project memory: [`CONTEXT.md`](../../CONTEXT.md:1).
- Generated experiment artifacts under [`experiments/cf_cycle_6/cllora_active_freeze_thresh_02_seed43`](cllora_active_freeze_thresh_02_seed43).

## Commands/checks run

1. Prepared output root and attempted approved conditional control copy:
   - `if not exist experiments\cf_cycle_6\cllora_active_freeze_thresh_02_seed43 mkdir experiments\cf_cycle_6\cllora_active_freeze_thresh_02_seed43 && xcopy /E /I /Y experiments\cf_cycle_4\nullspace_ablation_seed43\control experiments\cf_cycle_6\cllora_active_freeze_thresh_02_seed43\control`
   - Exit code: 0.
   - Later finding: because the root directory already existed, the conditional chain skipped the copy; this was detected by the comparison failure.
2. Ran treatment only:
   - `python experiments\cf_cycle_1\run_nullspace_ablation.py --arm treatment --task-suite reasoning3 --seed 43 --epochs 2 --rank 4 --max-length 256 --output-root experiments\cf_cycle_6\cllora_active_freeze_thresh_02_seed43 --treatment-variant cllora-active-freeze-thresh-02 --vram-threshold-mb 7500 > experiments\cf_cycle_6\cllora_active_freeze_thresh_02_seed43\treatment.log 2>&1`
   - Completion artifact: [`run_status.json`](cllora_active_freeze_thresh_02_seed43/treatment/run_status.json:1).
3. Ran standard treatment gate:
   - `python experiments\cf_cycle_4\validate_seed43_arm_gate.py --root experiments\cf_cycle_6\cllora_active_freeze_thresh_02_seed43 --arm treatment --vram-threshold-mb 7500 > experiments\cf_cycle_6\cllora_active_freeze_thresh_02_seed43\treatment_gate.json`
   - Output: [`treatment_gate.json`](cllora_active_freeze_thresh_02_seed43/treatment_gate.json:1).
4. Ran active/threshold treatment gate:
   - `python experiments\cf_cycle_5\validate_active_treatment_gate.py --root experiments\cf_cycle_6\cllora_active_freeze_thresh_02_seed43 --arm treatment --expected-variant cllora-active-freeze-thresh-02 --expected-frozen-basis-variance-threshold 0.02 --out experiments\cf_cycle_6\cllora_active_freeze_thresh_02_seed43\treatment_active_gate.json`
   - Output: [`treatment_active_gate.json`](cllora_active_freeze_thresh_02_seed43/treatment_active_gate.json:1).
5. First comparison attempt:
   - `python experiments\cf_cycle_1\compare_nullspace_ablation.py --root experiments\cf_cycle_6\cllora_active_freeze_thresh_02_seed43 --vram-threshold-mb 7500 > experiments\cf_cycle_6\cllora_active_freeze_thresh_02_seed43\comparison.json`
   - Exit code: 2.
   - Failure in [`comparison.json`](cllora_active_freeze_thresh_02_seed43/comparison.json:1): control missing required artifacts.
6. Root-cause correction and approved comparison rerun:
   - `xcopy /E /I /Y experiments\cf_cycle_4\nullspace_ablation_seed43\control experiments\cf_cycle_6\cllora_active_freeze_thresh_02_seed43\control && python experiments\cf_cycle_1\compare_nullspace_ablation.py --root experiments\cf_cycle_6\cllora_active_freeze_thresh_02_seed43 --vram-threshold-mb 7500 > experiments\cf_cycle_6\cllora_active_freeze_thresh_02_seed43\comparison.json`
   - Exit code: 0; 8 control files copied.
   - Final output: [`comparison.json`](cllora_active_freeze_thresh_02_seed43/comparison.json:1).

## Raw or summarized results

### Treatment configuration

- [`config.json`](cllora_active_freeze_thresh_02_seed43/treatment/config.json:1): variant `cllora-active-freeze-thresh-02`, [`frozen_basis_variance_threshold`](../../cascades/config.py:55) set to 0.02, [`enable_coso_nullspace`](../../cascades/config.py:31) true, [`enable_cllora_reassign`](../../cascades/config.py:32) true, [`enable_soft_ear`](../../cascades/config.py:38) true, [`ear_gamma`](../../cascades/config.py:37) 0.0001.

### Run status

- [`run_status.json`](cllora_active_freeze_thresh_02_seed43/treatment/run_status.json:1): completed.
- Peak VRAM: 6843.90966796875 MB, below 7500 MB.
- Wall time: 1867.082189321518 seconds.

### Metrics

- [`metrics.json`](cllora_active_freeze_thresh_02_seed43/treatment/metrics.json:1):
  - Average accuracy: 0.5669313066796887.
  - BWT: -0.026740575678466633.
  - Final accuracies: `[0.3913399701840882, 0.5441172124613676, 0.7653367373936102]`.
  - Diagonal accuracies: `[0.4319063638056756, 0.5570319701967135, 0.7653367373936102]`.
  - Old-task deltas: `[-0.04056639362158737, -0.012914757735345894]`.

### Instrumentation

- [`instrumentation.json`](cllora_active_freeze_thresh_02_seed43/treatment/instrumentation.json:292): 38 freeze events.
- Retained-column freeze events for CF-cycle-6: `[(3, 4), (4, 6), (3, 5), (3, 3), (5, 6), (4, 5), (6, 9), (9, 12), (6, 11), (7, 9), (10, 12), (9, 10)]`.
- [`reassign`](cllora_active_freeze_thresh_02_seed43/treatment/instrumentation.json:674) summary:
  - Calls total: 6900.
  - Calls with null sketch: 6900.
  - Calls with frozen basis: 3600.
  - Maximum frozen columns: 6.
  - Removed norm sum: 1.8263498862379492.
  - Removed norm per frozen call: 0.0005073194128438748.
  - Calls with active reassignment enabled: 6900.
  - Calls with active reassignment path: 6900.
  - Active adjustment norm sum: 48.84042342959583.
  - Active adjustment norm max: 0.12196692824363708.

### Gates

- [`treatment_gate.json`](cllora_active_freeze_thresh_02_seed43/treatment_gate.json:1): valid true; finite metrics/instrumentation; finite 3x3 accuracy matrix; projection active true; frozen basis non-empty true; 3600 calls with frozen basis; removed norm sum 1.8263498862379492.
- [`treatment_active_gate.json`](cllora_active_freeze_thresh_02_seed43/treatment_active_gate.json:1): valid true; threshold 0.02 validated; active reassignment path count positive; active adjustment norm finite and positive.

### Comparison against copied seed-43 control

- [`comparison.json`](cllora_active_freeze_thresh_02_seed43/comparison.json:1):
  - Valid: true.
  - BWT delta: -0.018122692840660815 points.
  - Average-accuracy delta: 0.03830429581168637 points.
  - Old-task gaps: `[0.559717469671106, -0.5959628553524277]` points.
  - Control peak VRAM: 6843.90966796875 MB.
  - Treatment peak VRAM: 6843.90966796875 MB.
  - Projection active: true.
  - Frozen basis non-empty: true.
  - Continue flag: false.

### Frozen-basis strength compared with CF-cycle-5

- CF-cycle-5 baseline from [`instrumentation.json`](../cf_cycle_5/cllora_active_seed43/treatment/instrumentation.json:674):
  - `reassign.max_frozen_cols`: 5.
  - `removed_norm_per_frozen_call`: 0.000609848923952288.
  - Active adjustment norm sum: 49.50860421800462.
- CF-cycle-6 threshold 0.02 result:
  - `reassign.max_frozen_cols`: 6, stronger by column count.
  - `removed_norm_per_frozen_call`: 0.0005073194128438748, weaker than CF-cycle-5 baseline.
  - Material-strength threshold for normalized removal: 0.00076231115494036.
  - Material-strength threshold met: false.

## Observed facts versus interpretation

### Observed facts

- The treatment arm completed and stayed under the VRAM threshold.
- Standard and active/threshold gates are valid.
- Active reassignment path executed on 6900 calls.
- Frozen projection executed on 3600 calls.
- The copied-control comparison is valid after the unconditional control copy.
- The comparison continue flag is false.
- BWT delta is negative, average-accuracy delta is slightly positive, and old-task gaps are mixed.
- The lower threshold increased maximum frozen columns from 5 to 6 relative to CF-cycle-5, but decreased normalized removed norm and missed the material-strength threshold.

### Interpretation

- Lowering the threshold to 0.02 produced a mechanically valid and active treatment, but did not improve the decision metric.
- Evidence weakens the specific hypothesis that simply admitting more frozen-basis columns at 0.02 increases useful frozen-basis strength.
- This is a bounded reduced-suite smoke/proxy result, not proof about all seeds or the full task set.

## Checks not run and why

- Full [`current4`](../cf_cycle_1/run_nullspace_ablation.py:26): skipped by explicit stop condition.
- v10 and Digital Twin experiments: skipped by explicit stop condition.
- Generative subset: skipped by explicit stop condition.
- Broad fuzzer: skipped by explicit stop condition.
- Paired fallback: skipped by explicit stop condition; treatment-only path was not blocked.
- Additional seeds: skipped by explicit bounded experiment approval.

## Result label

- Verdict label: contradicted / weak signal.
- Confidence level: medium for this seed-43 reduced-suite decision; low for broad generalization.

## Recommended next action

Do not proceed to larger tasks from this threshold-only change. The critic should mark the 0.02 threshold variant as below decision threshold and recommend a revised reduced-suite treatment design rather than full [`current4`](../cf_cycle_1/run_nullspace_ablation.py:26), v10, Digital Twin, generative subset, broad fuzzer, or paired fallback escalation.
