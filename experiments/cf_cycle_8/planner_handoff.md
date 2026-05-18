# CF-cycle-8 Planner Handoff — paired same-seed top-k validation

Date: 2026-05-17

handoff_target: llm-hypothesis-generator

## 1. Cycle objective

Cycle question: does a fresh paired same-seed reduced [`reasoning3`](experiments/cf_cycle_1/run_nullspace_ablation.py:32) control-versus-treatment validation confirm that cllora-active-freeze-topk-2 has enough reproducible utility and mechanism evidence to justify result-critic review, without broader escalation or solution language?

Planner objective: create a bounded validation packet seeded by [`experiments/cf_cycle_7/REPORT.md`](experiments/cf_cycle_7/REPORT.md:1) and [`experiments/cf_cycle_8/coordinator_handoff.md`](experiments/cf_cycle_8/coordinator_handoff.md:1). This planner step does not run GPU jobs, does not modify algorithm code, and does not redesign cadence or timing.

Recommended next mode: llm-hypothesis-generator.

Reason: the treatment protocol is already fixed, but the next mode should produce narrow validation hypotheses and expected artifact patterns before experiment design. It should not reopen broad mechanism search.

## 2. Current assumptions

- Assumption A1: reduced [`reasoning3`](experiments/cf_cycle_1/run_nullspace_ablation.py:32), seed 43, rank 4, max length 256, two epochs, and a 7500 MB VRAM guardrail remain the cheapest valid reduced validation envelope.
- Assumption A2: CF-cycle-7 copied-control evidence is useful as a seed but not enough for promotion; CF-cycle-8 must use a fresh control and a fresh treatment under the same code revision.
- Assumption A3: cllora-active-freeze-topk-2 is the only treatment arm in scope. Do not test rank changes, threshold-only variants, cadence changes, D-MoLE timing changes, or broad settings grids.
- Assumption A4: the active/capacity gate in [`experiments/cf_cycle_5/validate_active_treatment_gate.py`](experiments/cf_cycle_5/validate_active_treatment_gate.py:39) returns validity for config and active-path evidence, while mechanism threshold booleans are reported in the payload. The next worker must inspect those threshold booleans explicitly.
- Assumption A5: even a clean paired reduced-suite pass would be provisional. It can justify result-critic review or another bounded validation step, not a catastrophic-forgetting solution claim.

## 3. Evidence summary by confidence label

### Confirmed

- CF-cycle-7 closed with valid execution and a durable report in [`experiments/cf_cycle_7/REPORT.md`](experiments/cf_cycle_7/REPORT.md:1).
- The cllora-active-freeze-topk-2 treatment completed under reduced [`reasoning3`](experiments/cf_cycle_1/run_nullspace_ablation.py:32), seed 43, rank 4, max length 256, two epochs, and the 7500 MB guardrail. [`experiments/cf_cycle_7/REPORT.md`](experiments/cf_cycle_7/REPORT.md:56) records peak VRAM 6843.90966796875 MB.
- The standard treatment gate is valid, with finite artifacts, finite 3x3 accuracy matrix, non-empty frozen basis, 3600 frozen-basis calls, removed norm sum 3.530579238988139, projection active true, and 38 freeze events in [`experiments/cf_cycle_7/REPORT.md`](experiments/cf_cycle_7/REPORT.md:57).
- The active/capacity treatment gate is valid with treatment variant cllora-active-freeze-topk-2, frozen-basis threshold 0.05, top k per freeze 2, active config checks true, and active instrumentation checks true in [`experiments/cf_cycle_7/REPORT.md`](experiments/cf_cycle_7/REPORT.md:58).
- Mechanism strength cleared the CF-cycle-6 baseline, CF-cycle-5 baseline, and material threshold: removed norm sum 3.530579238988139, removed norm per frozen call 0.000980716455274483, and max frozen columns 2 in [`experiments/cf_cycle_7/REPORT.md`](experiments/cf_cycle_7/REPORT.md:59).
- Active reassignment remained active with 6900 active reassignment path calls and active adjustment norm sum 50.596070232306374 in [`experiments/cf_cycle_7/REPORT.md`](experiments/cf_cycle_7/REPORT.md:60).
- The copied-control comparison is valid and reports projection active true, frozen basis non-empty true, and no comparison failures in [`experiments/cf_cycle_7/REPORT.md`](experiments/cf_cycle_7/REPORT.md:61).

### Likely

- Top-k frozen-basis admission is useful enough to justify paired same-seed validation. The copied-control comparison reported delta BWT plus 0.2983417459082205 points and delta average accuracy plus 0.2692983654684511 points in [`experiments/cf_cycle_7/REPORT.md`](experiments/cf_cycle_7/REPORT.md:62).

### Plausible

- Top-k admission improved useful frozen-basis selectivity by admitting at most two strongest directions per freeze. This is plausible because normalized frozen removal cleared all reference thresholds in [`experiments/cf_cycle_7/REPORT.md`](experiments/cf_cycle_7/REPORT.md:63).

### Weak signal

- Utility is positive but below success because the copied-control comparison reported positive deltas while continue was false in [`experiments/cf_cycle_7/REPORT.md`](experiments/cf_cycle_7/REPORT.md:64).
- Old-task gaps remain mixed: 0.7995196940998484 and -0.20283620228340737 in [`experiments/cf_cycle_7/REPORT.md`](experiments/cf_cycle_7/REPORT.md:65).

### Contradicted

- Success, promotion, and larger-run escalation are contradicted or unsupported. The preserved result label is likely useful, not success, in [`experiments/cf_cycle_7/REPORT.md`](experiments/cf_cycle_7/REPORT.md:66).
- Catastrophic forgetting is not solved by this evidence. [`experiments/cf_cycle_7/REPORT.md`](experiments/cf_cycle_7/REPORT.md:67) explicitly limits interpretation to mechanism activation plus bounded positive utility.

### Blocked

- Full [`current4`](experiments/cf_cycle_1/run_nullspace_ablation.py:26), v10, Digital Twin, generative subset, broad fuzzer, cadence/timing redesign, rank changes, and catastrophic-forgetting solution language remain blocked until paired same-seed reduced validation is completed and reviewed, as recorded in [`experiments/cf_cycle_7/REPORT.md`](experiments/cf_cycle_7/REPORT.md:68).

## 4. Files and areas likely involved

These are execution and evidence surfaces only. Do not modify algorithm code in this planner cycle.

- Task envelope definitions: [`CURRENT4`](experiments/cf_cycle_1/run_nullspace_ablation.py:26) and [`REASONING3`](experiments/cf_cycle_1/run_nullspace_ablation.py:32).
- Treatment variant serialization: [`config_for_arm()`](experiments/cf_cycle_1/run_nullspace_ablation.py:79), especially the cllora-active-freeze-topk-2 branch at [`run_nullspace_ablation.py`](experiments/cf_cycle_1/run_nullspace_ablation.py:106).
- Run artifact writer and guardrail surface: [`run_arm()`](experiments/cf_cycle_1/run_nullspace_ablation.py:324).
- Runner CLI surface: [`parse_args()`](experiments/cf_cycle_1/run_nullspace_ablation.py:440).
- Standard seed-43 gate: [`validate_arm()`](experiments/cf_cycle_4/validate_seed43_arm_gate.py:76).
- Active/capacity treatment gate: [`validate_active_treatment()`](experiments/cf_cycle_5/validate_active_treatment_gate.py:39).
- Comparison gate and continue threshold: [`compare_runs()`](experiments/cf_cycle_1/compare_nullspace_ablation.py:105).
- Optional CPU data preflight: [`scan_reasoning3()`](experiments/cf_cycle_3/reasoning3_prepare_data_preflight.py:21).

Expected evidence sources for the next worker:

- Coordinator seed: [`experiments/cf_cycle_8/coordinator_handoff.md`](experiments/cf_cycle_8/coordinator_handoff.md:1).
- Prior closeout: [`experiments/cf_cycle_7/REPORT.md`](experiments/cf_cycle_7/REPORT.md:1).
- Prior treatment gate: [`experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/treatment_gate.json`](experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/treatment_gate.json:1).
- Prior active/capacity gate: [`experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/treatment_active_gate.json`](experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/treatment_active_gate.json:1).
- Prior copied-control comparison: [`experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/comparison.json`](experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/comparison.json:1).
- Project memory: [`CONTEXT.md`](CONTEXT.md:1).

## 5. Exact bounded protocol scope

Required validation shape:

- Task suite: reduced [`reasoning3`](experiments/cf_cycle_1/run_nullspace_ablation.py:32).
- Seed: 43.
- Rank: 4.
- Max length: 256.
- Epochs: 2.
- VRAM guardrail: 7500 MB.
- Treatment variant: cllora-active-freeze-topk-2.
- Frozen-basis admission threshold: 0.05.
- Frozen-basis top k per freeze: 2.
- Output root recommendation: [`experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired).
- Arm policy: run fresh control first, gate it, then run fresh treatment, gate it, then compare.
- Same-revision policy: both arms must be run under the same code revision and same reduced envelope. Compare the git revision fields in each arm artifact before accepting the comparison.
- Disallowed runner flags: do not use allow-nonfinite behavior and do not allow VRAM over-threshold behavior.
- Disallowed shortcuts: do not copy the CF-cycle-4 control, do not reuse CF-cycle-7 copied-control artifacts, and do not use a stale output root containing prior arm artifacts.

## 6. Required artifacts and gates

### Per-arm artifacts

Each accepted arm must contain all artifacts required by [`REQUIRED_ARTIFACTS`](experiments/cf_cycle_4/validate_seed43_arm_gate.py:19):

- [`config.json`](experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/treatment/config.json:1)-style configuration artifact.
- [`task_manifest.json`](experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/treatment/task_manifest.json:1)-style task manifest artifact.
- [`accuracy_matrix.npy`](experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/treatment/accuracy_matrix.npy)-style accuracy matrix artifact.
- [`cascades_results.csv`](experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/treatment/cascades_results.csv:1)-style CSV result artifact.
- [`metrics.json`](experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/treatment/metrics.json:1)-style metrics artifact.
- [`instrumentation.json`](experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/treatment/instrumentation.json:1)-style instrumentation artifact.
- [`run_status.json`](experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/treatment/run_status.json:1)-style run status artifact.

### Standard gates

- Control standard gate: use [`experiments/cf_cycle_4/validate_seed43_arm_gate.py`](experiments/cf_cycle_4/validate_seed43_arm_gate.py:1) with arm control and write [`control_gate.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/control_gate.json).
- Treatment standard gate: use [`experiments/cf_cycle_4/validate_seed43_arm_gate.py`](experiments/cf_cycle_4/validate_seed43_arm_gate.py:1) with arm treatment and write [`treatment_gate.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/treatment_gate.json).
- Both standard gates must report valid true, completed status, expected reduced [`reasoning3`](experiments/cf_cycle_1/run_nullspace_ablation.py:32) manifest paths, finite metrics, finite instrumentation, finite 3x3 accuracy matrix, and peak VRAM at or below 7500 MB.
- Treatment standard gate must additionally show non-empty frozen basis and projection active true, matching the treatment projection requirement in [`validate_seed43_arm_gate.py`](experiments/cf_cycle_4/validate_seed43_arm_gate.py:129).

### Active/capacity treatment gate

- Use [`experiments/cf_cycle_5/validate_active_treatment_gate.py`](experiments/cf_cycle_5/validate_active_treatment_gate.py:1) and write [`treatment_active_gate.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/treatment_active_gate.json).
- Required config checks: treatment variant cllora-active-freeze-topk-2, enable_coso_nullspace true, enable_cllora_reassign true, enable_soft_ear true, ear_gamma 1e-4, frozen-basis threshold 0.05, and top k per freeze 2.
- Required active evidence: calls with frozen basis greater than zero, removed norm sum positive, removed norm per frozen call finite, calls with active reassignment path greater than zero, and active adjustment norm sum positive.
- Mechanism threshold evidence to inspect explicitly: removed norm per frozen call should remain above the CF-cycle-6 baseline 0.0005073194128438748, CF-cycle-5 baseline 0.000609848923952288, and material-strength threshold 0.00076231115494036, as reported by [`validate_active_treatment_gate.py`](experiments/cf_cycle_5/validate_active_treatment_gate.py:166).

### Final comparison gate

- Use [`experiments/cf_cycle_1/compare_nullspace_ablation.py`](experiments/cf_cycle_1/compare_nullspace_ablation.py:1) only after both fresh arms are completed, finite, standard-gate valid, same-revision, and under the VRAM guardrail.
- Write [`comparison.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/comparison.json).
- The comparison must report valid true, no failures, projection active true, frozen basis non-empty true, both arm peaks at or below 7500 MB, delta BWT points, delta average accuracy points, old-task gap points, and continue decision.

## 7. Candidate checks or commands

Planner mode must not run these commands. They are the bounded command skeleton for the next execution-design mode.

Optional CPU data preflight:

```cmd
python experiments\cf_cycle_3\reasoning3_prepare_data_preflight.py --seed 43 --max-length 256 --out experiments\cf_cycle_8\cllora_active_freeze_topk2_seed43_paired\reasoning3_prepare_data_preflight.json --fail-on-invalid
```

Fresh control arm:

```cmd
python experiments\cf_cycle_1\run_nullspace_ablation.py --arm control --task-suite reasoning3 --seed 43 --epochs 2 --rank 4 --max-length 256 --output-root experiments\cf_cycle_8\cllora_active_freeze_topk2_seed43_paired --vram-threshold-mb 7500 > experiments\cf_cycle_8\cllora_active_freeze_topk2_seed43_paired\control.log 2>&1
```

Control standard gate:

```cmd
python experiments\cf_cycle_4\validate_seed43_arm_gate.py --root experiments\cf_cycle_8\cllora_active_freeze_topk2_seed43_paired --arm control --vram-threshold-mb 7500 > experiments\cf_cycle_8\cllora_active_freeze_topk2_seed43_paired\control_gate.json
```

Fresh treatment arm, only after the fresh control gate is valid:

```cmd
python experiments\cf_cycle_1\run_nullspace_ablation.py --arm treatment --task-suite reasoning3 --seed 43 --epochs 2 --rank 4 --max-length 256 --treatment-variant cllora-active-freeze-topk-2 --output-root experiments\cf_cycle_8\cllora_active_freeze_topk2_seed43_paired --vram-threshold-mb 7500 > experiments\cf_cycle_8\cllora_active_freeze_topk2_seed43_paired\treatment.log 2>&1
```

Treatment standard gate:

```cmd
python experiments\cf_cycle_4\validate_seed43_arm_gate.py --root experiments\cf_cycle_8\cllora_active_freeze_topk2_seed43_paired --arm treatment --vram-threshold-mb 7500 > experiments\cf_cycle_8\cllora_active_freeze_topk2_seed43_paired\treatment_gate.json
```

Treatment active/capacity gate:

```cmd
python experiments\cf_cycle_5\validate_active_treatment_gate.py --root experiments\cf_cycle_8\cllora_active_freeze_topk2_seed43_paired --arm treatment --out experiments\cf_cycle_8\cllora_active_freeze_topk2_seed43_paired\treatment_active_gate.json --expected-variant cllora-active-freeze-topk-2 --expected-frozen-basis-variance-threshold 0.05 --expected-frozen-basis-top-k-per-freeze 2
```

Final comparison, only after all fresh-arm gates are accepted:

```cmd
python experiments\cf_cycle_1\compare_nullspace_ablation.py --root experiments\cf_cycle_8\cllora_active_freeze_topk2_seed43_paired --vram-threshold-mb 7500 > experiments\cf_cycle_8\cllora_active_freeze_topk2_seed43_paired\comparison.json
```

## 8. Realistic success threshold

### Planner success

This planner cycle succeeds if [`experiments/cf_cycle_8/planner_handoff.md`](experiments/cf_cycle_8/planner_handoff.md:1) exists and gives the next mode a narrow validation protocol, gate criteria, artifact expectations, failure criteria, and stop condition without launching validation.

### Evidence-usable validation threshold

The future validation is evidence-usable if:

- Fresh control and fresh treatment both complete.
- Both standard gates are valid.
- The treatment active/capacity gate is valid.
- Mechanism threshold booleans are inspected and reported.
- Both arms are finite and under 7500 MB peak VRAM.
- Both arms use the same reduced [`reasoning3`](experiments/cf_cycle_1/run_nullspace_ablation.py:32) envelope and same code revision.
- Final comparison is valid.

### Provisional algorithmic success threshold

Treat the paired validation as provisional reduced-suite success only if [`compare_runs()`](experiments/cf_cycle_1/compare_nullspace_ablation.py:164) returns continue true: delta BWT at least plus 1.5 points, delta average accuracy no worse than minus 2.0 points, minimum old-task gap no worse than minus 3.0 points, both arm peaks under threshold, projection active, and frozen basis non-empty.

### Weak positive or inconclusive outcome

If comparison is valid but continue is false, the result is weak positive, weak negative, or inconclusive depending on deltas. Do not promote and do not expand. Send the evidence to result critic for bounded interpretation.

### Failure outcome

If either arm fails completion, finite checks, VRAM guardrail, standard gate, active/capacity gate, same-revision check, or comparison validity, stop and report the failed gate. Do not compare invalid arms and do not broaden scope.

## 9. Main risks and confounders

- Fresh-control drift may reduce or erase the copied-control positive deltas; this is the main uncertainty the paired retest is designed to resolve.
- Single-seed reduced [`reasoning3`](experiments/cf_cycle_1/run_nullspace_ablation.py:32) remains a proxy, not broad continual-learning proof.
- GPU nondeterminism or environment drift can produce small metric movements near the weak positive range.
- Same-revision evidence can be under-specified if uncommitted changes exist. The next worker should record revision and avoid changing code between arms.
- Existing active/capacity gate validity does not make the mechanism threshold booleans hard failures by itself; the next worker must inspect and report threshold booleans explicitly.
- The 7500 MB guardrail leaves limited headroom on an RTX 4060 Ti 8GB class environment. Any memory spike should stop validation rather than trigger scope changes.
- A positive reduced-suite result can justify critic review, not catastrophic-forgetting solution language.

## 10. Failure and stop criteria

- Stop if the output root contains stale control or treatment artifacts and no explicit archival or fresh-root decision has been made.
- Stop after a failed CPU data preflight if it is run and reports invalid.
- Stop after control failure; do not launch treatment until control is completed, finite, standard-gate valid, and under the VRAM guardrail.
- Stop after treatment failure; do not run comparison until treatment standard and active/capacity gates are accepted.
- Stop if control and treatment artifacts disagree on task suite, seed, epochs, rank, max length, model identity, or code revision.
- Stop if treatment config does not serialize cllora-active-freeze-topk-2, threshold 0.05, and top k per freeze 2.
- Stop if active reassignment evidence is absent or frozen projection is inactive.
- Stop if comparison is invalid or either arm exceeds 7500 MB peak VRAM.
- Stop after comparison and hand off to result critic; do not launch a second seed, full [`current4`](experiments/cf_cycle_1/run_nullspace_ablation.py:26), v10, Digital Twin, generative subset, broad fuzzer, cadence/timing redesign, or rank change in the same cycle.

## 11. Handoff to Hypothesis Generator

Recommended next mode: llm-hypothesis-generator.

Exact task for llm-hypothesis-generator:

1. Generate narrow validation hypotheses for the fixed paired same-seed protocol, not new algorithm treatments.
2. Rank the main possible outcomes: true top-k utility survives fresh control, copied-control benefit vanishes, mechanism remains strong but utility remains below threshold, active/capacity mechanism weakens, or guardrail/artifact failure blocks comparison.
3. For each outcome, state expected artifact-level observations in [`control_gate.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/control_gate.json), [`treatment_gate.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/treatment_gate.json), [`treatment_active_gate.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/treatment_active_gate.json), and [`comparison.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/comparison.json).
4. Preserve the fixed validation scope: seed 43, rank 4, max length 256, two epochs, 7500 MB guardrail, fresh control, fresh cllora-active-freeze-topk-2 treatment, same code revision, standard gates, active/capacity gate, final comparison only after valid fresh arms.
5. Recommend handoff to llm-experiment-designer with the bounded command sequence if the hypotheses do not reveal a blocking flaw.

Recommended seed: 43.

Stop condition for the next mode: stop after the narrow validation-hypothesis packet; do not run GPU jobs, do not modify code, do not redesign cadence or timing, and do not claim catastrophic forgetting is solved.

## 12. Report back to coordinator

- Current objective: bounded paired same-seed reduced [`reasoning3`](experiments/cf_cycle_1/run_nullspace_ablation.py:32) validation planning for cllora-active-freeze-topk-2.
- Evidence state: confirmed gates and mechanism strength, likely retest-worthy utility, plausible top-k selectivity, weak copied-control utility, contradicted promotion and solution claims, blocked larger escalation.
- Chosen next mode: llm-hypothesis-generator.
- Exact validation task: fresh control plus fresh top-k treatment under same code revision; seed 43, rank 4, max length 256, two epochs, 7500 MB; standard gates for both arms; active/capacity treatment gate; final comparison only after both arms are valid, finite, completed, fresh, and under guardrail.
- Stop condition: no GPU jobs or algorithm-code changes in planner or hypothesis mode; validation execution waits for experiment design.
- Single recommended next-cycle seed: 43.

