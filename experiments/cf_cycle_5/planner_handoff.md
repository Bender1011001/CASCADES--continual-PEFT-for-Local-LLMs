# CF-cycle-5 Planner Handoff — treatment-strength redesign

Date: 2026-05-17

handoff_target: llm-hypothesis-generator

## 1. Current objective

Start CF-cycle-5 as a bounded treatment-redesign cycle for catastrophic-forgetting mitigation. The cycle question is: which cheapest reduced-memory [`reasoning3`](experiments/cf_cycle_1/run_nullspace_ablation.py:32) treatment variant can turn confirmed active frozen null-space projection into a stronger BWT improvement without violating RTX 4060 Ti 8GB guardrails?

This is not a full-suite escalation cycle. Full [`current4`](experiments/cf_cycle_1/run_nullspace_ablation.py:26), v10, and Digital Twin runs remain blocked until a revised reduced-suite hypothesis and gate exist.

## 2. Current evidence state

### Confirmed

- CF-cycle-4 seed 43 completed valid sequential control and treatment artifacts under the 7500 MB threshold; [`experiments/cf_cycle_4/REPORT.md`](experiments/cf_cycle_4/REPORT.md:13) records seed, suite, epoch, rank, max-length, and guardrail conditions.
- Seed 43 active treatment projection was real: [`experiments/cf_cycle_4/REPORT.md`](experiments/cf_cycle_4/REPORT.md:20) records non-empty frozen basis, 3600 calls with frozen basis, and removed norm sum 3.100088352795865, sourced from [`experiments/cf_cycle_4/nullspace_ablation_seed43/treatment_gate.json`](experiments/cf_cycle_4/nullspace_ablation_seed43/treatment_gate.json:1).
- Seed 42 completed valid reduced-memory control and treatment after the data-loader fix; [`experiments/cf_cycle_3/REPORT.md`](experiments/cf_cycle_3/REPORT.md:21) reports valid comparison, active projection, non-empty frozen basis, under-threshold peaks, and delta BWT plus 0.36167033073388477 points.
- The current ablation wrapper deliberately used frozen null-space treatment without active CL-LoRA reassignment: [`enable_coso_nullspace`](experiments/cf_cycle_1/run_nullspace_ablation.py:84) is true only for treatment, while [`enable_cllora_reassign`](experiments/cf_cycle_1/run_nullspace_ablation.py:85) is false for both arms.
- In the adapter implementation, strict frozen-basis projection happens before the active CL-LoRA gate, while the active sketch path is gated by [`enable_cllora_reassign`](cascades/adapters.py:195). Therefore the past-task frozen projection was tested, but the active-sketch reassignment/soft-EAR path was not isolated as an enabled treatment.

### Likely

- The current isolated frozen-nullspace treatment is weak, mistimed, under-capacity, or confounded. Confidence: medium. Rationale: seed 42 was weak positive and below threshold in [`experiments/cf_cycle_3/REPORT.md`](experiments/cf_cycle_3/REPORT.md:68), while seed 43 was flat or slightly negative in [`experiments/cf_cycle_4/REPORT.md`](experiments/cf_cycle_4/REPORT.md:63).
- A stronger treatment should first change one or two reduced-suite knobs with direct artifact-level observability, not launch a broad grid or larger suite. Confidence: high for engineering value, medium for algorithmic payoff.

### Plausible

- CL-LoRA reassignment is the highest-value first redesign axis because the current treatment left the active reassignment gate off, despite the production default enabling [`enable_cllora_reassign`](cascades/config.py:32).
- Freeze timing may be mistimed because D-MoLE migration runs at the phase transition before the task-boundary freeze in [`train.py`](train.py:413), while the freeze itself starts at [`train.py`](train.py:441).
- Frozen-basis capacity or filtering may be too conservative because freeze keeps directions above a 5 percent structural-variance threshold in [`freeze_current_subspace()`](cascades/adapters.py:399) and stores only the extracted basis afterward.
- D-MoLE migration may confound which adapters are protected because promotion/demotion occurs in [`train.py`](train.py:413) before the frozen subspace snapshot in [`train.py`](train.py:441).
- Soft EAR gamma can matter only once active CL-LoRA reassignment is enabled, because [`soft_ear()`](cascades/math_ops.py:49) is reached from [`_cllora_reassign()`](cascades/adapters.py:207) only on the active-sketch path.

### Weak signal

- Seed 43 removed-norm sum was positive but modest relative to 3600 calls. This proves activity, not strength. No normalized removed-norm-per-gradient baseline exists yet.
- Existing evidence does not distinguish whether old-task retention is limited by projection strength, adapter selection, freeze timing, rank/capacity, or evaluation variance.

### Contradicted

- Isolated frozen-nullspace algorithmic success is contradicted for seed 43: [`experiments/cf_cycle_4/nullspace_ablation_seed43/comparison.json`](experiments/cf_cycle_4/nullspace_ablation_seed43/comparison.json:1) reports delta BWT minus 0.008929480489536235 points and continue false, summarized in [`experiments/cf_cycle_4/REPORT.md`](experiments/cf_cycle_4/REPORT.md:63).
- Lowering the success threshold after seeing the result is not supported; [`experiments/cf_cycle_3/REPORT.md`](experiments/cf_cycle_3/REPORT.md:69) preserves the plus 1.5 BWT-point threshold.

### Blocked

- Full [`current4`](experiments/cf_cycle_1/run_nullspace_ablation.py:26), v10, Digital Twin, and generative-subset evaluation are blocked until a revised reduced-suite treatment reaches a valid gate or a hypothesis generator explicitly justifies a smaller uncertainty-reduction experiment.
- Catastrophic forgetting is not solved.

## 3. Ranked uncertainty-reduction questions

| Rank | Redesign axis | Evidence quality | Falsifiable question for hypothesis generation | Why this rank |
|---:|---|---|---|---|
| 1 | Enable active CL-LoRA reassignment on top of strict frozen projection | Confirmed off in wrapper; plausible as under-strength cause | If [`enable_cllora_reassign`](cascades/config.py:32) is enabled while keeping strict frozen projection, does treatment increase BWT by at least plus 1.5 points versus same-seed control, with finite loss, peak VRAM at or below 7500 MB, and active instrumentation showing more useful projection mass? | Cheapest high-signal axis because current evidence proves frozen projection was active while active reassignment was not. |
| 2 | Frozen-basis capacity, variance threshold, or update cadence | Plausible; weak direct evidence | If the frozen basis keeps more useful directions than the current 5 percent structural-variance cutoff in [`freeze_current_subspace()`](cascades/adapters.py:399), does removed-norm-per-call and old-task retention improve without crushing new-task accuracy? | Directly targets under-capacity and is measurable through freeze-event counts, max frozen columns, and removed norm. |
| 3 | Earlier or stricter freeze timing | Plausible | If freezing occurs before D-MoLE migration/sleep/evaluation, or once mid-task after stable gradients, does old-task retention improve compared with post-task freeze at [`train.py`](train.py:441)? | Timing is a natural failure mode after two weak seeds, but it probably needs a small code knob. |
| 4 | D-MoLE migration confound | Plausible but higher-risk | If migration is disabled only at task boundaries, delayed until after freeze, or instrumented to show protected adapters are not demoted, does the treatment become less noisy? | Migration occurs before freeze in [`train.py`](train.py:413), but disabling D-MoLE globally can change adapter count/VRAM, so this should not be first. |
| 5 | Soft EAR gamma | Plausible only after rank 1 | Once active reassignment is enabled, does changing [`ear_gamma`](cascades/config.py:44) reduce amplification/noise while preserving retention? | Current treatment did not exercise the active soft-EAR path, so gamma is a second-stage local tuning axis. |
| 6 | Staged settings fuzzer | Weak signal; only if bounded | Can a tiny, predeclared set of at most four treatment variants identify a stronger configuration under one-seed GPU budget? | Useful only after the hypothesis generator narrows axes; broad search is explicitly out of budget. |

## 4. Files and areas likely involved

- Treatment wrapper and instrumentation: [`experiments/cf_cycle_1/run_nullspace_ablation.py`](experiments/cf_cycle_1/run_nullspace_ablation.py:79), especially the hard-coded treatment config and active-projection instrumentation.
- Comparison gate: [`experiments/cf_cycle_1/compare_nullspace_ablation.py`](experiments/cf_cycle_1/compare_nullspace_ablation.py:76), especially projection-active, frozen-basis, VRAM, and continue criteria.
- Adapter projection logic: [`_cllora_reassign()`](cascades/adapters.py:176), [`streaming_ear_update()`](cascades/adapters.py:358), and [`freeze_current_subspace()`](cascades/adapters.py:366).
- Training loop timing and migration order: [`train_cascades()`](train.py:196), D-MoLE migration at [`train.py`](train.py:413), freeze at [`train.py`](train.py:441), and sleep/eval after freeze at [`train.py`](train.py:453).
- Configuration defaults and knobs: [`AblationConfig`](cascades/config.py:15), especially [`enable_cllora_reassign`](cascades/config.py:32), [`ear_gamma`](cascades/config.py:44), and [`enable_soft_ear`](cascades/config.py:46).
- Existing external-review bundle: [`experiments/cf_cycle_5/imo_context_pack/README.md`](experiments/cf_cycle_5/imo_context_pack/README.md:1) and [`experiments/cf_cycle_5/imo_context_pack/MASTER_PROMPT.md`](experiments/cf_cycle_5/imo_context_pack/MASTER_PROMPT.md:1).

## 5. Candidate checks or commands for the next executable cycle

Do not run GPU work during hypothesis generation. The hypothesis step should select one bounded experiment and hand it to experiment design/code only after the criteria below are explicit.

Minimum CPU checks before any GPU treatment variant:

- Bytecode compile the changed files: [`train.py`](train.py:1), [`cascades/adapters.py`](cascades/adapters.py:1), [`cascades/config.py`](cascades/config.py:1), [`experiments/cf_cycle_1/run_nullspace_ablation.py`](experiments/cf_cycle_1/run_nullspace_ablation.py:1), and [`experiments/cf_cycle_1/compare_nullspace_ablation.py`](experiments/cf_cycle_1/compare_nullspace_ablation.py:1) if touched.
- Run the data and guardrail tests over [`tests/test_data.py`](tests/test_data.py:1) and [`tests/test_cf_cycle_2_guardrails.py`](tests/test_cf_cycle_2_guardrails.py:1).
- Re-run reduced-suite data preflight using [`experiments/cf_cycle_3/reasoning3_prepare_data_preflight.py`](experiments/cf_cycle_3/reasoning3_prepare_data_preflight.py:1) for the chosen seed and max length 256.
- Confirm the candidate treatment config is serialized in the treatment artifact under [`run_status.json`](experiments/cf_cycle_4/nullspace_ablation_seed43/treatment/run_status.json:1) style and is distinguishable from the CF-cycle-3/4 frozen-only treatment.

Recommended cheap GPU path for the next executable experiment, if hypothesis generation agrees:

1. Add or specify a one-knob treatment variant that enables [`enable_cllora_reassign`](cascades/config.py:32) while preserving the current strict frozen projection, [`enable_soft_ear`](cascades/config.py:46), rank 4, max length 256, two epochs, and 7500 MB threshold.
2. Prefer seed 43 treatment-only against the already valid seed-43 control as a provisional cheap check, but mark confidence medium because code provenance changes may make a rerun cleaner.
3. If seed 43 CL-LoRA treatment is valid and reaches at least plus 1.5 BWT points, rerun a paired sequential control/treatment on one additional seed before any larger-suite escalation.
4. If seed 43 is valid but below threshold, use instrumentation to choose rank 2 or rank 3 from the table rather than jumping to v10/current4.

## 6. Realistic success threshold

Success for the next reduced-suite treatment hypothesis:

- Valid completed treatment and comparable control artifacts with finite metrics, valid [`run_status.json`](experiments/cf_cycle_4/nullspace_ablation_seed43/control/run_status.json:1)-style status, and peak VRAM at or below 7500 MB.
- Active-treatment instrumentation: non-empty frozen basis, calls with frozen basis greater than zero, removed norm sum greater than zero, and preferably a normalized removed-norm-per-call readout added or computed.
- Delta BWT at least plus 1.5 points versus same-seed control, no large average-accuracy regression, and no old-task delta gap worse than the existing comparison gate in [`compare_nullspace_ablation.py`](experiments/cf_cycle_1/compare_nullspace_ablation.py:167).
- If delta BWT is positive but less than plus 1.5 points, the result is inconclusive or weak positive, not success.
- If delta BWT is flat/negative, the tested axis is likely insufficient unless instrumentation reveals an obvious invalid implementation or under-activation.

## 7. Main risks and confounders

- The reduced [`reasoning3`](experiments/cf_cycle_1/run_nullspace_ablation.py:32) suite is a proxy. Passing it would justify a critic review and maybe one more reduced seed, not a catastrophic-forgetting solution claim.
- Reusing an existing control is cheap but lowers provenance confidence if treatment code changes; rerunning paired control/treatment is cleaner but costs more GPU time.
- Enabling CL-LoRA active sketch may improve retention while harming plasticity; new-task accuracy and average accuracy must remain visible.
- D-MoLE migration and sleep are interleaved with freeze timing, so changing one setting can change adapter allocation, VRAM, and retention simultaneously.
- The external context pack notes that num-samples is not a reliable knob until wired through; do not use it as a fuzzer dimension.

## 8. Existing context-pack decision

Use the existing [`experiments/cf_cycle_5/imo_context_pack`](experiments/cf_cycle_5/imo_context_pack) as optional input to hypothesis generation or external review. It is already scoped for this problem and includes the relevant source, reports, and artifacts listed in [`experiments/cf_cycle_5/imo_context_pack/README.md`](experiments/cf_cycle_5/imo_context_pack/README.md:25).

Do not rebuild the pack before the hypothesis step. This planner handoff should be the primary local seed for llm-hypothesis-generator. If an external reviewer is invoked, add or point them to this planner handoff as a small addendum rather than recopying the entire bundle.

## 9. Specific task for llm-hypothesis-generator

Generate ranked, falsifiable CF-cycle-5 treatment hypotheses that explain why confirmed active frozen projection was feasible but below threshold across seed 42 and seed 43. The output should:

1. Rank the six axes above and state whether to keep this planner ranking or revise it.
2. For each top-three axis, provide mechanism, predicted artifact-level observations, failure criteria, and the cheapest valid experiment shape.
3. Choose exactly one recommended next experiment candidate, preferably the active CL-LoRA reassignment variant unless contradicted by the hypothesis analysis.
4. Preserve the hard constraints: RTX 4060 Ti 8GB, reduced [`reasoning3`](experiments/cf_cycle_1/run_nullspace_ablation.py:32) first, no full [`current4`](experiments/cf_cycle_1/run_nullspace_ablation.py:26), no v10 or Digital Twin launch, finite-loss and VRAM guardrails, valid run-status artifacts, active-treatment instrumentation, and no catastrophic-forgetting solution claim.
5. Define the handoff to experiment design/code as a one-cycle, less-than-10-minute-planning artifact with exact files to inspect or change and the gate artifacts required before GPU execution.

## 10. Stop condition for hypothesis step

Stop after producing a ranked hypothesis packet plus one recommended cheap path toward an executable reduced-suite experiment. Do not run GPU jobs. Do not modify treatment code. Do not claim catastrophic forgetting is solved.

## 11. What to report back

- Top redesign axis and reason.
- Confidence label for each ranked axis.
- One exact recommended next experiment candidate and its required artifacts.
- Any immediate artifact gaps, especially whether the wrapper needs a config knob before the experiment designer can write a protocol.
- Explicit non-goals and blocked escalation list.
