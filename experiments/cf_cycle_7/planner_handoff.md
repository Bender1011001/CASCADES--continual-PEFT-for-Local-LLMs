# CF-cycle-7 Planner Handoff — reduced reasoning3 capacity/cadence redesign

Date: 2026-05-17

handoff_target: llm-hypothesis-generator

## 1. Cycle objective

Produce one bounded reduced [`reasoning3`](experiments/cf_cycle_1/run_nullspace_ablation.py:32) redesign plan that retains the validated CL-LoRA-active standard gate and active reassignment gate while targeting frozen-basis capacity, cap pressure, and update cadence. The cycle question is: which cheapest capacity/cadence mechanism is most likely to turn already-active frozen projection plus active reassignment into stronger old-task retention without breaching the 7500 MB VRAM gate?

Success definition for CF-cycle-7 planning: hand off enough ranked candidate mechanisms, acceptance gates, and blocked non-goals for the next mode to choose one concrete reduced-suite treatment without rediscovering CF-cycle-6. This planner step does not run GPU jobs and does not modify algorithm code.

## 2. Current assumptions

- Assumption A1: reduced [`reasoning3`](experiments/cf_cycle_1/run_nullspace_ablation.py:32), seed 43, rank 4, max length 256, and two epochs remain the default bounded scope because CF-cycle-5 and CF-cycle-6 both produced valid treatment artifacts under this envelope.
- Assumption A2: a copied seed-43 control is acceptable only for a cheap provisional treatment-only check when rank, suite, seed, max length, epochs, model, and control semantics remain unchanged; if the next candidate changes rank or control-relevant training semantics, a paired control/treatment run is required before comparison confidence rises above medium.
- Assumption A3: threshold-only lowering is not the next promotion path. A future variant may include threshold validation as a gate field, but it must not be a threshold-only treatment.
- Assumption A4: the next useful uncertainty is mechanism selection, not execution. Several candidate knobs are plausible enough that the next mode should rank hypotheses before an experiment protocol is written.
- Assumption A5: catastrophic forgetting remains unsolved; any reduced-suite pass can justify only another reduced validation step or critic review, not a broad solution claim.

## 3. Evidence summary by confidence label

### Confirmed

- CF-cycle-6 closeout is durable in [`REPORT.md`](experiments/cf_cycle_6/REPORT.md:1), and project memory was updated in [`CONTEXT.md`](CONTEXT.md:1).
- Treatment variant cllora-active-freeze-thresh-02 completed under reduced [`reasoning3`](experiments/cf_cycle_1/run_nullspace_ablation.py:32), seed 43, rank 4, max length 256, two epochs, and peak VRAM 6843.90966796875 MB under the 7500 MB gate, as reported in [`REPORT.md`](experiments/cf_cycle_6/REPORT.md:56) and [`cllora_active_freeze_thresh_02_result_critic_packet.md`](experiments/cf_cycle_6/cllora_active_freeze_thresh_02_result_critic_packet.md:55).
- The standard treatment gate is valid in [`treatment_gate.json`](experiments/cf_cycle_6/cllora_active_freeze_thresh_02_seed43/treatment_gate.json:1), and the active/threshold gate is valid in [`treatment_active_gate.json`](experiments/cf_cycle_6/cllora_active_freeze_thresh_02_seed43/treatment_active_gate.json:1).
- Frozen projection was active: 3600 frozen-basis calls, removed norm sum 1.8263498862379492, and removed norm per frozen call 0.0005073194128438748 in [`cllora_active_freeze_thresh_02_result_critic_packet.md`](experiments/cf_cycle_6/cllora_active_freeze_thresh_02_result_critic_packet.md:74).
- Active reassignment was active: 6900 active path calls and active adjustment norm sum 48.84042342959583 in [`cllora_active_freeze_thresh_02_result_critic_packet.md`](experiments/cf_cycle_6/cllora_active_freeze_thresh_02_result_critic_packet.md:81).
- Final copied-control comparison is valid after unconditional control copy. [`comparison.json`](experiments/cf_cycle_6/cllora_active_freeze_thresh_02_seed43/comparison.json:1) reports delta BWT -0.018122692840660815, delta average accuracy +0.03830429581168637, old-task gaps [0.559717469671106, -0.5959628553524277], and continue false.

### Likely

- Threshold-only 0.02 is not the right next promotion path under current reduced-suite evidence. The result critic labels the hypothesis weakened/contradicted in [`cllora_active_freeze_thresh_02_result_critic_packet.md`](experiments/cf_cycle_6/cllora_active_freeze_thresh_02_result_critic_packet.md:129).
- A useful next experiment needs either a stronger frozen-basis capacity/cap-pressure mechanism or an update-cadence change while retaining active CL-LoRA behavior. Confidence: medium, because CF-cycle-6 proved activation but not decision-metric improvement.

### Plausible

- Frozen-basis admission may be admitting weak or diffuse directions rather than strong retention directions. CF-cycle-6 increased maximum frozen columns from 5 to 6 but reduced normalized removal relative to CF-cycle-5 in [`cllora_active_freeze_thresh_02_result_critic_packet.md`](experiments/cf_cycle_6/cllora_active_freeze_thresh_02_result_critic_packet.md:104).
- Update cadence may be limiting useful signal because active null-space refresh currently occurs every 25 batches in [`train.py`](train.py:385), while micro-sleep occurs every 100 batches in [`train.py`](train.py:402), D-MoLE migration occurs at task boundary in [`train.py`](train.py:413), and freeze occurs afterward in [`train.py`](train.py:441).
- Active adjustment may dominate strict frozen removal. CF-cycle-6 active adjustment norm sum was 48.84042342959583 while strict frozen removed norm sum was 1.8263498862379492 in [`cllora_active_freeze_thresh_02_result_critic_packet.md`](experiments/cf_cycle_6/cllora_active_freeze_thresh_02_result_critic_packet.md:74), suggesting cap pressure or active/frozen balance is worth testing.

### Weak signal

- Maximum frozen columns rose from the CF-cycle-5 value 5 to 6, but normalized removal fell from 0.000609848923952288 to 0.0005073194128438748 and missed the material-strength threshold 0.00076231115494036 in [`cllora_active_freeze_thresh_02_result_critic_packet.md`](experiments/cf_cycle_6/cllora_active_freeze_thresh_02_result_critic_packet.md:104).
- Old-task gaps remain mixed in [`comparison.json`](experiments/cf_cycle_6/cllora_active_freeze_thresh_02_seed43/comparison.json:1), so a single average-accuracy gain is not enough to infer retention improvement.

### Contradicted

- Promotion or escalation from the threshold-only 0.02 result is contradicted. The decision is revise, not proceed, in [`REPORT.md`](experiments/cf_cycle_6/REPORT.md:74).

### Blocked

- Catastrophic-forgetting solution claim, full [`current4`](experiments/cf_cycle_1/run_nullspace_ablation.py:26), v10, Digital Twin, generative subset, broad fuzzer, and immediate paired fallback remain blocked unless explicitly re-coordinated, as preserved in [`REPORT.md`](experiments/cf_cycle_6/REPORT.md:50) and [`CONTEXT.md`](CONTEXT.md:13).

## 4. Files and areas likely involved

- Treatment wrapper and variant serialization: [`config_for_arm()`](experiments/cf_cycle_1/run_nullspace_ablation.py:79), run artifact writing in [`run_arm()`](experiments/cf_cycle_1/run_nullspace_ablation.py:309), and CLI treatment choices in [`parse_args()`](experiments/cf_cycle_1/run_nullspace_ablation.py:425).
- Active/frozen instrumentation: [`install_instrumentation()`](experiments/cf_cycle_1/run_nullspace_ablation.py:178), especially frozen removal and active adjustment counters in [`wrapped_reassign()`](experiments/cf_cycle_1/run_nullspace_ablation.py:216) plus freeze-event capture in [`wrapped_freeze()`](experiments/cf_cycle_1/run_nullspace_ablation.py:248).
- Configurable treatment knobs: [`AblationConfig`](cascades/config.py:15), especially [`enable_coso_nullspace`](cascades/config.py:30), [`enable_cllora_reassign`](cascades/config.py:32), [`ear_gamma`](cascades/config.py:44), [`enable_soft_ear`](cascades/config.py:46), and [`frozen_basis_variance_threshold`](cascades/config.py:55).
- Frozen protection and active reassignment: [`_cllora_reassign()`](cascades/adapters.py:176), [`streaming_ear_update()`](cascades/adapters.py:358), and [`freeze_current_subspace()`](cascades/adapters.py:366).
- Update cadence and boundary ordering: [`train_cascades()`](train.py:196), null-space refresh at [`train.py`](train.py:385), micro-sleep at [`train.py`](train.py:402), D-MoLE migration at [`train.py`](train.py:413), freeze at [`train.py`](train.py:441), and sleep/eval after freeze at [`train.py`](train.py:453).
- Gate and comparison behavior: [`validate_seed43_arm_gate.py`](experiments/cf_cycle_4/validate_seed43_arm_gate.py:1), [`validate_active_treatment_gate.py`](experiments/cf_cycle_5/validate_active_treatment_gate.py:1), and [`compare_runs()`](experiments/cf_cycle_1/compare_nullspace_ablation.py:105).

## 5. Candidate redesign knobs

### Rank 1 — Capacity/cap-pressure without threshold-only lowering

Mechanism family: preserve CL-LoRA-active flags, but change how many frozen directions are retained or how strongly weak directions are filtered. Candidate shapes for hypothesis ranking:

1. Top-k-per-freeze or minimum-strong-direction policy: keep a bounded top-k set from [`freeze_current_subspace()`](cascades/adapters.py:366) rather than lowering the variance threshold alone. Prediction: maximum frozen columns may stay similar or rise modestly, removed norm per frozen call should increase above CF-cycle-6 and ideally above CF-cycle-5, and old-task gaps should become less mixed.
2. Capacity-probe rank increase: test whether rank 4 is under-capacity by moving to a small rank 6 or rank 8 reduced [`reasoning3`](experiments/cf_cycle_1/run_nullspace_ablation.py:32) candidate. This requires paired control/treatment at the same rank because the existing seed-43 control is rank 4.
3. Active/frozen pressure balance: adjust the active EAR pressure only if the hypothesis generator concludes the active adjustment norm is suppressing useful strict frozen protection. This should be framed as cap-pressure balance, not generic gamma tuning.

Planner preference: rank 1 should probably be a top-k or cap-pressure mechanism that can reuse rank 4 and seed 43 before considering higher-rank paired runs, because rank changes consume more GPU and break copied-control comparability.

### Rank 2 — Update cadence

Mechanism family: preserve CL-LoRA-active flags and threshold default, but change when active null-space summaries and frozen snapshots are refreshed.

1. More frequent active null-space refresh: expose the 25-batch refresh interval at [`train.py`](train.py:385) as a treatment knob, then test a smaller interval such as 10 or 12 batches. Prediction: active path remains 6900 calls, active adjustment norm distribution changes, and strict frozen removal may become stronger if sketches are less stale.
2. Epoch-boundary frozen snapshot: add a treatment knob that freezes after each epoch or once after the first epoch within a task, rather than only after the task boundary at [`train.py`](train.py:441). Prediction: more calls with frozen basis earlier in later epochs, higher removed norm per frozen call, but risk of lower same-task plasticity.
3. Cadence alignment with sleep: align refresh or freeze before micro-sleep at [`train.py`](train.py:402) or before task-level sleep at [`train.py`](train.py:453), then inspect whether sleep consolidates already-protected directions.

Planner preference: cadence hypotheses need artifact expectations before implementation because earlier freeze could improve retention or crush learning. Hypothesis generation should decide whether null-space refresh cadence or freeze cadence is the cheaper first probe.

### Rank 3 — Diagnostic instrumentation before GPU, if needed

Mechanism family: CPU-only or log-only additions that clarify capacity/cadence pressure before spending GPU. Candidate additions include per-freeze retained eigenvalue ratios, per-freeze retained U/V columns, and per-call removed norm quantiles. This can be justified if the hypothesis generator cannot choose between top-k capacity and cadence with current aggregate artifacts.

Planner preference: use instrumentation only if it remains CPU-light and does not become a broad fuzzer. Do not block the loop on perfect observability if one bounded treatment is clearly rankable.

### Explicit alternate — Freeze timing relative to D-MoLE migration

If capacity/cap-pressure and update-cadence hypotheses are not compelling, the alternate is to change freeze timing relative to D-MoLE migration: freeze before D-MoLE migration at [`train.py`](train.py:413) or delay D-MoLE migration until after freeze at [`train.py`](train.py:441). Prediction: protected adapter directions are snapshotted before critical/non-critical layer migration changes the adapter population. This alternate is higher risk because it changes boundary semantics and may affect adapter allocation, so it should not be first unless ranked mechanisms above look weak.

## 6. Candidate checks or commands

Planner step: no commands are required beyond reading the evidence files already listed.

If the next mode hands off to experiment design/code, require these checks before any GPU run:

- Use `python -m py_compile` on any touched planning/protocol files plus the touched code files, especially [`run_nullspace_ablation.py`](experiments/cf_cycle_1/run_nullspace_ablation.py:1), [`config.py`](cascades/config.py:1), [`adapters.py`](cascades/adapters.py:1), [`train.py`](train.py:1), and the relevant validators.
- Use `pytest` on [`test_data.py`](tests/test_data.py:1) and [`test_cf_cycle_2_guardrails.py`](tests/test_cf_cycle_2_guardrails.py:1) before GPU work.
- Re-run the reduced-suite data preflight for seed 43 at max length 256 using [`reasoning3_prepare_data_preflight.py`](experiments/cf_cycle_3/reasoning3_prepare_data_preflight.py:1).
- Validate treatment config serialization before GPU: the treatment artifact must distinguish its capacity/cadence knob from cllora-active and cllora-active-freeze-thresh-02 in [`config_for_arm()`](experiments/cf_cycle_1/run_nullspace_ablation.py:79).
- Run the standard gate with [`validate_seed43_arm_gate.py`](experiments/cf_cycle_4/validate_seed43_arm_gate.py:1), the active/capacity-cadence gate extended from [`validate_active_treatment_gate.py`](experiments/cf_cycle_5/validate_active_treatment_gate.py:1), then final comparison with [`compare_nullspace_ablation.py`](experiments/cf_cycle_1/compare_nullspace_ablation.py:1).

## 7. Acceptance gates for a reduced-suite experiment

### Standard CL-LoRA-active gates

- Run status completed, finite metrics, finite instrumentation, and finite 3x3 accuracy matrix under reduced [`reasoning3`](experiments/cf_cycle_1/run_nullspace_ablation.py:32).
- Peak VRAM at or below 7500 MB for every accepted arm.
- Non-empty frozen basis, calls with frozen basis greater than zero, removed norm sum greater than zero, and projection active true.
- Active reassignment evidence: active reassignment path count greater than zero, active adjustment norm sum finite and positive, and active CL-LoRA config flags serialized true.
- Treatment config must serialize the exact capacity/cadence knob and preserve the validated CL-LoRA-active baseline unless the chosen hypothesis explicitly changes one field.

### Comparison gates

- Comparison must be valid, with required control and treatment artifacts present. If using copied control, copy unconditionally before comparison to avoid the CF-cycle-6 conditional-copy pitfall recorded in [`cllora_active_freeze_thresh_02_result_critic_packet.md`](experiments/cf_cycle_6/cllora_active_freeze_thresh_02_result_critic_packet.md:40).
- Decision success requires [`compare_runs()`](experiments/cf_cycle_1/compare_nullspace_ablation.py:105) continue true: delta BWT at least +1.5 points, delta average accuracy no worse than -2.0 points, minimum old-task gap no worse than -3.0 points, VRAM under threshold, projection active, and frozen basis non-empty.
- Mechanism success should also prefer removed norm per frozen call above CF-cycle-6 0.0005073194128438748, ideally above CF-cycle-5 0.000609848923952288, and ideally near or above the material-strength threshold 0.00076231115494036.
- A positive delta BWT below +1.5 points is weak positive or inconclusive, not success.
- A threshold-only 0.02 repetition is failure by design because it repeats the contradicted CF-cycle-6 path.

### Overclaiming guard

- Even a valid reduced-suite success does not solve catastrophic forgetting. It can justify result-critic review and possibly one additional reduced seed or paired control/treatment replication, not full [`current4`](experiments/cf_cycle_1/run_nullspace_ablation.py:26), v10, Digital Twin, or solution language.

## 8. Main risks and confounders

- Reduced [`reasoning3`](experiments/cf_cycle_1/run_nullspace_ablation.py:32) remains a proxy; single-seed evidence can guide the loop but not prove broad behavior.
- Copied-control comparisons are cheaper but lower confidence if treatment code changes control-relevant semantics.
- Rank or model-capacity changes improve capacity testing but force paired reruns and may increase VRAM.
- Earlier freeze can protect old directions but may reduce same-task plasticity and average accuracy.
- D-MoLE migration order is a real confound because migration at [`train.py`](train.py:413) precedes freeze at [`train.py`](train.py:441), but changing it may alter layer selection and adapter topology.
- Aggregate removed norm may hide distribution shifts; if candidate mechanisms are tied, add quantile/eigenvalue instrumentation before GPU rather than launching a broad grid.

## 9. Blocked escalations and non-goals

- Do not run GPU jobs in this planner step.
- Do not modify algorithm code in this planner step.
- Do not repeat a threshold-only promotion path.
- Do not claim catastrophic forgetting is solved.
- Do not launch full [`current4`](experiments/cf_cycle_1/run_nullspace_ablation.py:26), v10, Digital Twin, generative subset, broad settings fuzzer, or immediate paired fallback unless the coordinator explicitly reopens scope.
- Do not weaken finite-loss, VRAM, active projection, active reassignment, or comparison validity gates.

## 10. Handoff to Hypothesis Generator

Recommended next mode: llm-hypothesis-generator.

Reason: multiple candidate mechanisms remain plausible. Frozen-basis top-k/cap-pressure, rank-capacity probing, null-space refresh cadence, epoch-boundary freeze cadence, and D-MoLE-relative freeze timing need ranking before the experiment designer can specify one bounded treatment without overfitting to CF-cycle-6.

Exact task for llm-hypothesis-generator:

1. Generate ranked CF-cycle-7 hypotheses for why CL-LoRA-active plus threshold 0.02 was mechanically active but missed the decision metric.
2. Focus the top ranking on frozen-basis capacity/cap-pressure and update cadence. Explicitly decide whether top-k/cap-pressure, rank-capacity, null-space refresh cadence, or epoch-boundary freeze cadence should be the first bounded treatment.
3. Preserve freeze timing relative to D-MoLE migration as the explicit alternate if the capacity/cadence path is not compelling.
4. For the top three mechanisms, state predicted artifact-level observations, failure criteria, required code/protocol surfaces, and whether copied seed-43 control is valid or paired control is required.
5. Choose exactly one recommended next experiment candidate and one single seed.

Recommended next-cycle seed: 43.

Stop condition for the next mode: stop after producing ranked hypotheses and one recommended bounded reduced-suite experiment candidate; do not run GPU jobs, do not modify code, and do not claim catastrophic forgetting is solved.

## 11. Report back to coordinator

- Current objective: CF-cycle-7 planning packet for a bounded reduced [`reasoning3`](experiments/cf_cycle_1/run_nullspace_ablation.py:32) CL-LoRA-active redesign focused on frozen-basis capacity/cap pressure and update cadence, with D-MoLE-relative freeze timing as alternate.
- Evidence state: confirmed gates/execution/comparison; likely threshold-only 0.02 not promotable; plausible capacity/cap-pressure or cadence limit; weak signal from higher frozen column count but weaker normalized removal; contradicted promotion/escalation; blocked larger-suite and solution claims.
- Chosen next mode: llm-hypothesis-generator.
- Exact task for that mode: rank capacity/cap-pressure and update-cadence mechanisms, preserve D-MoLE freeze-timing alternate, and choose exactly one reduced-suite experiment candidate with required artifacts and gates.
- Stop condition: stop after the hypothesis packet; no GPU jobs, no code modification, no catastrophic-forgetting claim.
- Single recommended next-cycle seed: 43.
