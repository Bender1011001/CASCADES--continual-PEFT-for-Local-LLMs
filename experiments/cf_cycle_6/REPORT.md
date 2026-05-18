# CF-cycle-6 Report — lower frozen-basis admission-threshold closeout

Date: 2026-05-17

## 1. Cycle objective

Close CF-cycle-6 durably after the lower frozen-basis admission-threshold treatment-only experiment, preserving the confirmed gate evidence, copied-control comparison, and result-critic conclusion without overstating the result.

Objective label: valid execution; threshold-only promotion contradicted; weak diagnostic signal; catastrophic forgetting is not solved.

## 2. What was done

- Followed the bounded treatment-only path recorded in [`experiments/cf_cycle_6/cllora_active_freeze_thresh_02_result_critic_packet.md`](cllora_active_freeze_thresh_02_result_critic_packet.md:12): reduced reasoning3, seed 43, rank 4, max length 256, two epochs, 7500 MB VRAM gate, and treatment variant cllora-active-freeze-thresh-02.
- Preserved CL-LoRA-active behavior while lowering the frozen-basis admission threshold to 0.02, as summarized in [`experiments/cf_cycle_6/cllora_active_freeze_thresh_02_result_critic_packet.md`](cllora_active_freeze_thresh_02_result_critic_packet.md:51).
- Copied the valid CF-cycle-4 seed-43 control into [`experiments/cf_cycle_6/cllora_active_freeze_thresh_02_seed43/control`](cllora_active_freeze_thresh_02_seed43/control) after the first conditional copy skipped because the root directory already existed.
- Validated the treatment with the standard gate in [`experiments/cf_cycle_6/cllora_active_freeze_thresh_02_seed43/treatment_gate.json`](cllora_active_freeze_thresh_02_seed43/treatment_gate.json:1).
- Validated the active/threshold treatment conditions with [`experiments/cf_cycle_6/cllora_active_freeze_thresh_02_seed43/treatment_active_gate.json`](cllora_active_freeze_thresh_02_seed43/treatment_active_gate.json:1).
- Compared the treatment against the copied control in [`experiments/cf_cycle_6/cllora_active_freeze_thresh_02_seed43/comparison.json`](cllora_active_freeze_thresh_02_seed43/comparison.json:1).
- Preserved the result critic verdict supplied for this closeout: valid execution, contradicted promotion, weak diagnostic signal; revise toward bounded capacity/cap-pressure and update-cadence redesign while retaining CL-LoRA-active gates.
- Wrote this durable closeout report and updated [`CONTEXT.md`](../../CONTEXT.md:1). No GPU jobs were run and no algorithm code was modified in this report-writer step.

## 3. Files changed or inspected

### Changed during report-writer closeout

- [`experiments/cf_cycle_6/REPORT.md`](REPORT.md:1) — durable CF-cycle-6 closeout report.
- [`CONTEXT.md`](../../CONTEXT.md:1) — project memory updated with the closeout decision and CF-cycle-7 seed.

### Evidence files inspected or summarized

- [`experiments/cf_cycle_6/cllora_active_freeze_thresh_02_result_critic_packet.md`](cllora_active_freeze_thresh_02_result_critic_packet.md:1) — evidence packet, command record, observed facts, skipped checks, and recommended next action.
- [`experiments/cf_cycle_6/cllora_active_freeze_thresh_02_seed43/comparison.json`](cllora_active_freeze_thresh_02_seed43/comparison.json:1) — final copied-control comparison after unconditional control copy.
- [`experiments/cf_cycle_6/cllora_active_freeze_thresh_02_seed43/treatment_active_gate.json`](cllora_active_freeze_thresh_02_seed43/treatment_active_gate.json:1) — active reassignment and threshold-specific treatment gate.
- [`experiments/cf_cycle_6/cllora_active_freeze_thresh_02_seed43/treatment_gate.json`](cllora_active_freeze_thresh_02_seed43/treatment_gate.json:1) — standard seed-43 treatment gate.
- [`CONTEXT.md`](../../CONTEXT.md:1) — project memory before closeout update.

## 4. Commands/checks run

Report-writer note: this closeout did not run GPU jobs, tests, or code commands. It records prior CF-cycle-6 command outcomes from [`experiments/cf_cycle_6/cllora_active_freeze_thresh_02_result_critic_packet.md`](cllora_active_freeze_thresh_02_result_critic_packet.md:25).

Recorded CF-cycle-6 execution commands and checks:

1. Prepared the output root and attempted conditional control copy. Exact command and outcome are recorded in [`experiments/cf_cycle_6/cllora_active_freeze_thresh_02_result_critic_packet.md`](cllora_active_freeze_thresh_02_result_critic_packet.md:27). Exit code was 0, but the conditional chain skipped the copy because the root already existed.
2. Ran the treatment-only reduced reasoning3 GPU experiment. Exact command is recorded in [`experiments/cf_cycle_6/cllora_active_freeze_thresh_02_result_critic_packet.md`](cllora_active_freeze_thresh_02_result_critic_packet.md:31). The run completed and wrote treatment artifacts under [`experiments/cf_cycle_6/cllora_active_freeze_thresh_02_seed43/treatment`](cllora_active_freeze_thresh_02_seed43/treatment).
3. Ran the standard treatment gate. Exact command and output path are recorded in [`experiments/cf_cycle_6/cllora_active_freeze_thresh_02_result_critic_packet.md`](cllora_active_freeze_thresh_02_result_critic_packet.md:34). Output is [`experiments/cf_cycle_6/cllora_active_freeze_thresh_02_seed43/treatment_gate.json`](cllora_active_freeze_thresh_02_seed43/treatment_gate.json:1).
4. Ran the active/threshold treatment gate. Exact command and output path are recorded in [`experiments/cf_cycle_6/cllora_active_freeze_thresh_02_result_critic_packet.md`](cllora_active_freeze_thresh_02_result_critic_packet.md:37). Output is [`experiments/cf_cycle_6/cllora_active_freeze_thresh_02_seed43/treatment_active_gate.json`](cllora_active_freeze_thresh_02_seed43/treatment_active_gate.json:1).
5. First comparison attempt failed because required copied-control artifacts were missing. Exact command and failure note are recorded in [`experiments/cf_cycle_6/cllora_active_freeze_thresh_02_result_critic_packet.md`](cllora_active_freeze_thresh_02_result_critic_packet.md:40).
6. Repeated the control copy unconditionally and reran comparison. Exact command and outcome are recorded in [`experiments/cf_cycle_6/cllora_active_freeze_thresh_02_result_critic_packet.md`](cllora_active_freeze_thresh_02_result_critic_packet.md:44). Final comparison output is [`experiments/cf_cycle_6/cllora_active_freeze_thresh_02_seed43/comparison.json`](cllora_active_freeze_thresh_02_seed43/comparison.json:1).

Skipped checks in this closeout: no paired fallback, no additional seeds, no full current4, no v10 run, no Digital Twin run, no generative subset, no broad settings fuzzer, and no algorithm-code modification. These remain out of scope or blocked by the contradicted/weak threshold-only result.

## 5. Evidence summary

### Current evidence state

- Confirmed: the treatment variant cllora-active-freeze-thresh-02 completed under reduced reasoning3, seed 43, rank 4, max length 256, and two epochs. [`experiments/cf_cycle_6/cllora_active_freeze_thresh_02_seed43/treatment_gate.json`](cllora_active_freeze_thresh_02_seed43/treatment_gate.json:1) reports completed status and peak VRAM 6843.90966796875 MB under the 7500 MB gate.
- Confirmed: the standard treatment gate is valid. [`experiments/cf_cycle_6/cllora_active_freeze_thresh_02_seed43/treatment_gate.json`](cllora_active_freeze_thresh_02_seed43/treatment_gate.json:1) reports valid true, required artifacts present, finite metrics/instrumentation, finite 3x3 accuracy matrix, non-empty frozen basis, 3600 frozen-basis calls, removed norm sum 1.8263498862379492, projection active true, and freeze event count 38.
- Confirmed: the active/threshold gate is valid. [`experiments/cf_cycle_6/cllora_active_freeze_thresh_02_seed43/treatment_active_gate.json`](cllora_active_freeze_thresh_02_seed43/treatment_active_gate.json:1) reports valid true, treatment variant cllora-active-freeze-thresh-02, frozen-basis threshold 0.02, all active config checks true, and all active instrumentation checks true.
- Confirmed: frozen projection was active. [`experiments/cf_cycle_6/cllora_active_freeze_thresh_02_seed43/treatment_active_gate.json`](cllora_active_freeze_thresh_02_seed43/treatment_active_gate.json:32) reports 3600 frozen-basis calls, removed norm sum 1.8263498862379492, and removed norm per frozen call 0.0005073194128438748.
- Confirmed: active reassignment was active. [`experiments/cf_cycle_6/cllora_active_freeze_thresh_02_seed43/treatment_active_gate.json`](cllora_active_freeze_thresh_02_seed43/treatment_active_gate.json:32) reports 6900 active path calls and active adjustment norm sum 48.84042342959583.
- Confirmed: the comparison is valid after the final unconditional control copy. [`experiments/cf_cycle_6/cllora_active_freeze_thresh_02_seed43/comparison.json`](cllora_active_freeze_thresh_02_seed43/comparison.json:1) reports valid true, no failures, projection active true, frozen basis non-empty true, and continue false.
- Contradicted: threshold-only promotion is not supported. [`experiments/cf_cycle_6/cllora_active_freeze_thresh_02_seed43/comparison.json`](cllora_active_freeze_thresh_02_seed43/comparison.json:4) reports delta BWT -0.018122692840660815 and continue false, despite delta average accuracy +0.03830429581168637.
- Weak signal: old-task gaps are mixed. [`experiments/cf_cycle_6/cllora_active_freeze_thresh_02_seed43/comparison.json`](cllora_active_freeze_thresh_02_seed43/comparison.json:6) reports old-task gaps 0.559717469671106 and -0.5959628553524277.
- Weak signal: lower threshold increased maximum frozen columns from the CF-cycle-5 value 5 to 6, but normalized frozen removal decreased from 0.000609848923952288 to 0.0005073194128438748 and missed the material-strength threshold 0.00076231115494036, as summarized in [`experiments/cf_cycle_6/cllora_active_freeze_thresh_02_result_critic_packet.md`](cllora_active_freeze_thresh_02_result_critic_packet.md:104).
- Plausible: the next useful uncertainty is no longer whether active gates can run; it is whether frozen-basis capacity, cap pressure, or update cadence is limiting useful retention signal.
- Blocked: catastrophic forgetting is not solved. Full current4, v10, Digital Twin, generative subset, broad fuzzer, immediate paired fallback, and promotion/escalation remain blocked until a stronger reduced-suite gate exists.

### Result critic verdict preserved

Result label: valid execution; contradicted promotion; weak diagnostic signal.

The critic accepted the execution and gate evidence, but rejected threshold-only promotion because the decision metric did not improve, continue remained false, and the frozen-basis strength comparison weakened on normalized removal. The critic’s decision is revise, not proceed, not escalate, and not immediate paired fallback.

## 6. Decision: revise

Revise the treatment design before spending larger-run GPU time.

Do not promote the threshold-only 0.02 variant. Do not escalate to full current4, v10, Digital Twin, generative subset, or broad settings fuzzer from this result. Do not immediately pair-fallback unless the coordinator explicitly prioritizes copied-control comparability cleanup over the higher-value redesign path.

The lower threshold was a useful bounded diagnostic: it proved the gate can validate the new threshold and that the run stayed within the RTX 4060 Ti 8GB guardrail. It did not prove a useful anti-forgetting mechanism. Catastrophic forgetting remains unsolved.

## 7. Confidence: medium

- High confidence for valid execution and gate status: run status, standard gate, active/threshold gate, and final comparison exist and report valid or completed states.
- High confidence that frozen projection and active reassignment were exercised: frozen-basis calls were 3600 and active path calls were 6900.
- High confidence that threshold-only promotion is contradicted for this bounded seed-43 reduced-suite decision: delta BWT is negative and continue is false.
- Medium confidence in the algorithmic interpretation: this is one reduced-suite seed with a copied-control comparison, enough to block escalation and guide redesign, not enough to characterize all possible thresholds or full-suite behavior.
- Low confidence for broad generalization to full current4, v10, or Digital Twin settings because those checks were intentionally not run.

Overall confidence: medium.

## 8. Next-cycle seed objective

Recommended CF-cycle-7 seed objective: design a bounded reduced reasoning3 treatment redesign that retains the validated CL-LoRA-active gates but targets frozen-basis capacity, cap pressure, and update cadence instead of lowering only the admission threshold.

Expected redesign focus:

- Keep the active/threshold treatment gate pattern from [`experiments/cf_cycle_6/cllora_active_freeze_thresh_02_seed43/treatment_active_gate.json`](cllora_active_freeze_thresh_02_seed43/treatment_active_gate.json:1).
- Treat threshold-only 0.02 as below decision threshold and do not promote it.
- Ask whether useful retention is capped by frozen-basis capacity, update cadence, or pressure from the active adjustment path.
- Preserve freeze timing relative to D-MoLE migration as the explicit alternate if capacity/cap-pressure and cadence analysis does not yield a bounded next experiment.

Why this seed: CF-cycle-6 confirmed the lower threshold is mechanically active but weaker on normalized removal and negative on BWT delta. The highest-value next action is a redesigned reduced-suite plan that changes capacity/cadence pressure while keeping the validated active gates, not another threshold-only escalation.

## 9. Handoff to LLM Loop Coordinator

handoff_target: llm-loop-coordinator

continue_pivot_or_stop: continue the loop, but pivot from threshold-only promotion to bounded capacity/cap-pressure and update-cadence redesign.

Current objective: close CF-cycle-6 and seed CF-cycle-7 without launching more GPU work.

Current evidence state:

- Confirmed: treatment completed; standard gate valid; active/threshold gate valid; frozen projection active; active reassignment active; comparison valid after final control copy.
- Likely: threshold-only 0.02 is not the right next promotion path under the current reduced-suite evidence.
- Plausible: frozen-basis capacity, cap pressure, or update cadence is a higher-value next redesign target than admission-threshold lowering alone.
- Weak signal: maximum frozen columns increased from 5 to 6, but normalized removal decreased and old-task gaps stayed mixed.
- Contradicted: promotion/escalation from this 0.02 threshold-only result.
- Blocked: catastrophic-forgetting solution claim, full current4, v10, Digital Twin, generative subset, broad fuzzer, and immediate paired fallback.

Next mode to use: llm-research-planner.

Specific task for that mode: produce a CF-cycle-7 planning packet for a bounded reduced reasoning3 redesign that retains CL-LoRA-active standard and active gates, targets frozen-basis capacity/cap-pressure and update cadence, and keeps freeze timing relative to D-MoLE migration as the alternate. The planner should include objective, constraints, candidate knobs, acceptance gates, blocked escalations, and the recommended handoff to hypothesis generation or experiment design.

Stop condition: stop after producing the CF-cycle-7 planning packet; do not run GPU jobs and do not modify algorithm code in the coordinator/planner handoff step.

What to report back: current objective, evidence state by confidence label, chosen next mode, exact task for that mode, stop condition, and the single recommended next-cycle seed.

