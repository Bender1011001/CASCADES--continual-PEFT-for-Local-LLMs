# CF-cycle-7 Report — top-k frozen-basis admission closeout

Date: 2026-05-17

## 1. Cycle objective

Close CF-cycle-7 durably after the bounded top-k frozen-basis admission experiment, preserving confirmed gate evidence, the mechanism-strength improvement, the result critic conclusion, and the next-cycle seed without overstating the result.

Objective label: valid execution; confirmed mechanism improvement; likely useful or weak positive utility; not success; catastrophic forgetting is not solved.

## 2. What was done

- Preserved the prior bounded treatment-only execution recorded in [`experiments/cf_cycle_7/cllora_active_freeze_topk2_result_critic_packet.md`](experiments/cf_cycle_7/cllora_active_freeze_topk2_result_critic_packet.md:9): reduced [`reasoning3`](experiments/cf_cycle_1/run_nullspace_ablation.py:32), seed 43, rank 4, max length 256, two epochs, 7500 MB VRAM gate, and treatment variant cllora-active-freeze-topk-2.
- Preserved treatment configuration evidence from [`experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/treatment_active_gate.json`](experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/treatment_active_gate.json:24): frozen-basis admission threshold 0.05 and top k per freeze 2, while retaining the CL-LoRA-active config flags.
- Preserved standard treatment-gate evidence from [`experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/treatment_gate.json`](experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/treatment_gate.json:1).
- Preserved active/capacity-gate evidence from [`experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/treatment_active_gate.json`](experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/treatment_active_gate.json:1).
- Preserved copied-control comparison evidence from [`experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/comparison.json`](experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/comparison.json:1).
- Preserved the supplied result critic verdict: valid execution, confirmed mechanism improvement, likely useful or weak positive utility, not success, no catastrophic-forgetting claim, no full-suite escalation, and bounded paired same-seed validation next.
- Wrote this durable closeout report and updated [`CONTEXT.md`](CONTEXT.md:1). No GPU jobs were run and no algorithm code was modified in this report-writer step.

## 3. Files changed or inspected

### Changed during report-writer closeout

- [`experiments/cf_cycle_7/REPORT.md`](experiments/cf_cycle_7/REPORT.md:1) — durable CF-cycle-7 closeout report.
- [`CONTEXT.md`](CONTEXT.md:1) — project memory updated with the closeout decision and coordinator seed.

### Evidence files inspected or summarized

- [`experiments/cf_cycle_7/cllora_active_freeze_topk2_result_critic_packet.md`](experiments/cf_cycle_7/cllora_active_freeze_topk2_result_critic_packet.md:1) — evidence packet, observed facts, command record, result label, confidence, and handoff material.
- [`experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/comparison.json`](experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/comparison.json:1) — copied-control comparison result.
- [`experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/treatment_active_gate.json`](experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/treatment_active_gate.json:1) — active reassignment plus top-k capacity gate.
- [`experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/treatment_gate.json`](experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/treatment_gate.json:1) — standard reduced [`reasoning3`](experiments/cf_cycle_1/run_nullspace_ablation.py:32) treatment gate.
- [`experiments/cf_cycle_7/planner_handoff.md`](experiments/cf_cycle_7/planner_handoff.md:1) — original capacity/cadence planning context that motivated the top-k treatment.
- [`CONTEXT.md`](CONTEXT.md:1) — project memory before closeout update.

## 4. Commands/checks run

Report-writer note: this closeout did not run GPU jobs, tests, or shell commands. It records prior CF-cycle-7 command outcomes from [`experiments/cf_cycle_7/cllora_active_freeze_topk2_result_critic_packet.md`](experiments/cf_cycle_7/cllora_active_freeze_topk2_result_critic_packet.md:26).

Recorded CF-cycle-7 execution commands and checks:

1. Prepared the output root and copied the seed-43 control. Exact command and exit status are recorded in [`experiments/cf_cycle_7/cllora_active_freeze_topk2_result_critic_packet.md`](experiments/cf_cycle_7/cllora_active_freeze_topk2_result_critic_packet.md:28).
2. Ran the treatment-only reduced [`reasoning3`](experiments/cf_cycle_1/run_nullspace_ablation.py:32) GPU experiment. Exact command and exit status are recorded in [`experiments/cf_cycle_7/cllora_active_freeze_topk2_result_critic_packet.md`](experiments/cf_cycle_7/cllora_active_freeze_topk2_result_critic_packet.md:34).
3. Ran the standard treatment gate. Exact command and output path are recorded in [`experiments/cf_cycle_7/cllora_active_freeze_topk2_result_critic_packet.md`](experiments/cf_cycle_7/cllora_active_freeze_topk2_result_critic_packet.md:40).
4. Ran the active/capacity gate for the top-k treatment. Exact command and output path are recorded in [`experiments/cf_cycle_7/cllora_active_freeze_topk2_result_critic_packet.md`](experiments/cf_cycle_7/cllora_active_freeze_topk2_result_critic_packet.md:46).
5. Re-copied the seed-43 control and ran comparison. Exact command and outcome are recorded in [`experiments/cf_cycle_7/cllora_active_freeze_topk2_result_critic_packet.md`](experiments/cf_cycle_7/cllora_active_freeze_topk2_result_critic_packet.md:52).
6. Ran a fresh verification assertion over required artifacts and mechanism thresholds. Exact verification command location and exit status are recorded in [`experiments/cf_cycle_7/cllora_active_freeze_topk2_result_critic_packet.md`](experiments/cf_cycle_7/cllora_active_freeze_topk2_result_critic_packet.md:58).

Skipped checks in this closeout: no paired same-seed rerun, no additional seeds, no full [`current4`](experiments/cf_cycle_1/run_nullspace_ablation.py:26), no v10 run, no Digital Twin run, no generative subset, no broad settings fuzzer, no cadence redesign, no D-MoLE timing redesign, no rank change, and no algorithm-code modification.

## 5. Evidence summary

### Current evidence state

- Confirmed: the top-k treatment completed under reduced [`reasoning3`](experiments/cf_cycle_1/run_nullspace_ablation.py:32), seed 43, rank 4, max length 256, two epochs, and the 7500 MB VRAM guardrail. [`experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/treatment_gate.json`](experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/treatment_gate.json:6) reports completed status, peak VRAM 6843.90966796875 MB, and wall time 1066.974350452423 seconds.
- Confirmed: the standard treatment gate is valid. [`experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/treatment_gate.json`](experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/treatment_gate.json:1) reports valid true, required artifacts present, finite metrics/instrumentation, finite 3x3 accuracy matrix, under-threshold VRAM, non-empty frozen basis, 3600 frozen-basis calls, removed norm sum 3.530579238988139, projection active true, and 38 freeze events.
- Confirmed: the active/capacity gate is valid. [`experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/treatment_active_gate.json`](experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/treatment_active_gate.json:1) reports valid true, treatment variant cllora-active-freeze-topk-2, threshold 0.05, top k per freeze 2, active config checks true, and active instrumentation checks true.
- Confirmed: mechanism strength improved materially. [`experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/treatment_active_gate.json`](experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/treatment_active_gate.json:34) reports removed norm sum 3.530579238988139, removed norm per frozen call 0.000980716455274483, max frozen columns 2, and flags above the CF-cycle-6 baseline, CF-cycle-5 baseline, and material-strength threshold.
- Confirmed: active reassignment remained active. [`experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/treatment_active_gate.json`](experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/treatment_active_gate.json:42) reports 6900 active reassignment path calls and active adjustment norm sum 50.596070232306374.
- Confirmed: the copied-control comparison is valid. [`experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/comparison.json`](experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/comparison.json:1) reports valid true, no failures, projection active true, frozen basis non-empty true, and continue false.
- Likely: the top-k treatment is useful enough to justify a bounded retest, because [`experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/comparison.json`](experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/comparison.json:4) reports delta BWT plus 0.2983417459082205 points and delta average accuracy plus 0.2692983654684511 points.
- Plausible: top-k admission is reducing weak-direction admission and improving the strict frozen-projection mechanism relative to the CF-cycle-6 threshold-only path. This is plausible because normalized frozen removal cleared all reference thresholds in [`experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/treatment_active_gate.json`](experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/treatment_active_gate.json:37).
- Weak signal: utility is positive but below success. [`experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/comparison.json`](experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/comparison.json:4) reports positive BWT and average-accuracy deltas, but [`experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/comparison.json`](experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/comparison.json:15) reports continue false.
- Weak signal: old-task gaps remain mixed. [`experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/comparison.json`](experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/comparison.json:6) reports gaps of 0.7995196940998484 and -0.20283620228340737.
- Contradicted: success, promotion, and larger-run escalation are not supported by this evidence. The result critic label in [`experiments/cf_cycle_7/cllora_active_freeze_topk2_result_critic_packet.md`](experiments/cf_cycle_7/cllora_active_freeze_topk2_result_critic_packet.md:85) is likely useful, not success.
- Contradicted or unsupported: catastrophic forgetting is not solved. [`experiments/cf_cycle_7/cllora_active_freeze_topk2_result_critic_packet.md`](experiments/cf_cycle_7/cllora_active_freeze_topk2_result_critic_packet.md:81) explicitly limits the interpretation to mechanism activation plus bounded positive utility.
- Blocked: full [`current4`](experiments/cf_cycle_1/run_nullspace_ablation.py:26), v10, Digital Twin, generative subset, broad fuzzer, cadence/timing redesign, rank changes, and catastrophic-forgetting solution language remain blocked until the paired same-seed reduced validation is completed and reviewed.

### Result critic verdict preserved

Result label: likely useful or weak positive; not success.

The critic accepted execution and gate validity, accepted the mechanism-strength improvement, treated the positive copied-control utility deltas as useful but weak, rejected catastrophic-forgetting or success language, rejected full-suite escalation, and recommended bounded paired same-seed validation next.

## 6. Decision: retest

Retest in a bounded paired same-seed reduced validation before spending larger-run GPU time or redesigning cadence/timing.

The top-k treatment is the strongest mechanism result so far in this local CF loop because its normalized removed norm exceeds the CF-cycle-6 baseline, CF-cycle-5 baseline, and material threshold while preserving active reassignment. The copied-control comparison is positive on BWT and average accuracy, but the gate still reports continue false and old-task gaps remain mixed. That is enough to justify a paired same-seed reduced retest; it is not enough to promote, escalate, or claim catastrophic forgetting is solved.

## 7. Confidence: medium

- High confidence for valid execution and gate status: run status, standard gate, active/capacity gate, comparison, and verification artifacts exist and report valid or completed states.
- High confidence that the mechanism improved materially under this treatment: removed norm per frozen call is 0.000980716455274483, above all stated reference thresholds, and active reassignment remained active.
- Medium confidence in useful utility: BWT and average-accuracy deltas are positive, but this is one reduced-suite seed with a copied control, continue false, and mixed old-task gaps.
- High confidence that catastrophic forgetting is not solved by this evidence alone.
- Low confidence for broad generalization to full [`current4`](experiments/cf_cycle_1/run_nullspace_ablation.py:26), v10, or Digital Twin settings because those checks were intentionally not run.

Overall confidence: medium.

## 8. Next-cycle seed objective

Recommended next-cycle seed objective: run a paired same-seed reduced [`reasoning3`](experiments/cf_cycle_1/run_nullspace_ablation.py:32) validation of cllora-active-freeze-topk-2 before any cadence/timing redesign or larger escalation.

Specific bounded validation shape:

- Use seed 43 first, with rank 4, max length 256, two epochs, and a 7500 MB VRAM threshold.
- Run a fresh control and fresh treatment under the same code revision and same reduced [`reasoning3`](experiments/cf_cycle_1/run_nullspace_ablation.py:32) envelope, rather than relying only on the copied CF-cycle-4 control.
- Validate both accepted arms with the standard seed-43 gate pattern in [`experiments/cf_cycle_4/validate_seed43_arm_gate.py`](experiments/cf_cycle_4/validate_seed43_arm_gate.py:1).
- Validate the top-k treatment with the active/capacity gate pattern in [`experiments/cf_cycle_5/validate_active_treatment_gate.py`](experiments/cf_cycle_5/validate_active_treatment_gate.py:1), including threshold 0.05, top k per freeze 2, active reassignment evidence, max frozen columns, and normalized removed norm thresholds.
- Compare only after both arms are fresh, valid, finite, and under the VRAM guardrail using [`compare_nullspace_ablation.py`](experiments/cf_cycle_1/compare_nullspace_ablation.py:1).

Why this seed: the top-k result is the first CF-cycle-5 through CF-cycle-7 treatment that combines materially stronger frozen-basis mechanism evidence with positive copied-control BWT and average-accuracy deltas. The cheapest uncertainty reducer is paired same-seed validation under the same reduced scope, not a larger suite, rank change, cadence change, D-MoLE timing change, or broad fuzzer.

## 9. Handoff to LLM Loop Coordinator

handoff_target: llm-loop-coordinator

continue_pivot_or_stop: continue the loop with a bounded retest, not promotion and not larger escalation.

Current objective: close CF-cycle-7 and seed the next coordinator cycle without launching more GPU work.

Current evidence state:

- Confirmed: treatment completed; standard gate valid; active/capacity gate valid; frozen projection active; active reassignment active; mechanism strength above CF-cycle-6 baseline, CF-cycle-5 baseline, and material threshold; comparison valid.
- Likely: top-k frozen-basis admission is useful enough to justify a paired same-seed retest.
- Plausible: top-k admission improved useful frozen-basis selectivity by admitting at most two strongest directions per freeze.
- Weak signal: copied-control utility is positive but below success, with continue false and mixed old-task gaps.
- Contradicted: success, promotion, full-suite escalation, and catastrophic-forgetting solution claims.
- Blocked: full [`current4`](experiments/cf_cycle_1/run_nullspace_ablation.py:26), v10, Digital Twin, generative subset, broad fuzzer, cadence/timing redesign, rank change, and solution language before paired same-seed validation.

Next mode to use: llm-loop-coordinator.

Specific task for that mode: start the next cycle by routing a bounded paired same-seed reduced [`reasoning3`](experiments/cf_cycle_1/run_nullspace_ablation.py:32) validation of cllora-active-freeze-topk-2. The validation should use seed 43, rank 4, max length 256, two epochs, and 7500 MB VRAM guardrail; run fresh control and treatment; require standard gates, active/capacity gate, and final comparison; and avoid cadence/timing redesign or larger escalation until this paired validation is reviewed.

Stop condition: stop after creating the coordinator handoff for the bounded paired validation; do not run GPU jobs, do not modify algorithm code, and do not claim catastrophic forgetting is solved in the coordinator handoff step.

What to report back: current objective, evidence state by confidence label, chosen next mode, exact paired-validation task, stop condition, and the single recommended next-cycle seed.

