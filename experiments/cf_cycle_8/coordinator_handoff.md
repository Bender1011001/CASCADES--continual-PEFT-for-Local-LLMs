# CF-cycle-8 Coordinator Handoff — paired same-seed top-k validation

Date: 2026-05-17

## 1. Current objective

Start CF-cycle-8 from the CF-cycle-7 closeout evidence in [`experiments/cf_cycle_7/REPORT.md`](experiments/cf_cycle_7/REPORT.md:1), without launching GPU work in the coordinator step.

Working objective for the next cycle: route a bounded paired same-seed reduced [`reasoning3`](experiments/cf_cycle_1/run_nullspace_ablation.py:32) validation of cllora-active-freeze-topk-2 before any cadence or timing redesign, larger-suite escalation, rank change, or catastrophic-forgetting solution language.

## 2. Current evidence state

- Confirmed: treatment completed; standard gate valid; active/capacity gate valid; frozen projection active; active reassignment active; mechanism strength above the CF-cycle-6 baseline, CF-cycle-5 baseline, and material threshold; comparison valid.
- Likely: top-k frozen-basis admission is useful enough to justify a paired same-seed retest.
- Plausible: top-k admission improved useful frozen-basis selectivity by admitting at most two strongest directions per freeze.
- Weak signal: copied-control utility is positive but below success, with continue false and mixed old-task gaps.
- Contradicted: success, promotion, full-suite escalation, and catastrophic-forgetting solution claims.
- Blocked: full [`current4`](experiments/cf_cycle_1/run_nullspace_ablation.py:26), v10, Digital Twin, generative subset, broad fuzzer, cadence/timing redesign, rank change, and solution language before paired same-seed validation.

## 3. Next mode to use

Next mode: llm-research-planner.

Reason: this is a new loop cycle seeded by a report-writer closeout. The objective is already narrow, so the planner should fast-track a bounded validation plan rather than reopen broad hypothesis discovery.

## 4. Specific task for the next mode

Create a concise CF-cycle-8 planning packet for a paired same-seed reduced [`reasoning3`](experiments/cf_cycle_1/run_nullspace_ablation.py:32) validation of cllora-active-freeze-topk-2.

Required validation shape:

- Use seed 43.
- Use rank 4.
- Use max length 256.
- Use two epochs.
- Use a 7500 MB VRAM guardrail.
- Run a fresh control arm and a fresh treatment arm under the same code revision and same reduced [`reasoning3`](experiments/cf_cycle_1/run_nullspace_ablation.py:32) envelope.
- Require standard gates for both accepted arms using the seed-43 gate pattern in [`experiments/cf_cycle_4/validate_seed43_arm_gate.py`](experiments/cf_cycle_4/validate_seed43_arm_gate.py:1).
- Require the active/capacity treatment gate using the active gate pattern in [`experiments/cf_cycle_5/validate_active_treatment_gate.py`](experiments/cf_cycle_5/validate_active_treatment_gate.py:1), including frozen-basis threshold 0.05, top k per freeze 2, active reassignment evidence, max frozen columns, and normalized removed-norm thresholds.
- Run the final comparison only after both fresh arms are completed, finite, valid, and under the VRAM guardrail, using [`experiments/cf_cycle_1/compare_nullspace_ablation.py`](experiments/cf_cycle_1/compare_nullspace_ablation.py:1).

Non-goals for the next mode:

- Do not run GPU jobs.
- Do not modify algorithm code.
- Do not redesign cadence or timing.
- Do not expand to full [`current4`](experiments/cf_cycle_1/run_nullspace_ablation.py:26), v10, Digital Twin, generative subset, or broad fuzzer.
- Do not claim catastrophic forgetting is solved.

## 5. Stop condition

Stop after producing the CF-cycle-8 planning packet and handoff to the next appropriate execution-design mode. The next mode should report the exact validation protocol, required commands, required artifacts, gate criteria, failure handling, and stop condition. It should not launch the paired validation itself.

## 6. What to report back

Report back:

- Current objective.
- Evidence state by confidence label.
- Chosen next mode.
- Exact paired-validation task.
- Stop condition.
- Single recommended next-cycle seed.

## 7. Single recommended next-cycle seed

Seed 43.

## 8. Coordinator decision

Decision: continue the loop with bounded paired validation planning, not promotion and not larger escalation.

The expected value is positive and downside is contained because the next cycle spends only planning effort first, then a reduced paired validation if approved by later modes. The strongest uncertainty reducer is a fresh same-seed control plus fresh top-k treatment comparison under the same reduced envelope used by CF-cycle-7.

