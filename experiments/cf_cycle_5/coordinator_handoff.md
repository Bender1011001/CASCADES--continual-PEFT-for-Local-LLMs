# CF-cycle-5 Coordinator Handoff — treatment-strength redesign

Date: 2026-05-17

## 1. Current objective

Work toward solving catastrophic forgetting by starting CF-cycle-5 as a bounded treatment-redesign cycle, not a larger GPU escalation cycle.

The smallest useful objective is: produce a ranked, falsifiable plan for a stronger reduced-memory reasoning3 treatment after the active frozen null-space projection was proven feasible but below threshold across seeds 42 and 43.

## 2. Current evidence state

- Confirmed: CF-cycle-4 closed with valid seed-43 reduced-memory control and treatment artifacts in [`experiments/cf_cycle_4/REPORT.md`](../cf_cycle_4/REPORT.md:1).
- Confirmed: seed 43 had active projection evidence in [`experiments/cf_cycle_4/nullspace_ablation_seed43/treatment_gate.json`](../cf_cycle_4/nullspace_ablation_seed43/treatment_gate.json:1), including non-empty frozen basis, 3600 calls with frozen basis, and positive removed-norm sum.
- Confirmed: seed 43 comparison was valid and under the 7500 MB VRAM threshold in [`experiments/cf_cycle_4/nullspace_ablation_seed43/comparison.json`](../cf_cycle_4/nullspace_ablation_seed43/comparison.json:1).
- Contradicted: algorithmic success for the isolated frozen-nullspace treatment; seed 43 delta BWT was -0.008929480489536235 points, and continue was false in [`experiments/cf_cycle_4/nullspace_ablation_seed43/comparison.json`](../cf_cycle_4/nullspace_ablation_seed43/comparison.json:1).
- Likely: the current treatment is weak, mistimed, under-capacity, or confounded, because seed 42 in [`experiments/cf_cycle_3/REPORT.md`](../cf_cycle_3/REPORT.md:1) was small positive but below threshold while seed 43 was flat or slightly negative.
- Blocked: full current4, v10, and Digital Twin escalation remain blocked until a revised reduced-suite hypothesis and gate exist.
- Unsupported: catastrophic forgetting is not solved.

Confidence level: medium overall; high for feasibility and active projection, medium for weak-treatment diagnosis, unknown for revised variants.

## 3. Next mode to use

Use llm-research-planner.

Reason: the next bottleneck is not implementation or GPU launch. The loop needs a structured redesign packet that turns the broad goal of solving catastrophic forgetting into a concrete hypothesis-generation task with constraints, candidate mechanisms, and a cheap experiment boundary.

## 4. Specific task for that mode

Create a CF-cycle-5 research planning packet for treatment-strength redesign.

The planner should:

1. Read the durable project state in [`CONTEXT.md`](../../CONTEXT.md:1), the latest closeout in [`experiments/cf_cycle_4/REPORT.md`](../cf_cycle_4/REPORT.md:1), the prior seed-42 closeout in [`experiments/cf_cycle_3/REPORT.md`](../cf_cycle_3/REPORT.md:1), and the external-review context pack guide in [`experiments/cf_cycle_5/imo_context_pack/README.md`](imo_context_pack/README.md:1).
2. Treat seed 42 and seed 43 as enough evidence to block larger-run escalation, not enough evidence to abandon the project.
3. Produce a concise planning packet for llm-hypothesis-generator that ranks the redesign axes to investigate next: CL-LoRA reassignment, earlier or stricter freeze timing, frozen-basis capacity or threshold, soft EAR gamma, D-MoLE migration confound, and a staged settings-fuzzer only if it remains small and GPU-budgeted.
4. Preserve hard constraints: RTX 4060 Ti 8GB, reduced reasoning3 first, no full current4 or Digital Twin launch yet, finite-loss and VRAM guardrails, valid run_status artifacts, active-treatment instrumentation, and no catastrophic-forgetting solution claim.
5. Identify whether the existing external context pack should be used as an input to the hypothesis step or updated after the planner packet.

## 5. Stop condition

Stop when the planner has produced a handoff packet with:

- a narrowed CF-cycle-5 objective;
- known evidence and confidence labels;
- ranked uncertainty-reduction questions;
- explicit constraints and non-goals;
- required output for llm-hypothesis-generator;
- one recommended cheap next path toward an executable experiment.

Do not run GPU jobs. Do not modify treatment code. Do not claim catastrophic forgetting is solved.

## 6. What to report back

Report:

- the narrowed cycle objective;
- the top redesign axes to hand to hypothesis generation;
- the evidence quality label for each axis;
- any immediate artifact gaps;
- the exact llm-hypothesis-generator handoff.

## Active cycle record

- cycle_objective: CF-cycle-5 treatment-strength redesign for catastrophic-forgetting mitigation.
- cycle_number_or_label: CF-cycle-5.
- current_evidence_summary: active frozen-nullspace projection is feasible but below threshold across seed 42 and seed 43; larger-run escalation blocked.
- active_handoff_target: llm-research-planner.
- stop_or_continue_condition: continue only through planning and hypothesis redesign until a bounded revised treatment protocol exists.
- files_touched_by_coordinator: [`experiments/cf_cycle_5/coordinator_handoff.md`](coordinator_handoff.md:1) and [`CONTEXT.md`](../../CONTEXT.md:1).
- commands_run_by_coordinator: none; evidence was read from existing reports.
