# CF-cycle-8 Report — paired top-k validation closeout

Date: 2026-05-17

## 1. Cycle objective

Close CF-cycle-8 durably after the fresh paired same-seed validation for [`cllora-active-freeze-topk-2`](experiments/cf_cycle_1/run_nullspace_ablation.py:79), preserving the result critic verdict, evidence boundaries, and next-cycle seed without promoting the mechanism beyond the evidence.

Objective label: evidence-usable; strong mechanism; reproducible weak positive; not promotion-passing; catastrophic forgetting is not solved.

## 2. What was done

- Preserved the CF-cycle-8 paired-validation verdict from [`cllora_active_freeze_topk2_seed43_paired_result_critic_packet.md`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired_result_critic_packet.md:1): copied-control comparability is resolved for the fixed seed-43 reduced [`reasoning3`](experiments/cf_cycle_1/run_nullspace_ablation.py:32) setup, but the utility signal is too small for promotion.
- Recorded the user-approved validation scope: reduced [`reasoning3`](experiments/cf_cycle_1/run_nullspace_ablation.py:32), seed 43, rank 4, max length 256, two epochs, and a 7500 MB VRAM threshold.
- Preserved freshness and same-envelope evidence: the run used fresh control and treatment artifacts under the same recorded revision, with [`control/run_status.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/control/run_status.json:13) and [`treatment/run_status.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/treatment/run_status.json:13) both recording revision 7c4e01d.
- Preserved the dirty-but-fixed worktree caveat from [`cllora_active_freeze_topk2_seed43_paired_result_critic_packet.md`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired_result_critic_packet.md:28): uncommitted worktree state limits reproducibility, but does not break same-run comparability.
- Preserved valid data and gate evidence from [`reasoning3_prepare_data_preflight.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/reasoning3_prepare_data_preflight.json:1), [`control_gate.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/control_gate.json:1), [`treatment_gate.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/treatment_gate.json:1), [`treatment_active_gate.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/treatment_active_gate.json:1), and [`comparison.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/comparison.json:1).
- Wrote this durable closeout report and updated [`CONTEXT.md`](CONTEXT.md:1). No GPU jobs, tests, shell commands, or algorithm-code changes were performed in this report-writer closeout.

## 3. Files changed or inspected

### Changed during report-writer closeout

- [`experiments/cf_cycle_8/REPORT.md`](experiments/cf_cycle_8/REPORT.md:1) — durable CF-cycle-8 closeout report.
- [`CONTEXT.md`](CONTEXT.md:1) — project memory updated with the closeout decision and coordinator seed.

### Evidence files inspected or summarized

- [`cllora_active_freeze_topk2_seed43_paired_result_critic_packet.md`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired_result_critic_packet.md:1) — result critic packet and corrected conclusion.
- [`reasoning3_prepare_data_preflight.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/reasoning3_prepare_data_preflight.json:1) — reduced-suite data preflight.
- [`control_gate.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/control_gate.json:1) — fresh control standard gate.
- [`treatment_gate.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/treatment_gate.json:1) — fresh treatment standard gate.
- [`treatment_active_gate.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/treatment_active_gate.json:1) — active/capacity mechanism gate.
- [`comparison.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/comparison.json:1) — final paired comparison.
- [`control/metrics.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/control/metrics.json:1) and [`treatment/metrics.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/treatment/metrics.json:1) — paired arm metrics.
- [`control/run_status.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/control/run_status.json:1) and [`treatment/run_status.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/treatment/run_status.json:1) — paired arm execution status and revision metadata.

## 4. Commands/checks run

Report-writer closeout: no new shell commands, tests, GPU jobs, or code-modifying commands were run.

Recorded CF-cycle-8 validation checks and artifacts:

1. Data preflight was valid in [`reasoning3_prepare_data_preflight.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/reasoning3_prepare_data_preflight.json:6), with zero zero-label batches across the three reduced [`reasoning3`](experiments/cf_cycle_1/run_nullspace_ablation.py:32) tasks.
2. Fresh control standard gate was valid in [`control_gate.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/control_gate.json:2), completed under the 7500 MB threshold in [`control_gate.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/control_gate.json:42), and had no parameter mismatches in [`control_gate.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/control_gate.json:29).
3. Fresh treatment standard gate was valid in [`treatment_gate.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/treatment_gate.json:2), completed under the 7500 MB threshold in [`treatment_gate.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/treatment_gate.json:42), and had no parameter mismatches in [`treatment_gate.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/treatment_gate.json:29).
4. Treatment mechanism gate was valid in [`treatment_active_gate.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/treatment_active_gate.json:2), with active config checks passing in [`treatment_active_gate.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/treatment_active_gate.json:6) and mechanism thresholds passing in [`treatment_active_gate.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/treatment_active_gate.json:47).
5. Final paired comparison was valid in [`comparison.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/comparison.json:2), had an empty failure list in [`comparison.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/comparison.json:3), and reported continue false in [`comparison.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/comparison.json:15).
6. Exact command templates for the validation sequence are preserved in [`planner_handoff.md`](experiments/cf_cycle_8/planner_handoff.md:140), covering data preflight, fresh control, control gate, fresh treatment, treatment gate, active/capacity gate, and comparison. This closeout did not rerun them.

Skipped checks in this closeout: no additional seed, no full [`current4`](experiments/cf_cycle_1/run_nullspace_ablation.py:26), no v10 run, no Digital Twin run, no generative subset, no broad settings fuzzer, no cadence redesign, no projection timing redesign, no active-reassignment redesign, and no algorithm-code modification.

## 5. Evidence summary

### Confirmed

- The paired comparison is usable for the fixed seed-43 reduced [`reasoning3`](experiments/cf_cycle_1/run_nullspace_ablation.py:32) setup. [`comparison.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/comparison.json:2) is valid and [`comparison.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/comparison.json:3) has no failures.
- Copied-control comparability from CF-cycle-7 is resolved for this narrow setup. [`control/run_status.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/control/run_status.json:13) and [`treatment/run_status.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/treatment/run_status.json:13) record the same revision 7c4e01d, while [`control_gate.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/control_gate.json:29) and [`treatment_gate.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/treatment_gate.json:29) show no parameter mismatches.
- Data preflight is valid. [`reasoning3_prepare_data_preflight.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/reasoning3_prepare_data_preflight.json:6) reports valid true and the task entries in [`reasoning3_prepare_data_preflight.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/reasoning3_prepare_data_preflight.json:12), [`reasoning3_prepare_data_preflight.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/reasoning3_prepare_data_preflight.json:22), and [`reasoning3_prepare_data_preflight.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/reasoning3_prepare_data_preflight.json:32) show no zero-label batches.
- Both arms completed under the same reduced envelope: [`control_gate.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/control_gate.json:7) through [`control_gate.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/control_gate.json:19) and [`treatment_gate.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/treatment_gate.json:7) through [`treatment_gate.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/treatment_gate.json:19) record reduced [`reasoning3`](experiments/cf_cycle_1/run_nullspace_ablation.py:32), seed 43, rank 4, max length 256, two epochs, 7500 MB threshold, and peak VRAM 6843.90966796875 MB.
- The top-k mechanism is strong and materially active. [`treatment_gate.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/treatment_gate.json:52) reports a non-empty frozen basis, 3600 frozen-basis calls, removed norm sum 3.530579238988139, projection active true, and 38 freeze events. [`treatment_active_gate.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/treatment_active_gate.json:34) reports max frozen columns 2, removed norm per frozen call 0.000980716455274483, and active adjustment norm sum 50.596070232306374.
- The active/capacity gate clears the material-strength references. [`treatment_active_gate.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/treatment_active_gate.json:39), [`treatment_active_gate.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/treatment_active_gate.json:40), and [`treatment_active_gate.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/treatment_active_gate.json:41) report above CF-cycle-6 baseline, above CF-cycle-5 baseline, and above material-strength threshold true.

### Likely

- The treatment has a real but weak positive utility signal in this fixed setup. [`comparison.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/comparison.json:4) reports BWT delta plus 0.2983417459082205 points and [`comparison.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/comparison.json:5) reports average-accuracy delta plus 0.2692983654684511 points.
- The weak positive signal is reproducible across the copied-control and fresh-paired seed-43 reduced validation because the fresh paired comparison did not erase the CF-cycle-7 positive sign.

### Weak signal

- Old-task utility is mixed, not uniformly improved. [`comparison.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/comparison.json:6) reports old-task gaps of plus 0.7995196940998484 and minus 0.20283620228340737.
- Absolute BWT remains negative in both arms. [`control/metrics.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/control/metrics.json:9) reports BWT -0.026559348750060024 and [`treatment/metrics.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/treatment/metrics.json:9) reports BWT -0.02357593129097782.

### Contradicted or unsupported

- Promotion is not supported. [`comparison.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/comparison.json:4) reports only plus 0.2983417459082205 BWT points, far below the plus 1.5 point promotion threshold used in prior cycle decisions, and [`comparison.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/comparison.json:15) reports continue false.
- Catastrophic forgetting is not solved. The treatment lessens forgetting slightly on this reduced proxy but does not eliminate negative BWT, mixed old-task gaps, or the need for redesign.
- Full [`current4`](experiments/cf_cycle_1/run_nullspace_ablation.py:26), v10, Digital Twin, generative subset, broad settings fuzzer, or broad escalation is not justified solely from this result.

### Confounders and gaps

- Single seed only: seed 43.
- Reduced proxy only: reduced [`reasoning3`](experiments/cf_cycle_1/run_nullspace_ablation.py:32), rank 4, max length 256, and two epochs.
- Dirty-but-fixed worktree envelope: same-run comparability is acceptable, but exact reproducibility requires preserving revision 7c4e01d plus the uncommitted-envelope caveat from [`cllora_active_freeze_topk2_seed43_paired_result_critic_packet.md`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired_result_critic_packet.md:28).
- Utility/mechanism mismatch remains unresolved: strong projection and active reassignment evidence did not convert into enough BWT gain.

## 6. Decision: revise

Revise rather than escalate.

Keep [`cllora-active-freeze-topk-2`](experiments/cf_cycle_1/run_nullspace_ablation.py:79) as a live mechanism candidate because the paired evidence is usable, the active/top-k mechanism is strong, and the weak positive utility signal is reproducible. Do not promote it as a solution and do not launch full-suite or broad escalation solely from this result because [`comparison.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/comparison.json:15) reports continue false and the BWT lift is only plus 0.2983417459082205 points.

The next useful work is design revision aimed at converting strong mechanism activity into larger old-task retention. The expected value is positive because the mechanism is real; the downside of bounded planning is contained. The expected value of broad escalation is not positive yet because the measured utility remains sub-threshold.

## 7. Confidence: medium

- High confidence for execution and artifact validity in this exact experiment: preflight, standard gates, active/capacity gate, same-envelope checks, and comparison all passed.
- Medium-high confidence that the CF-cycle-7 copied-control comparability concern is resolved for the fixed seed-43 reduced proxy.
- Medium confidence that the treatment has a real but weak positive utility signal under this setup.
- High confidence that promotion, broad generalization, and catastrophic-forgetting solution claims are unsupported.
- Low confidence that the current top-k setting will generalize to full [`current4`](experiments/cf_cycle_1/run_nullspace_ablation.py:26), v10, Digital Twin, or generative settings without redesign.

Overall confidence: medium.

## 8. Next-cycle seed objective

Recommended next-cycle seed objective: investigate why strong top-k active-freeze mechanism evidence does not translate into enough BWT improvement, then design the smallest bounded reduced-suite revision likely to increase retained-task utility.

Seed constraints:

- Continue with coordination and planning first, not GPU execution.
- Preserve the paired-arm gate discipline used in CF-cycle-8.
- Keep [`cllora-active-freeze-topk-2`](experiments/cf_cycle_1/run_nullspace_ablation.py:79) as a live mechanism component, but revise the intervention rather than scaling the same configuration.
- Consider bounded redesign hypotheses around projection timing or cadence, utility-coupled frozen-basis admission, and active reassignment magnitude or coupling.
- Do not run any GPU job without explicit user approval.
- Do not modify algorithm code in the coordinator handoff step.
- Do not claim catastrophic forgetting is solved.

Why this seed: the key unresolved issue is no longer whether the top-k mechanism activates. It does. The unresolved issue is why that strong mechanism produces only plus 0.2983417459082205 BWT points and mixed old-task gaps in [`comparison.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/comparison.json:4). A design-revision loop is cheaper and more informative than full [`current4`](experiments/cf_cycle_1/run_nullspace_ablation.py:26), v10, Digital Twin, generative subset, or broad fuzzer escalation.

## 9. Handoff to LLM Loop Coordinator

handoff_target: llm-loop-coordinator

continue_pivot_or_stop: continue with a design-revision cycle, not promotion and not broad escalation.

Current objective: close CF-cycle-8 and seed the next coordinator cycle without launching more GPU work.

Current evidence state:

- Confirmed: fixed seed-43 paired comparability is resolved; preflight valid; control gate valid; treatment gate valid; active/capacity gate valid; comparison valid; same recorded revision; no standard-gate parameter mismatches.
- Confirmed: the active/top-k mechanism is strong, with non-empty frozen basis, 3600 frozen-basis calls, max frozen columns 2, removed norm per frozen call 0.000980716455274483, and active adjustment norm sum 50.596070232306374.
- Likely: the top-k treatment has real weak positive utility on this reduced proxy.
- Weak signal: old-task benefit is mixed and absolute BWT remains negative.
- Contradicted or unsupported: promotion, broad utility, catastrophic-forgetting solution claims, and full-suite escalation.
- Blocked: full [`current4`](experiments/cf_cycle_1/run_nullspace_ablation.py:26), v10, Digital Twin, generative subset, broad settings fuzzer, and broad escalation without a stronger reduced-suite utility result.

Next mode to use: llm-loop-coordinator.

Specific task for that mode: start the next cycle by routing to planning for a bounded design-revision packet. The planner should focus on why strong top-k projection plus active reassignment did not produce enough BWT improvement, rank redesign hypotheses around projection timing or cadence, utility coupling, and active reassignment magnitude, and propose the cheapest reduced-suite experiment candidate with explicit gates. No GPU run should be started without explicit approval.

Stop condition for coordinator: produce the next planning handoff only; do not run GPU jobs, do not modify algorithm code, and do not claim catastrophic forgetting is solved.

