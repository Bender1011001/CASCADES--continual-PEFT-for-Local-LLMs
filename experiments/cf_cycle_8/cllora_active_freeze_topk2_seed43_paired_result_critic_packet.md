# CF-cycle-8 paired top-k result critic packet

Handoff target: llm-report-writer

## 1. Claim reviewed

Fresh same-seed paired validation for [`cllora-active-freeze-topk-2`](../cf_cycle_1/run_nullspace_ablation.py:79) resolves the CF-cycle-7 copied-control comparability concern enough to judge whether the top-k treatment has reproducible utility under the reduced [`reasoning3`](../cf_cycle_1/run_nullspace_ablation.py:32) proxy.

## 2. Evidence quality

Evidence quality: moderate to strong for this exact reduced seed-43 paired comparison; weak to moderate for broad catastrophic-forgetting utility.

The key comparison artifact is valid with no failures in [`comparison.json`](cllora_active_freeze_topk2_seed43_paired/comparison.json:1). The data preflight is valid in [`reasoning3_prepare_data_preflight.json`](cllora_active_freeze_topk2_seed43_paired/reasoning3_prepare_data_preflight.json:1). Both arms completed under the same recorded revision and envelope in [`control/run_status.json`](cllora_active_freeze_topk2_seed43_paired/control/run_status.json:1) and [`treatment/run_status.json`](cllora_active_freeze_topk2_seed43_paired/treatment/run_status.json:1). Both standard gates are valid in [`control_gate.json`](cllora_active_freeze_topk2_seed43_paired/control_gate.json:1) and [`treatment_gate.json`](cllora_active_freeze_topk2_seed43_paired/treatment_gate.json:1). The active-capacity gate is valid in [`treatment_active_gate.json`](cllora_active_freeze_topk2_seed43_paired/treatment_active_gate.json:1).

## 3. Main support

- The copied-control concern is resolved for this fixed setup: CF-cycle-8 ran a fresh control and fresh treatment under the same seed, rank, maximum length, epoch count, VRAM threshold, and recorded revision in [`control/run_status.json`](cllora_active_freeze_topk2_seed43_paired/control/run_status.json:1) and [`treatment/run_status.json`](cllora_active_freeze_topk2_seed43_paired/treatment/run_status.json:1).
- The final comparison is usable: [`comparison.json`](cllora_active_freeze_topk2_seed43_paired/comparison.json:1) reports valid true, no failures, BWT delta plus 0.2983417459082205 points, average accuracy delta plus 0.2692983654684511 points, active projection true, nonempty frozen basis true, and continue false.
- The treatment mechanism is not a phantom: [`treatment_gate.json`](cllora_active_freeze_topk2_seed43_paired/treatment_gate.json:52) reports projection active with 3600 frozen-basis calls and removed norm sum 3.530579238988139. [`treatment_active_gate.json`](cllora_active_freeze_topk2_seed43_paired/treatment_active_gate.json:34) reports removed norm per frozen call 0.000980716455274483, max frozen columns 2, active reassignment path calls 6900, and active adjustment norm sum 50.596070232306374.
- The treatment improves the paired reduced proxy slightly: [`control/metrics.json`](cllora_active_freeze_topk2_seed43_paired/control/metrics.json:8) reports average accuracy 0.5665482637215719 and BWT -0.026559348750060024, while [`treatment/metrics.json`](cllora_active_freeze_topk2_seed43_paired/treatment/metrics.json:8) reports average accuracy 0.5692412473762564 and BWT -0.02357593129097782.

## 4. Main confounders or failure modes

- Single seed and reduced proxy: this is seed 43 only, reduced [`reasoning3`](../cf_cycle_1/run_nullspace_ablation.py:32), rank 4, maximum length 256, and two epochs. It does not establish broad current4, v10, Digital Twin, generative, or production-scale behavior.
- Effect size is below the promotion threshold: [`comparison.json`](cllora_active_freeze_topk2_seed43_paired/comparison.json:4) reports BWT delta plus 0.2983417459082205 points, far below the plus 1.5 point target used in prior cycle decisions; [`comparison.json`](cllora_active_freeze_topk2_seed43_paired/comparison.json:15) reports continue false.
- Old-task utility is mixed: [`comparison.json`](cllora_active_freeze_topk2_seed43_paired/comparison.json:6) reports old-task delta gaps plus 0.7995196940998484 and minus 0.20283620228340737, so the positive average is not uniformly distributed across retained tasks.
- Absolute BWT remains negative in both arms: [`control/metrics.json`](cllora_active_freeze_topk2_seed43_paired/control/metrics.json:9) and [`treatment/metrics.json`](cllora_active_freeze_topk2_seed43_paired/treatment/metrics.json:9) show the treatment lessens forgetting on this proxy but does not eliminate it.
- The worktree had uncommitted changes treated as the fixed envelope. This does not invalidate same-run comparability, but it limits reproducibility unless the report preserves the exact revision plus dirty-envelope caveat.

## 5. Corrected conclusion

The copied-control comparability concern from CF-cycle-7 is resolved for the narrow seed-43 reduced [`reasoning3`](../cf_cycle_1/run_nullspace_ablation.py:32) validation: a fresh paired control produces the same weak-positive deltas rather than erasing the signal. That supports the label evidence-usable, strong mechanism, reproducible weak positive, not promotion-passing.

The top-k treatment should remain a live candidate mechanism, not a promoted solution. The mechanism evidence is strong enough to keep it in the design space, but the utility signal is too small and too mixed to justify escalation to full current4, v10, Digital Twin, generative subset, or broad fuzzer work. The next cycle should revise algorithm design around converting strong mechanism into larger retained-task utility rather than simply scaling the same configuration.

## 6. Confidence level

- Execution and artifact validity: high for this exact experiment, because the preflight, arm gates, active gate, same-envelope check, and final comparison all passed.
- Copied-control concern resolution: medium-high for this exact fixed seed and reduced proxy.
- Claim that the treatment has some real but weak utility on this setup: medium.
- Claim that this is a general catastrophic-forgetting fix: low to contradicted by sub-threshold and negative absolute BWT evidence.

## 7. Proceed, revise, retest, abandon, or blocked

Decision: revise.

Do not abandon [`cllora-active-freeze-topk-2`](../cf_cycle_1/run_nullspace_ablation.py:79), because the active and frozen-basis mechanism evidence is materially real and the paired utility sign is reproducibly positive. Do not proceed to larger escalation, because [`comparison.json`](cllora_active_freeze_topk2_seed43_paired/comparison.json:15) reports continue false and the BWT lift is only plus 0.2983417459082205 points.

## 8. Cheapest next action

Hand off to llm-report-writer for durable closeout of CF-cycle-8, preserving the corrected conclusion and the non-escalation boundary.

Recommended next-cycle seed after reporting: route back to coordination and then planning for a design-revision cycle that keeps the validated top-k active-freeze mechanism as a component but changes the intervention objective or timing to target larger old-task retention. The cheapest useful next experiment should be another bounded reduced-suite design change, not a full-suite escalation. Candidate design axes are cadence or timing of freeze relative to D-MoLE migration, utility-weighted frozen-basis admission, or active adjustment scaling, with the same paired-arm gate discipline used here.

## Hypothesis verdicts

- Fresh paired validation resolves copied-control comparability concern: supported for this narrow fixed setup.
- True top-k utility survives fresh paired control: partially supported as a reproducible weak positive, not as a promotion success.
- Strong mechanism but sub-threshold utility: supported.
- Copied-control benefit vanishes under fresh control: contradicted for this fixed seed-43 reduced proxy.
- Active-capacity mechanism weakens into non-materiality: contradicted.
- Guardrail or artifact failure blocks comparison: contradicted.

## Handoff target

llm-report-writer

The report writer should not overstate this as catastrophic-forgetting solved, promotion-passing, broad utility, or a reason to launch full current4 or Digital Twin work. The report should close CF-cycle-8 as evidence-usable, strong mechanism, reproducible weak positive, not promotion-passing, with revise as the next decision.

## Output contract fields

### evidence_quality

Moderate to strong for exact paired seed-43 reduced-proxy comparability; weak to moderate for broad algorithmic claims.

### hypothesis_verdicts

- Copied-control comparability cleanup: supported for the fixed paired setup.
- Top-k utility: partially supported as reproducible weak positive.
- Promotion or catastrophic-forgetting solution: contradicted or unsupported.
- Mechanism materiality: supported.

### confounders_and_gaps

Single seed, reduced proxy, dirty-but-fixed worktree envelope, mixed old-task gaps, negative absolute BWT, and below-threshold effect size.

### recommended_next_cycle_seed

Use the CF-cycle-8 closeout as the seed for a design-revision cycle, not as a seed for scale-up. Keep the paired-arm gate shape and target the smallest reduced-suite design change likely to turn validated mechanism strength into larger old-task retention.

### handoff_target

llm-report-writer
