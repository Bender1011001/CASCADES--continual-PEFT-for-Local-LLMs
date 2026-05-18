# CF-cycle-3 result-critic packet — reduced-memory reasoning3 treatment comparison

Date: 2026-05-17

Handoff target: llm-report-writer

## 1. Claim reviewed

The reduced-memory reasoning3 control/treatment comparison after the data-loader fix is a valid feasibility run and provides enough evidence to decide whether frozen null-space treatment should escalate toward full current4 or v10 evaluation.

## 2. Evidence quality

Moderate overall.

- Strong for feasibility validity: both arms completed with matching parameters and under the 7500 MB VRAM threshold in [`control/run_status.json`](experiments/cf_cycle_3/nullspace_ablation_retry/control/run_status.json:1), [`treatment/run_status.json`](experiments/cf_cycle_3/nullspace_ablation_retry/treatment/run_status.json:1), and [`comparison.json`](experiments/cf_cycle_3/nullspace_ablation_retry/comparison.json:1).
- Moderate for bounded algorithmic direction: treatment slightly improved the proxy metrics in [`comparison.json`](experiments/cf_cycle_3/nullspace_ablation_retry/comparison.json:4), but the effect is below the predeclared BWT threshold in [`EXPERIMENT_PROTOCOL.md`](experiments/cf_cycle_2/EXPERIMENT_PROTOCOL.md:114).
- Weak for broad catastrophic-forgetting claims: this is one seed, a 3-task reduced suite, max length 256, 2 epochs, and proxy matrix evaluation rather than full current4 or generative behavior.

## 3. Main support

- The comparison gate reports valid true, no failures, active projection evidence, non-empty frozen basis evidence, and under-threshold VRAM in [`comparison.json`](experiments/cf_cycle_3/nullspace_ablation_retry/comparison.json:2).
- Control completed with seed 42, rank 4, max length 256, 2 epochs, revision 7c4e01d, and peak VRAM 6848.810546875 MB in [`control/run_status.json`](experiments/cf_cycle_3/nullspace_ablation_retry/control/run_status.json:3).
- Treatment completed with the same suite, seed, rank, max length, epochs, revision, and peak VRAM 6850.75390625 MB in [`treatment/run_status.json`](experiments/cf_cycle_3/nullspace_ablation_retry/treatment/run_status.json:3).
- Task manifests match across all three reasoning tasks in [`control/task_manifest.json`](experiments/cf_cycle_3/nullspace_ablation_retry/control/task_manifest.json:1) and [`treatment/task_manifest.json`](experiments/cf_cycle_3/nullspace_ablation_retry/treatment/task_manifest.json:1).
- Treatment projection was not a no-op: [`instrumentation.json`](experiments/cf_cycle_3/nullspace_ablation_retry/treatment/instrumentation.json:694) reports calls with frozen basis and positive removed norm, and non-empty frozen basis events appear in [`instrumentation.json`](experiments/cf_cycle_3/nullspace_ablation_retry/treatment/instrumentation.json:385).
- Treatment improved BWT by only 0.36167033073388477 points and average accuracy by 0.24463211986087696 points in [`comparison.json`](experiments/cf_cycle_3/nullspace_ablation_retry/comparison.json:4). The old-task relative gaps are positive in [`comparison.json`](experiments/cf_cycle_3/nullspace_ablation_retry/comparison.json:6).
- The comparison logic requires at least 1.5 BWT points, no more than 2.0 points average-accuracy regression, and no old-task relative gap worse than minus 3.0 points before continuing in [`compare_runs()`](experiments/cf_cycle_1/compare_nullspace_ablation.py:164). The protocol says a valid comparison with continue false is valid reduced-memory evidence but does not support algorithmic escalation in [`EXPERIMENT_PROTOCOL.md`](experiments/cf_cycle_2/EXPERIMENT_PROTOCOL.md:120).

## 4. Main confounders or failure modes

- One-seed evidence is noisy. Seed 42 can justify a bounded next action, but not a stable effect estimate.
- The reduced reasoning3 suite is not current4, not v10, and does not include the Digital Twin task. It should not be generalized to the broad catastrophic-forgetting goal.
- Max length 256 plus response-preserving truncation through [`_build_supervised_sequence()`](cascades/data.py:80) and [`prepare_data()`](cascades/data.py:107) makes the two arms comparable to each other, but not directly comparable to old invalid current4 artifacts or longer-context runs.
- Treatment still has negative absolute forgetting: [`treatment/metrics.json`](experiments/cf_cycle_3/nullspace_ablation_retry/treatment/metrics.json:9) reports BWT minus 0.03252861922089642, and [`treatment/metrics.json`](experiments/cf_cycle_3/nullspace_ablation_retry/treatment/metrics.json:20) reports a task-0 old-task delta around minus 6.35 points.
- The current working tree is dirty. Both run-status files record revision 7c4e01d in [`control/run_status.json`](experiments/cf_cycle_3/nullspace_ablation_retry/control/run_status.json:12) and [`treatment/run_status.json`](experiments/cf_cycle_3/nullspace_ablation_retry/treatment/run_status.json:12), and current HEAD matches that revision, but the run-status schema does not capture dirty-tree state. Provenance is acceptable for a loop decision, not archival-grade reproducibility.
- Matrix metrics are proxy evidence. No generative subset, held-out output-quality check, or real user-facing retention test was run.

## 5. Corrected conclusion

This is a valid feasibility run. It does not support the claim that frozen null-space treatment solves catastrophic forgetting. The treatment was active and did not harm the reduced proxy metrics, but the measured BWT gain is far below the predeclared escalation threshold and absolute forgetting remains negative.

Interpret continue false as: do not escalate to full current4 or v10 yet. Do not interpret it as a feasibility failure or proof that the whole null-space idea is dead.

## 6. Confidence level

- High confidence that feasibility checks passed.
- High confidence that algorithmic success threshold was not met.
- Medium confidence that the current treatment configuration is at best a weak positive signal, not a breakthrough.
- Low confidence about generalization beyond this seed and reduced suite.

## 7. Proceed / revise / retest / abandon / blocked

Decision: block escalation and retest once under the same reduced protocol before larger runs.

- Block full current4, v10, and Digital Twin escalation from this evidence.
- Do not adjust success criteria downward after seeing the result.
- Do not claim catastrophic forgetting is solved.
- Retest one additional seed under the same reduced protocol to estimate whether the small positive BWT signal is stable or noise.
- If the additional seed again misses the BWT threshold or flips sign, revise the treatment rather than spending more runs on the same weak configuration.

## 8. Cheapest next action

Run one seed-43 replication of the same reduced-memory reasoning3 protocol, sequential control then treatment, in a fresh output root. Use the same pass/fail gate from [`EXPERIMENT_PROTOCOL.md`](experiments/cf_cycle_2/EXPERIMENT_PROTOCOL.md:114). This is cheaper and cleaner than immediate full-suite escalation, safer than changing success criteria, and more decision-relevant than a generative subset before the proxy effect clears its own threshold.

Recommended next-cycle seed for the loop:

CF-cycle-4 should test whether the weak positive seed-42 effect is reproducible with one additional reduced-memory reasoning3 seed. If seed 43 does not reach the predeclared BWT threshold without accuracy or old-task regressions, pivot to a revised treatment-strength hypothesis such as stronger projection, altered frozen-basis construction, or reassignment behavior before any full-suite run.

## Hypothesis verdicts

| Hypothesis or claim | Verdict | Confidence | Notes |
|---|---|---:|---|
| Data-loader fix enabled a valid reduced-memory comparison | Supported | High | Both arms completed and comparison is valid. |
| Treatment projection was active | Supported | High | Frozen-basis calls and removed norm are present. |
| Treatment improved BWT enough for escalation | Contradicted | High | Delta BWT is 0.3617 points, below the 1.5 point threshold. |
| Treatment harms reduced proxy accuracy | Not supported | Medium | Average accuracy and relative old-task gaps are slightly positive. |
| Null-space treatment solves catastrophic forgetting | Unsupported | High | One seed, reduced suite, negative absolute BWT remains, and no generative validation. |

## Report-writer guardrails

- Report the run as valid feasibility evidence, not as solved catastrophic forgetting.
- Preserve the distinction between relative old-task gaps passing and absolute forgetting remaining negative.
- Preserve the dirty-tree provenance caveat.
- State that continue false blocks escalation, not all future investigation.
- Seed the next loop with one additional reduced-memory seed before broader or more expensive experiments.
