# CF-cycle-1 Result Critic Packet

## Evidence quality

- Verdict: low-to-moderate for harness diagnosis; insufficient for algorithmic null-space claims.
- Strong evidence exists that the audit artifacts and direct wrapper were created, including the explicit GPU-precondition warning in [`experiments/cf_cycle_1/README.md`](experiments/cf_cycle_1/README.md:37) and [`experiments/cf_cycle_1/harness_audit.log`](experiments/cf_cycle_1/harness_audit.log:92).
- Strong evidence exists that a control arm was launched with the intended control flags in [`experiments/cf_cycle_1/nullspace_ablation/control/config.json`](experiments/cf_cycle_1/nullspace_ablation/control/config.json:1) and current4 manifest in [`experiments/cf_cycle_1/nullspace_ablation/control/task_manifest.json`](experiments/cf_cycle_1/nullspace_ablation/control/task_manifest.json:1).
- The run is partial and non-comparable: no [`experiments/cf_cycle_1/nullspace_ablation/control/metrics.json`](experiments/cf_cycle_1/nullspace_ablation/control/metrics.json), no [`experiments/cf_cycle_1/nullspace_ablation/control/instrumentation.json`](experiments/cf_cycle_1/nullspace_ablation/control/instrumentation.json), and no treatment arm artifacts exist.
- The control log shows repeated VRAM threshold breaches before and during later tasks, including Task 1 at [`experiments/cf_cycle_1/nullspace_ablation/control.log`](experiments/cf_cycle_1/nullspace_ablation/control.log:66), Task 2 at [`experiments/cf_cycle_1/nullspace_ablation/control.log`](experiments/cf_cycle_1/nullspace_ablation/control.log:131), and Task 3 at [`experiments/cf_cycle_1/nullspace_ablation/control.log`](experiments/cf_cycle_1/nullspace_ablation/control.log:179).
- The control log shows non-finite training on the Digital Twin task with loss nan at [`experiments/cf_cycle_1/nullspace_ablation/control.log`](experiments/cf_cycle_1/nullspace_ablation/control.log:181), so any eventual final metrics from this process should be treated as failed-run diagnostics unless the log later proves recovery and finite weights.

## Protocol adherence

- Launching the GPU control arm before a returned experiment-designer handoff was a protocol deviation. The README says not to run GPU commands unless explicit launch follows audit review at [`experiments/cf_cycle_1/README.md`](experiments/cf_cycle_1/README.md:37), and the audit snapshot says GPU preconditions were not satisfied at [`experiments/cf_cycle_1/harness_audit.log`](experiments/cf_cycle_1/harness_audit.log:92).
- The deviation does not make the control log useless; it makes it diagnostic, not confirmatory.
- The interrupted handoff is best classified as an orchestration failure: the loop lost the decision checkpoint between audit/design and GPU execution. It is not currently evidence of a core training-code bug by itself.

## Hypothesis verdicts

1. Frozen null-space protection improves old-task retention: not addressed. Treatment was not run, and projection activation evidence required by [`experiments/cf_cycle_1/README.md`](experiments/cf_cycle_1/README.md:69) is absent.
2. Harness drift/runner mismatch explains non-comparable prior results: supported. The audit documents task-suite drift, dropped runner overrides, and unused sample controls at [`experiments/cf_cycle_1/harness_audit.log`](experiments/cf_cycle_1/harness_audit.log:72).
3. Current4 at rank 8 and max length 384 is safe on RTX 4060 Ti 8GB: weakened or falsified for this run. The observed peaks exceed the threshold, and the run reaches non-finite loss on Task 3.
4. Current4 is a clean first algorithmic forgetting ablation: weakened. It is valuable as the project’s real path, but Digital Twin confounds parametric recall/noise with BWT geometry. The parametric-memory report identifies noisy Digital Twin data at [`reports/parametric_memory_issue/REPORT.md`](reports/parametric_memory_issue/REPORT.md:51).

## Confounders and gaps

- No treatment metrics; no comparison can be made.
- No completed control metrics or instrumentation; the run is still incomplete or failed.
- VRAM warnings are warnings in [`train.py`](train.py:350), not hard stops, so the run can continue after violating experiment constraints.
- [`experiments/cf_cycle_1/compare_nullspace_ablation.py`](experiments/cf_cycle_1/compare_nullspace_ablation.py:40) only enforces the VRAM threshold on treatment, not control, and does not guard against non-finite metrics.
- The direct wrapper’s task-suite mutation is probably sufficient in its current fresh-subprocess order because it mutates [`cascades.data`](cascades/data.py:35) before importing [`train.py`](train.py:44) and then patches [`train.NUM_TASKS`](train.py:294) via [`experiments/cf_cycle_1/run_nullspace_ablation.py`](experiments/cf_cycle_1/run_nullspace_ablation.py:209). It remains brittle because it depends on import order and module globals instead of explicit task-file plumbing.

## Immediate recommendation

- Block treatment. Do not run the current4 treatment arm from this artifact set.
- Do not kill the running control process without user or Roo approval. If approval is available, terminate it because non-finite loss plus threshold breaches mean it is no longer a valid ablation arm. Without approval, let it finish only as a failed diagnostic and archive the log.
- Before any new GPU treatment, add hard finite-loss and peak-VRAM guardrails, and make comparison require both arms to be finite and below threshold.

## Recommended next-cycle seed

Redesign CF-cycle-1 as a clean feasibility-first reasoning3 ablation: use [`experiments/cf_cycle_1/run_nullspace_ablation.py`](experiments/cf_cycle_1/run_nullspace_ablation.py:284) with reasoning3, reduced memory settings such as rank 4 and max length 256, and hard aborts on non-finite loss or peak VRAM above 7500 MB. Only after reasoning3 produces finite control/treatment metrics and active projection evidence should the loop return to current4 or Digital Twin-specific retention.

## Handoff target

handoff_target: llm-report-writer

Please write a durable report that states the current control run is diagnostic only, treatment is blocked, the GPU launch was a protocol deviation relative to the audit artifacts, and the strongest next cycle is a reduced-memory reasoning3 ablation with hard runtime guardrails.
