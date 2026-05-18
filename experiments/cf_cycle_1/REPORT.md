# CF-cycle-1 Report: Frozen Null-Space Ablation Audit

## Cycle report

- Objective: turn the broad catastrophic-forgetting goal into an auditable, controlled frozen null-space ablation without claiming that catastrophic forgetting is solved.
- Planned control: [`enable_coso_nullspace=False`](experiments/cf_cycle_1/nullspace_ablation/control/config.json:15) and [`enable_cllora_reassign=False`](experiments/cf_cycle_1/nullspace_ablation/control/config.json:16).
- Planned treatment: [`enable_coso_nullspace=True`](experiments/cf_cycle_1/run_nullspace_ablation.py:84) and [`enable_cllora_reassign=False`](experiments/cf_cycle_1/run_nullspace_ablation.py:85), not executed in this cycle.
- Intended success criteria: treatment BWT better than control by at least +1.5 points, final ACC no more than 2 points worse, no old-task delta worse by more than 3 points, and peak VRAM below 7500 MB.
- Outcome: this cycle produced useful harness diagnostics and a partial invalid control-run diagnostic; it did not produce valid algorithmic evidence for frozen null-space protection.

## Durable artifacts

| Artifact | Status | Purpose |
|---|---:|---|
| [`experiments/cf_cycle_1/README.md`](experiments/cf_cycle_1/README.md:1) | Created | Audit-first design, CPU-light checks, explicit GPU commands, arm definitions, required artifacts, and projection guardrails. |
| [`experiments/cf_cycle_1/harness_audit.log`](experiments/cf_cycle_1/harness_audit.log:1) | Created | Harness-drift and precondition snapshot. |
| [`experiments/cf_cycle_1/run_nullspace_ablation.py`](experiments/cf_cycle_1/run_nullspace_ablation.py:1) | Created | Direct wrapper with task suites, arm configs, instrumentation, and metrics persistence. |
| [`experiments/cf_cycle_1/result_critic_packet.md`](experiments/cf_cycle_1/result_critic_packet.md:1) | Created | Skeptical critique and next-cycle recommendation. |
| [`experiments/cf_cycle_1/nullspace_ablation/control/config.json`](experiments/cf_cycle_1/nullspace_ablation/control/config.json:1) | Partial | Confirms the launched control-arm configuration. |
| [`experiments/cf_cycle_1/nullspace_ablation/control.log`](experiments/cf_cycle_1/nullspace_ablation/control.log:1) | Partial diagnostic | CUDA control run log; invalid as confirmatory evidence due to protocol deviation, VRAM breaches, and non-finite loss. |

## Evidence summary

### Planner and hypothesis narrowing

- The loop narrowed the user goal from catastrophic-forgetting improvement to a two-arm null-space ablation with [`enable_cllora_reassign=False`](experiments/cf_cycle_1/run_nullspace_ablation.py:85) in both arms so the treatment isolates frozen-basis projection rather than active sketch reassignment.
- The runner supports both [`current4`](experiments/cf_cycle_1/run_nullspace_ablation.py:26) and [`reasoning3`](experiments/cf_cycle_1/run_nullspace_ablation.py:32), but the first proposed GPU design used [`current4`](experiments/cf_cycle_1/harness_audit.log:7), which includes the large Digital Twin task at [`data/task3_digital_twin.jsonl`](experiments/cf_cycle_1/harness_audit.log:43).

### Harness audit findings

- The current official suite is four tasks, shown by [`current_num_tasks`](experiments/cf_cycle_1/harness_audit.log:7), while the lighter reasoning suite is three tasks, shown by [`reasoning3_task_files`](experiments/cf_cycle_1/harness_audit.log:14).
- [`research_runner.ExperimentRunner._run_cascades()`](research_runner.py:1) is not safe for this controlled ablation because the audit records [`forwards_rank=false`](experiments/cf_cycle_1/harness_audit.log:74), [`forwards_max_length=false`](experiments/cf_cycle_1/harness_audit.log:75), and [`pops_task_files=true`](experiments/cf_cycle_1/harness_audit.log:76).
- [`num_samples`](experiments/cf_cycle_1/harness_audit.log:79) is not wired to [`prepare_data()`](experiments/cf_cycle_1/harness_audit.log:85), so sample-count control should not be treated as part of the experiment contract yet.
- GPU preconditions were explicitly not satisfied: [`gpu_preconditions.ok=false`](experiments/cf_cycle_1/harness_audit.log:92) with the reason at [`experiments/cf_cycle_1/harness_audit.log`](experiments/cf_cycle_1/harness_audit.log:94).

### CPU-light verification

- Verification command executed after the interrupted work and exited 0: Python bytecode compilation over [`experiments/cf_cycle_1/harness_audit.py`](experiments/cf_cycle_1/harness_audit.py:1), [`experiments/cf_cycle_1/run_nullspace_ablation.py`](experiments/cf_cycle_1/run_nullspace_ablation.py:1), and [`experiments/cf_cycle_1/compare_nullspace_ablation.py`](experiments/cf_cycle_1/compare_nullspace_ablation.py:1), followed by the quiet pytest target [`tests/test_data.py`](tests/test_data.py:1).
- Result: 11 tests in [`tests/test_data.py`](tests/test_data.py:1) passed.
- This supersedes the earlier stale note in [`CONTEXT.md`](CONTEXT.md:27) that said [`tests/test_data.py`](tests/test_data.py:1) was failing.

### GPU control-arm launch and deviation

- The experiment README explicitly states not to run GPU commands until after audit review and explicit launch approval at [`experiments/cf_cycle_1/README.md`](experiments/cf_cycle_1/README.md:37).
- Despite that, a control arm was launched. Its config confirms [`arm=control`](experiments/cf_cycle_1/nullspace_ablation/control/config.json:2), [`task_suite=current4`](experiments/cf_cycle_1/nullspace_ablation/control/config.json:3), [`seed=42`](experiments/cf_cycle_1/nullspace_ablation/control/config.json:5), [`epochs=2`](experiments/cf_cycle_1/nullspace_ablation/control/config.json:6), [`rank=8`](experiments/cf_cycle_1/nullspace_ablation/control/config.json:7), [`max_length=384`](experiments/cf_cycle_1/nullspace_ablation/control/config.json:8), [`enable_sleep=true`](experiments/cf_cycle_1/nullspace_ablation/control/config.json:10), and [`git_revision=7c4e01d`](experiments/cf_cycle_1/nullspace_ablation/control/config.json:27).
- This GPU launch is a protocol deviation relative to [`experiments/cf_cycle_1/README.md`](experiments/cf_cycle_1/README.md:39) and [`gpu_preconditions.ok=false`](experiments/cf_cycle_1/harness_audit.log:93).

### GPU diagnostics from partial control log

- The control run used CUDA, shown by [`Device: cuda`](experiments/cf_cycle_1/nullspace_ablation/control.log:5), with the intended control flags in [`experiments/cf_cycle_1/nullspace_ablation/control.log`](experiments/cf_cycle_1/nullspace_ablation/control.log:6).
- Peak VRAM exceeded the 7500 MB guardrail at Task 1: [`Peak 7843MB`](experiments/cf_cycle_1/nullspace_ablation/control.log:67).
- Peak VRAM exceeded the guardrail again at Task 2 with negative free headroom: [`free=-370MB`](experiments/cf_cycle_1/nullspace_ablation/control.log:131) and [`Peak 8666MB`](experiments/cf_cycle_1/nullspace_ablation/control.log:132).
- Peak VRAM exceeded the guardrail again at Task 3 with worse negative free headroom: [`free=-1002MB`](experiments/cf_cycle_1/nullspace_ablation/control.log:179) and [`Peak 8719MB`](experiments/cf_cycle_1/nullspace_ablation/control.log:180).
- The Digital Twin task reached non-finite training loss: [`loss=nan`](experiments/cf_cycle_1/nullspace_ablation/control.log:181), with another [`loss=nan`](experiments/cf_cycle_1/nullspace_ablation/control.log:191).
- No completed [`experiments/cf_cycle_1/nullspace_ablation/control/metrics.json`](experiments/cf_cycle_1/nullspace_ablation/control/metrics.json), no completed [`experiments/cf_cycle_1/nullspace_ablation/control/instrumentation.json`](experiments/cf_cycle_1/nullspace_ablation/control/instrumentation.json), and no treatment artifacts exist.

## Critic verdicts and decisions

- Evidence quality is low-to-moderate for harness diagnosis and insufficient for algorithmic null-space claims, matching [`experiments/cf_cycle_1/result_critic_packet.md`](experiments/cf_cycle_1/result_critic_packet.md:5).
- The current control run is diagnostic only, not confirmatory, because the GPU launch violated the audit checkpoint, exceeded VRAM guardrails, reached non-finite loss, and has no completed metrics or instrumentation.
- Treatment is blocked. Do not run a [`current4`](experiments/cf_cycle_1/run_nullspace_ablation.py:26) treatment from this artifact set.
- Do not claim catastrophic forgetting is solved. This cycle diagnosed harness issues and prevented an invalid treatment comparison.
- If process control approval exists, terminate any still-running invalid control process; otherwise, let it finish only as a failed diagnostic and archive the log.

## Immediate recommendation

Before any further GPU treatment:

1. Add hard aborts for non-finite loss and peak VRAM above 7500 MB.
2. Make comparison require finite metrics for both arms and peak VRAM below threshold for both arms.
3. Re-run a smaller feasibility ablation before returning to the Digital Twin task.
4. Treat projection evidence as mandatory: frozen bases must become non-empty, [`_cllora_reassign()`](cascades/adapters.py:1) must receive a frozen basis, and removed norm must be positive, consistent with the guardrail at [`experiments/cf_cycle_1/README.md`](experiments/cf_cycle_1/README.md:69).

## Next-cycle seed

Ranked next objective: continue with a reduced-memory [`reasoning3`](experiments/cf_cycle_1/run_nullspace_ablation.py:32) null-space feasibility ablation using [`experiments/cf_cycle_1/run_nullspace_ablation.py`](experiments/cf_cycle_1/run_nullspace_ablation.py:284), for example rank 4 and max length 256, with hard finite-loss and VRAM-abort guardrails.

Success criteria for the next cycle:

- Both control and treatment complete with finite [`metrics.json`](experiments/cf_cycle_1/nullspace_ablation/control/metrics.json) and [`instrumentation.json`](experiments/cf_cycle_1/nullspace_ablation/control/instrumentation.json) equivalents.
- Both arms stay below 7500 MB peak VRAM.
- Treatment has active projection evidence, not just enabled config flags.
- Only after finite control/treatment metrics and projection evidence should the loop return to [`current4`](experiments/cf_cycle_1/run_nullspace_ablation.py:26) or Digital Twin-specific retention.

## Continue, pivot, or stop

- Decision: continue, but pivot from the invalid [`current4`](experiments/cf_cycle_1/run_nullspace_ablation.py:26) artifact set to a reduced-memory [`reasoning3`](experiments/cf_cycle_1/run_nullspace_ablation.py:32) feasibility ablation.
- Stop condition: stop only if the reduced-memory feasibility run still cannot produce finite metrics below the VRAM threshold after hard guardrails are added.

## Handoff target

- handoff_target: llm-loop-coordinator
- Coordinator seed: reduced-memory reasoning3 null-space ablation with hard finite/VRAM guardrails.
- Coordinator instruction: start the next cycle by sending the planner/designer to redesign and patch the runner guardrails before any new treatment launch.
