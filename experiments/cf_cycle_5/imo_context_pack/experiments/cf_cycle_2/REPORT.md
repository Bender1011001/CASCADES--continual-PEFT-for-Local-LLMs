# CF-cycle-2 Report — Guardrails, reduced-memory protocol, and blocked treatment

## 1. Cycle objective

Close CF-cycle-2 durably after implementing evidence-quality guardrails, documenting the approved reduced-memory reasoning3 protocol, launching the gated control run, and stopping before treatment when the control run failed a hard finite-loss guardrail.

## 2. What was done

- Implemented default-off training guardrails in [`train.py`](../../train.py:171): [`TrainingGuardrailViolation`](../../train.py:171), non-finite loss checks, VRAM threshold checks, [`abort_on_nonfinite`](../../train.py:199), and [`vram_threshold_mb`](../../train.py:199) support.
- Updated the direct ablation runner in [`experiments/cf_cycle_1/run_nullspace_ablation.py`](../cf_cycle_1/run_nullspace_ablation.py:205) to pass guardrails into training, track observed VRAM, expose guardrail CLI flags, and persist [`run_status.json`](nullspace_ablation/control/run_status.json:1) for completed or failed runs.
- Updated the comparator in [`experiments/cf_cycle_1/compare_nullspace_ablation.py`](../cf_cycle_1/compare_nullspace_ablation.py:102) to reject missing, failed, non-finite, over-threshold, or projection-inactive evidence before accepting a comparison.
- Added synthetic invalid-evidence tests in [`tests/test_cf_cycle_2_guardrails.py`](../../tests/test_cf_cycle_2_guardrails.py:1).
- Documented the reduced-memory protocol in [`experiments/cf_cycle_2/README.md`](README.md:1) and [`experiments/cf_cycle_2/EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md:1).
- Launched the control-only GPU run under the approved protocol after a preflight GPU-idle check.
- Stopped the cycle after the control persisted failed_guardrail, blocking treatment and preventing invalid null-space mitigation claims.

## 3. Files changed or inspected

### Guardrail and protocol artifacts

- Changed: [`train.py`](../../train.py:171) — default-off finite-loss and VRAM guardrails.
- Changed: [`experiments/cf_cycle_1/run_nullspace_ablation.py`](../cf_cycle_1/run_nullspace_ablation.py:205) — run status persistence, observed VRAM tracking, and guardrail flag plumbing.
- Changed: [`experiments/cf_cycle_1/compare_nullspace_ablation.py`](../cf_cycle_1/compare_nullspace_ablation.py:102) — invalid evidence rejection.
- Changed: [`tests/test_cf_cycle_2_guardrails.py`](../../tests/test_cf_cycle_2_guardrails.py:1) — CPU-only guardrail rejection tests.
- Changed: [`experiments/cf_cycle_2/README.md`](README.md:1) — cycle runbook and reduced-memory protocol summary.
- Changed: [`experiments/cf_cycle_2/EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md:1) — explicit sequential launch and stop conditions.

### Evidence artifacts

- Inspected: [`experiments/cf_cycle_2/nullspace_ablation/control.log`](nullspace_ablation/control.log:1) — control launch and peak VRAM trace.
- Inspected: [`experiments/cf_cycle_2/nullspace_ablation/control/run_status.json`](nullspace_ablation/control/run_status.json:1) — failed_guardrail status and non-finite loss reason.
- Inspected: [`experiments/cf_cycle_2/result_critic_packet.md`](result_critic_packet.md:1) — critic verdict and next-cycle recommendation.
- Created by report writer: [`experiments/cf_cycle_2/REPORT.md`](REPORT.md:1) — this durable closeout report.
- Updated by report writer: [`CONTEXT.md`](../../CONTEXT.md:1) — project memory and next-cycle handoff.

## 4. Commands/checks run

### CPU-light verification before launch

- Python bytecode compilation check over [`train.py`](../../train.py:1), [`experiments/cf_cycle_1/run_nullspace_ablation.py`](../cf_cycle_1/run_nullspace_ablation.py:1), and [`experiments/cf_cycle_1/compare_nullspace_ablation.py`](../cf_cycle_1/compare_nullspace_ablation.py:1): exit 0.
- Pytest check over [`tests/test_data.py`](../../tests/test_data.py:1) and [`tests/test_cf_cycle_2_guardrails.py`](../../tests/test_cf_cycle_2_guardrails.py:1): exit 0, 16 passed.
- Experiment-designer rerun: combined syntax/tests plus CLI help checks for [`experiments/cf_cycle_1/run_nullspace_ablation.py`](../cf_cycle_1/run_nullspace_ablation.py:364) and [`experiments/cf_cycle_1/compare_nullspace_ablation.py`](../cf_cycle_1/compare_nullspace_ablation.py:176): exit 0.

### GPU launch and rejection checks

- Pre-launch GPU check: `nvidia-smi` showed 835 MiB / 8188 MiB used and 0% utilization, which was idle enough for the reduced-memory control.
- Control run launched with task-suite reasoning3, seed 42, 2 epochs, rank 4, max length 256, and 7500 MB VRAM threshold, writing under [`experiments/cf_cycle_2/nullspace_ablation`](nullspace_ablation/).
- The control log recorded the expected control configuration and first-backward peak VRAM under threshold at [`experiments/cf_cycle_2/nullspace_ablation/control.log`](nullspace_ablation/control.log:24).
- The control failed the finite-loss guardrail and wrote failed status to [`experiments/cf_cycle_2/nullspace_ablation/control/run_status.json`](nullspace_ablation/control/run_status.json:1).
- The comparison command exited 2 and produced invalid JSON because control did not have a completed metrics artifact and treatment artifacts were absent. This confirmed invalid evidence rejection rather than producing algorithm evidence.

### Report-writer checks not rerun

- This report writer did not rerun syntax, pytest, CLI help, GPU launch, or comparison commands. The report records the coordinator, experiment-designer, and result-critic handoff evidence.

## 5. Evidence summary

- Guardrail implementation is present and was CPU-verified before launch.
- The reduced-memory control started with VRAM below the 7500 MB threshold; [`experiments/cf_cycle_2/nullspace_ablation/control.log`](nullspace_ablation/control.log:24) recorded peak=6612MB after the first backward pass.
- The persisted control status is failed_guardrail with reason non-finite training loss task=0 epoch=1 batch=93, peak VRAM 6630.5986328125 MB, and wall time 95.15s in [`experiments/cf_cycle_2/nullspace_ablation/control/run_status.json`](nullspace_ablation/control/run_status.json:1).
- GPU usage returned to low usage after the failed control; no high-memory training Python process remained.
- Memory exhaustion is weakened as a primary explanation because peak VRAM stayed under the protocol threshold.
- Likely causes shifted toward optimization or adapter dynamics, or data-specific instability in reasoning3 Task 0, but root cause remains low-confidence without batch-level instrumentation.
- No valid treatment run exists. No valid comparison exists. No null-space mitigation or catastrophic-forgetting improvement can be claimed.

## 6. Decision: blocked

Treatment remains blocked. The failed control meets the stop condition in [`experiments/cf_cycle_2/EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md:91). CF-cycle-2 should close as a guardrail-success and control-failure cycle, not as a catastrophic-forgetting solution cycle.

Proceed only by starting CF-cycle-3 as debug-first reproduction work. Do not launch treatment until the reasoning3 Task 0 non-finite loss is reproduced, instrumented, and mitigated under a finite, under-threshold control.

## 7. Confidence: medium

- Medium confidence that the guardrails behaved usefully: the control failed fast, persisted rejected evidence, and prevented treatment from running on an invalid baseline.
- Medium confidence that VRAM exhaustion is not the immediate failure cause for this run because observed peak VRAM stayed below threshold.
- Low confidence on the exact non-finite-loss root cause until batch-level sample, token-length, loss, gradient, and adapter diagnostics are collected.
- Unknown confidence for frozen null-space mitigation because treatment never launched and comparison evidence is invalid.

## 8. Next-cycle seed objective

CF-cycle-3 should debug-first reproduce the reasoning3 Task 0 non-finite training loss around batch 93 before any treatment launch.

Minimum next actions:

1. Instrument Task 0 batches near batch 93 with sample IDs, token lengths, input/label statistics, loss values, gradient norms, adapter norms, and SVC/principal-expansion state.
2. Re-run a short control reproduction under the same reduced-memory envelope and guardrails.
3. Test lower learning rates and/or disabled SVC/principal expansion if the non-finite loss reproduces.
4. Only after a completed, finite, under-threshold control exists, reconsider launching treatment.

## 9. Handoff to LLM Loop Coordinator

handoff_target: llm-loop-coordinator

continue_pivot_or_stop: continue, but pivot to debug-first CF-cycle-3 before algorithm treatment.

next_cycle_seed: Reproduce and instrument the reasoning3 Task 0 batch ~93 non-finite loss, then test low-risk stability changes such as lower learning rate and disabled SVC or principal expansion before any null-space treatment launch.

Coordinator routing recommendation: start CF-cycle-3 in Debug mode if immediate instrumentation and reproduction is preferred; use LLM Research Planner only if the coordinator wants a short diagnostic plan before code changes.

Non-negotiable carry-forward: treatment remains blocked, and catastrophic forgetting is not solved.
