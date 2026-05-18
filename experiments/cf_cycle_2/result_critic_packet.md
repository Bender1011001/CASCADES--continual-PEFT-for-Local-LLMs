# CF-cycle-2 Result Critic Packet

## Claim reviewed

The approved reduced-memory [`reasoning3`](experiments/cf_cycle_2/nullspace_ablation/control/run_status.json:4) control launch tested whether the repaired guardrails allow a finite, under-threshold control arm to complete before spending GPU time on treatment.

## Evidence quality

Moderate for guardrail and stop/continue decisions; weak to unavailable for any null-space mitigation or catastrophic-forgetting claim.

## Main support

- The protocol explicitly required control-first execution and treatment only after a completed, finite, under-threshold control at [`experiments/cf_cycle_2/EXPERIMENT_PROTOCOL.md`](experiments/cf_cycle_2/EXPERIMENT_PROTOCOL.md:41).
- The same protocol states that any failed control gate must stop treatment and preserve failed-run evidence at [`experiments/cf_cycle_2/EXPERIMENT_PROTOCOL.md`](experiments/cf_cycle_2/EXPERIMENT_PROTOCOL.md:91).
- The control launch used the expected control configuration: [`enable_coso_nullspace=False`](experiments/cf_cycle_2/nullspace_ablation/control/config.json:18), [`enable_cllora_reassign=False`](experiments/cf_cycle_2/nullspace_ablation/control/config.json:19), [`rank=4`](experiments/cf_cycle_2/nullspace_ablation/control/config.json:7), [`max_length=256`](experiments/cf_cycle_2/nullspace_ablation/control/config.json:8), and [`epochs=2`](experiments/cf_cycle_2/nullspace_ablation/control/config.json:6).
- The control guardrail persisted [`status=failed_guardrail`](experiments/cf_cycle_2/nullspace_ablation/control/run_status.json:3) with reason [`non-finite training loss task=0 epoch=1 batch=93`](experiments/cf_cycle_2/nullspace_ablation/control/run_status.json:13).
- Peak VRAM was finite and under threshold: [`peak_vram_mb=6630.5986328125`](experiments/cf_cycle_2/nullspace_ablation/control/run_status.json:14) versus [`vram_threshold_mb=7500.0`](experiments/cf_cycle_2/nullspace_ablation/control/run_status.json:9).
- The task manifest matches the intended three-task suite: GSM8K at [`experiments/cf_cycle_2/nullspace_ablation/control/task_manifest.json`](experiments/cf_cycle_2/nullspace_ablation/control/task_manifest.json:4), ARC at [`experiments/cf_cycle_2/nullspace_ablation/control/task_manifest.json`](experiments/cf_cycle_2/nullspace_ablation/control/task_manifest.json:11), and CSQA at [`experiments/cf_cycle_2/nullspace_ablation/control/task_manifest.json`](experiments/cf_cycle_2/nullspace_ablation/control/task_manifest.json:18).
- The comparison gate requires both arms and required artifacts at [`experiments/cf_cycle_1/compare_nullspace_ablation.py`](experiments/cf_cycle_1/compare_nullspace_ablation.py:10) and [`experiments/cf_cycle_1/compare_nullspace_ablation.py`](experiments/cf_cycle_1/compare_nullspace_ablation.py:11), then rejects missing artifacts at [`experiments/cf_cycle_1/compare_nullspace_ablation.py`](experiments/cf_cycle_1/compare_nullspace_ablation.py:40).

## Main confounders or failure modes

- The failure is localized only to one control run with seed [`seed=42`](experiments/cf_cycle_2/nullspace_ablation/control/run_status.json:5); it is enough to stop treatment under protocol, but not enough to diagnose root cause conclusively.
- The last visible training log is batch 50 with finite loss at [`experiments/cf_cycle_2/nullspace_ablation/control.log`](experiments/cf_cycle_2/nullspace_ablation/control.log:25), while the abort reason identifies batch 93 at [`experiments/cf_cycle_2/nullspace_ablation/control/run_status.json`](experiments/cf_cycle_2/nullspace_ablation/control/run_status.json:13); the transition from finite to non-finite is not instrumented in the artifact.
- The run included higher-risk dynamics that could plausibly destabilize training independently of null-space treatment: SVC enabled at [`experiments/cf_cycle_2/nullspace_ablation/control/config.json`](experiments/cf_cycle_2/nullspace_ablation/control/config.json:20), principal expansion enabled at [`experiments/cf_cycle_2/nullspace_ablation/control/config.json`](experiments/cf_cycle_2/nullspace_ablation/control/config.json:26), sleep enabled at [`experiments/cf_cycle_2/nullspace_ablation/control/config.json`](experiments/cf_cycle_2/nullspace_ablation/control/config.json:10), and relatively high learning rates logged at [`experiments/cf_cycle_2/nullspace_ablation/control.log`](experiments/cf_cycle_2/nullspace_ablation/control.log:8).
- A data-specific issue remains plausible because the failure occurred in Task 0 from [`data/task0_gsm8k_cot.jsonl`](experiments/cf_cycle_2/nullspace_ablation/control/task_manifest.json:4), but no batch-level sample identifiers or token statistics were persisted.
- The absence of [`metrics.json`](experiments/cf_cycle_2/EXPERIMENT_PROTOCOL.md:74) is expected after fail-fast behavior, but it means no accuracy, BWT, or old-task-delta interpretation is available.

## Corrected conclusion

The control failure does meet the protocol stop condition. Treatment should remain blocked. CF-cycle-2 validates that the repaired guardrails can fail fast, persist a failed-run status, and prevent incomplete evidence from being accepted; it does not validate reduced-memory finite training, treatment projection behavior, or catastrophic-forgetting mitigation.

Likely interpretation, using evidence labels:

- Confirmed: non-finite control loss occurred before completion, and the run stayed below the VRAM threshold.
- Likely: the guardrail behaved as intended by aborting before downstream artifacts could be misread as valid evidence.
- Plausible: instability came from optimization or adapter dynamics rather than raw GPU memory exhaustion, because peak VRAM stayed below threshold.
- Plausible but unproven: Task 0 batch content, high learning rates, SVC/principal expansion, quantization noise, or sleep interaction triggered the non-finite loss.
- Weak signal: the batch 93 location may identify a reproducible problematic minibatch, but current artifacts do not show sample IDs or token-level details.
- Contradicted: this is not evidence that null-space treatment fails, because treatment was correctly not launched.

## Confidence level

High for stop treatment and hand to report writer. Medium for guardrail-validation interpretation. Low for root-cause attribution.

## Proceed / revise / retest / abandon / blocked

Blocked for treatment. Proceed to report writer for durable closeout of CF-cycle-2. Retest only after a focused debug/code step identifies or reduces the non-finite source.

## Cheapest next action

Create CF-cycle-3 as a debug-first cycle: reproduce Task 0 around batch 93 with minimal GPU time, persist batch/sample IDs and loss components, then run a one-epoch or bounded-step control probe with lower learning rates and optionally disabled SVC/principal expansion before any new treatment launch.

## Evidence quality

Moderate: command/config/status artifacts are sufficient for stop/continue; root-cause evidence is incomplete; algorithmic evidence is unavailable.

## Hypothesis verdicts

- H1 guardrails catch invalid reduced-memory runs: supported at medium confidence by [`status=failed_guardrail`](experiments/cf_cycle_2/nullspace_ablation/control/run_status.json:3), persisted reason at [`experiments/cf_cycle_2/nullspace_ablation/control/run_status.json`](experiments/cf_cycle_2/nullspace_ablation/control/run_status.json:13), and under-threshold VRAM at [`experiments/cf_cycle_2/nullspace_ablation/control/run_status.json`](experiments/cf_cycle_2/nullspace_ablation/control/run_status.json:14).
- H2 reduced-memory control can produce finite comparable evidence: weakened or falsified for this exact configuration and seed, because no [`metrics.json`](experiments/cf_cycle_2/EXPERIMENT_PROTOCOL.md:74) was produced and [`run_status`](experiments/cf_cycle_2/nullspace_ablation/control/run_status.json:3) is failed.
- H3 treatment null-space behavior improves forgetting: not addressed; treatment did not run by design.
- H4 comparison rejects incomplete artifacts: supported by script requirements at [`experiments/cf_cycle_1/compare_nullspace_ablation.py`](experiments/cf_cycle_1/compare_nullspace_ablation.py:10) and missing-artifact rejection at [`experiments/cf_cycle_1/compare_nullspace_ablation.py`](experiments/cf_cycle_1/compare_nullspace_ablation.py:40), plus the reported nonzero comparison outcome.

## Confounders and gaps

- Missing batch-level data provenance for the failing minibatch.
- Missing loss-component, gradient-norm, and parameter-norm traces from batches 50 to 93.
- Unknown whether failure reproduces across the same seed after rerun or after disabling SVC/principal expansion.
- Unknown whether the failure is caused by data, optimizer learning rate, quantization, sleep/SVC/principal expansion, or guardrail sensitivity.

## Recommended next-cycle seed

CF-cycle-3 should debug non-finite loss in [`reasoning3`](experiments/cf_cycle_2/nullspace_ablation/control/run_status.json:4) control Task 0 near batch 93 using a CPU/GPU-light reproduction path. Instrument batch/sample IDs, token lengths, loss, gradient norms, and adapter norms; then test the cheapest stabilizing changes in order: lower learning rates, disabled SVC/principal expansion, and targeted data inspection. Do not launch treatment until a finite completed control exists.

## Handoff target

llm-report-writer
