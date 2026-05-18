# CF-cycle-3 Report — data-loader fix, reduced-memory retry, and blocked escalation

Date: 2026-05-17

## 1. Cycle objective

Close CF-cycle-3 durably after diagnosing and fixing the reduced-memory reasoning3 all-ignored-label bug, rerunning the corrected seed-42 control and treatment under the same guardrails, comparing the arms, and preserving the result-critic decision for the next coordinator cycle.

Evidence state: confirmed for the data bug and feasibility gates; likely but not proven for the small positive treatment direction; unsupported for any claim that catastrophic forgetting is solved.

## 2. What was done

- Debugged the non-finite control failure from CF-cycle-2. The root cause packet [`root_cause_packet.md`](root_cause_packet.md:1) confirmed that Task 0 Batch 93 at max length 256 produced all-ignored labels because prompt-first truncation left no supervised response labels.
- Created static probe artifacts: [`batch93_probe.py`](batch93_probe.py:1), [`batch93_static_probe.json`](batch93_static_probe.json:1), and [`batch93_static_probe.verify.json`](batch93_static_probe.verify.json:1).
- Fixed the loader by adding response-preserving truncation in [`_build_supervised_sequence()`](../../cascades/data.py:80), used by [`prepare_data()`](../../cascades/data.py:107), so reduced max length still reserves at least one response label when possible.
- Added regression coverage in [`tests/test_data.py`](../../tests/test_data.py:147). Green evidence reported 14 data tests passing, and the fixed static scan reported no zero-label batches with batch 93 valid labels equal to 1 in [`batch93_static_probe.verify.json`](batch93_static_probe.verify.json:6).
- Added and ran reasoning3 preflight checks with [`reasoning3_prepare_data_preflight.py`](reasoning3_prepare_data_preflight.py:1). The output [`reasoning3_prepare_data_preflight.json`](reasoning3_prepare_data_preflight.json:1) reports valid true, all three reasoning3 tasks valid, no zero-label batches, and max sequence length 256.
- Added and ran the control-gate validator [`validate_control_gate.py`](validate_control_gate.py:1). The output [`control_gate_validation.json`](control_gate_validation.json:1) reports valid true for the retried control.
- Completed the corrected control under [`nullspace_ablation_retry/control/run_status.json`](nullspace_ablation_retry/control/run_status.json:1): status completed, seed 42, rank 4, max length 256, 2 epochs, peak VRAM 6848.810546875 MB.
- Completed the treatment-only run under [`nullspace_ablation_retry/treatment/run_status.json`](nullspace_ablation_retry/treatment/run_status.json:1): status completed, seed 42, rank 4, max length 256, 2 epochs, peak VRAM 6850.75390625 MB. The run was treatment-only; the runner's both-arm launch mode was not used.
- Compared the completed arms and wrote [`comparison.json`](nullspace_ablation_retry/comparison.json:1): valid true, active projection, non-empty frozen basis, both peaks below 7500 MB, delta BWT 0.36167033073388477 points, delta average accuracy 0.24463211986087696 points, positive old-task delta gaps, and continue false.
- Preserved the result-critic verdict in [`treatment_comparison_result_critic_packet.md`](treatment_comparison_result_critic_packet.md:1): valid feasibility evidence, no catastrophic-forgetting solution claim, block full current4/v10/Digital Twin escalation because the BWT improvement is below the +1.5 point threshold in [`EXPERIMENT_PROTOCOL.md`](../cf_cycle_2/EXPERIMENT_PROTOCOL.md:114).

## 3. Files changed or inspected

### Changed during CF-cycle-3

- [`cascades/data.py`](../../cascades/data.py:80) — added [`_build_supervised_sequence()`](../../cascades/data.py:80) response-preserving truncation and routed [`prepare_data()`](../../cascades/data.py:107) through it.
- [`tests/test_data.py`](../../tests/test_data.py:147) — added regression tests for all-ignored-label prevention and long-prompt response supervision.
- [`reasoning3_prepare_data_preflight.py`](reasoning3_prepare_data_preflight.py:1) — added reduced-memory data preflight.
- [`validate_control_gate.py`](validate_control_gate.py:1) — added control artifact validation gate.
- [`REPORT.md`](REPORT.md:1) — this durable closeout report.
- [`CONTEXT.md`](../../CONTEXT.md:1) — updated project memory and next-cycle seed.

### Evidence and handoff artifacts inspected or summarized

- [`root_cause_packet.md`](root_cause_packet.md:1) — confirmed batch-93 zero-supervision root cause.
- [`batch93_static_probe.json`](batch93_static_probe.json:1) — original all-ignored-label static diagnosis.
- [`batch93_static_probe.verify.json`](batch93_static_probe.verify.json:1) — fixed-loader verification scan.
- [`reasoning3_prepare_data_preflight.json`](reasoning3_prepare_data_preflight.json:1) — all reasoning3 tasks pass data-preflight gate.
- [`control_gate_validation.json`](control_gate_validation.json:1) — retried control passes gate.
- [`control_retry_result_critic_packet.md`](control_retry_result_critic_packet.md:1) — critic accepted the control gate.
- [`nullspace_ablation_retry/control/run_status.json`](nullspace_ablation_retry/control/run_status.json:1), [`nullspace_ablation_retry/control/metrics.json`](nullspace_ablation_retry/control/metrics.json:1), and [`nullspace_ablation_retry/control/instrumentation.json`](nullspace_ablation_retry/control/instrumentation.json:1) — completed control evidence.
- [`nullspace_ablation_retry/treatment/run_status.json`](nullspace_ablation_retry/treatment/run_status.json:1), [`nullspace_ablation_retry/treatment/metrics.json`](nullspace_ablation_retry/treatment/metrics.json:1), and [`nullspace_ablation_retry/treatment/instrumentation.json`](nullspace_ablation_retry/treatment/instrumentation.json:1) — completed active-treatment evidence.
- [`comparison.json`](nullspace_ablation_retry/comparison.json:1) — valid comparison with continue false.
- [`treatment_comparison_result_critic_packet.md`](treatment_comparison_result_critic_packet.md:1) — final critic verdict and CF-cycle-4 seed.

## 4. Commands/checks run

Report-writer note: this closeout did not rerun tests or GPU jobs. It records the confirmed command outcomes from the coordinator, debugger, code, experiment-designer, and critic handoffs.

- Regression test check: python module pytest over [`tests/test_data.py`](../../tests/test_data.py:1) with quiet mode; result was exit 0 with 14 data tests passing.
- Syntax check: python module bytecode compilation over [`cascades/data.py`](../../cascades/data.py:1) and [`batch93_probe.py`](batch93_probe.py:1); result was exit 0.
- Fixed static scan: [`batch93_probe.py`](batch93_probe.py:1) wrote [`batch93_static_probe.verify.json`](batch93_static_probe.verify.json:1); result showed batch 93 valid labels equal to 1 and zero zero-label batches.
- Reasoning3 data preflight: [`reasoning3_prepare_data_preflight.py`](reasoning3_prepare_data_preflight.py:1) wrote [`reasoning3_prepare_data_preflight.json`](reasoning3_prepare_data_preflight.json:1); result was valid true across all three reasoning3 tasks.
- Control-gate validation: python module bytecode compilation over [`reasoning3_prepare_data_preflight.py`](reasoning3_prepare_data_preflight.py:1) and [`validate_control_gate.py`](validate_control_gate.py:1), then [`validate_control_gate.py`](validate_control_gate.py:1) wrote [`control_gate_validation.json`](control_gate_validation.json:1); result was exit 0 and valid true.
- Corrected control launch: python [`run_nullspace_ablation.py`](../cf_cycle_1/run_nullspace_ablation.py:1) with control arm, reasoning3 suite, seed 42, 2 epochs, rank 4, max length 256, output root [`nullspace_ablation_retry`](nullspace_ablation_retry/), and 7500 MB threshold; output was [`control.log`](nullspace_ablation_retry/control.log:1), [`control/run_status.json`](nullspace_ablation_retry/control/run_status.json:1), and finite control metrics in [`control/metrics.json`](nullspace_ablation_retry/control/metrics.json:1).
- Treatment-only launch: python [`run_nullspace_ablation.py`](../cf_cycle_1/run_nullspace_ablation.py:1) with treatment arm, reasoning3 suite, seed 42, 2 epochs, rank 4, max length 256, output root [`nullspace_ablation_retry`](nullspace_ablation_retry/), and 7500 MB threshold; output was [`treatment.log`](nullspace_ablation_retry/treatment.log:1), [`treatment/run_status.json`](nullspace_ablation_retry/treatment/run_status.json:1), and finite treatment metrics in [`treatment/metrics.json`](nullspace_ablation_retry/treatment/metrics.json:1).
- Comparison check: python [`compare_nullspace_ablation.py`](../cf_cycle_1/compare_nullspace_ablation.py:1) over [`nullspace_ablation_retry`](nullspace_ablation_retry/) with a 7500 MB threshold; result was exit 0 and wrote [`comparison.json`](nullspace_ablation_retry/comparison.json:1).

## 5. Evidence summary

- Confirmed: the all-ignored-label failure was a data-preparation bug, not an immediate VRAM, optimizer, or null-space treatment result. [`root_cause_packet.md`](root_cause_packet.md:52) identifies prompt-first truncation as the confirmed mechanism.
- Confirmed: the response-preserving truncation fix prevents the reproduced zero-label condition. [`batch93_static_probe.verify.json`](batch93_static_probe.verify.json:6) reports batch 93 valid labels equal to 1, and [`reasoning3_prepare_data_preflight.json`](reasoning3_prepare_data_preflight.json:6) reports valid true.
- Confirmed: the corrected control completed under the guardrails. [`control/run_status.json`](nullspace_ablation_retry/control/run_status.json:3) reports completed status and [`control/run_status.json`](nullspace_ablation_retry/control/run_status.json:13) reports peak VRAM 6848.810546875 MB.
- Confirmed: the treatment was tested for seed 42 and completed under the guardrails. [`treatment/run_status.json`](nullspace_ablation_retry/treatment/run_status.json:3) reports completed status and [`treatment/run_status.json`](nullspace_ablation_retry/treatment/run_status.json:13) reports peak VRAM 6850.75390625 MB.
- Confirmed: treatment projection was active and the frozen basis was non-empty. [`comparison.json`](nullspace_ablation_retry/comparison.json:13) reports projection active true and [`comparison.json`](nullspace_ablation_retry/comparison.json:14) reports frozen basis non-empty true.
- Confirmed: treatment did not harm the reduced proxy metrics for seed 42. [`comparison.json`](nullspace_ablation_retry/comparison.json:4) reports delta BWT plus 0.36167033073388477 points, [`comparison.json`](nullspace_ablation_retry/comparison.json:5) reports delta average accuracy plus 0.24463211986087696 points, and [`comparison.json`](nullspace_ablation_retry/comparison.json:6) reports positive old-task delta gaps.
- Contradicted: seed-42 treatment did not meet the mitigation success threshold. [`EXPERIMENT_PROTOCOL.md`](../cf_cycle_2/EXPERIMENT_PROTOCOL.md:114) requires delta BWT at least +1.5 points for algorithmic success consideration, while [`comparison.json`](nullspace_ablation_retry/comparison.json:4) reports only +0.36167033073388477 points.
- Unsupported: catastrophic forgetting is not solved. [`treatment/metrics.json`](nullspace_ablation_retry/treatment/metrics.json:9) still reports negative absolute BWT, and the result is one seed on a 3-task reduced suite with proxy matrix metrics.
- Caveat: provenance is acceptable for the loop decision but not archival-grade reproducibility. The critic noted that run-status files capture git revision 7c4e01d but not dirty-tree state in [`treatment_comparison_result_critic_packet.md`](treatment_comparison_result_critic_packet.md:35).
- Skipped checks: no full current4 run, no v10 run, no Digital Twin task, no generative subset, and no additional seed were completed in CF-cycle-3.

## 6. Decision: retest

Retest once under the same reduced-memory reasoning3 protocol before larger runs.

Proceed with CF-cycle-4 as one more reduced-memory replication, not with full-suite escalation. The seed-42 treatment is valid feasibility evidence and a weak positive signal, but continue false in [`comparison.json`](nullspace_ablation_retry/comparison.json:15) blocks current4, v10, and Digital Twin escalation under the success criteria in [`EXPERIMENT_PROTOCOL.md`](../cf_cycle_2/EXPERIMENT_PROTOCOL.md:120).

Do not claim catastrophic forgetting is solved. Do not lower the success threshold after seeing the result.

## 7. Confidence: medium

- High confidence that the root cause and data-loader fix addressed the all-ignored-label failure: the static probe, regression tests, and preflight all point to the same mechanism.
- High confidence that seed-42 control and treatment are valid feasibility artifacts: both completed, metrics are finite, task parameters match, and VRAM stayed below threshold.
- High confidence that algorithmic escalation criteria were not met: delta BWT is below the predeclared +1.5 point threshold.
- Medium confidence that the current treatment configuration has a weak positive effect on the reduced proxy suite: all deltas moved in the favorable direction, but the effect is small.
- Low confidence that the seed-42 result generalizes: it is one seed, one short reduced suite, max length 256, 2 epochs, and proxy matrix metrics only.
- Unknown confidence for full catastrophic-forgetting mitigation on current4, v10, or Digital Twin tasks because those checks were intentionally not run after continue false.

## 8. Next-cycle seed objective

CF-cycle-4 should run one additional reduced-memory reasoning3 replication with seed 43, sequential control then treatment, using the same gates and avoiding the runner's both-arm launch mode.

Recommended CF-cycle-4 decision rule:

- If seed 43 reaches the predeclared delta-BWT threshold without accuracy or old-task regressions, send the result to critic before considering broader escalation.
- If seed 43 also stays below +1.5 BWT improvement or flips sign, pivot to a revised treatment-strength hypothesis before larger current4, v10, or Digital Twin runs.

## 9. Handoff to LLM Loop Coordinator

handoff_target: llm-loop-coordinator

continue_pivot_or_stop: continue, but retest once before escalation.

next_cycle_seed: CF-cycle-4 should run a seed-43 reduced-memory reasoning3 replication with sequential control then treatment, the same finite-loss, VRAM, projection, frozen-basis, and comparison gates, and no both-arm launch. If seed 43 misses the +1.5 BWT threshold, pivot to revised treatment-strength hypotheses instead of spending larger-run GPU time.

Coordinator routing recommendation: start CF-cycle-4 with LLM Research Planner or Experiment Designer depending on whether the coordinator wants a short seed-43 run plan first. The next executable work is bounded experiment execution, not another root-cause debugging pass.

Non-negotiable carry-forward: seed-42 treatment has been tested, but mitigation success threshold was not met; catastrophic forgetting is not solved.
