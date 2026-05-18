# CF-cycle-4 Report — seed-43 reduced-memory replication and treatment-strength pivot

Date: 2026-05-17

## 1. Cycle objective

Close CF-cycle-4 durably after retesting the active frozen null-space treatment on the reduced-memory reasoning3 proxy suite with seed 43, following seed 42's valid-but-below-threshold result in [`experiments/cf_cycle_3/REPORT.md`](../cf_cycle_3/REPORT.md:1).

Objective label: confirmed execution objective; contradicted algorithmic-success objective; unsupported catastrophic-forgetting solution claim.

## 2. What was done

- Wrote and followed the seed-43 protocol in [`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md:1): sequential control then treatment, no both-arm launch mode, reasoning3 task suite, seed 43, two epochs, rank 4, max length 256, and 7500 MB VRAM threshold.
- Added [`validate_seed43_arm_gate.py`](validate_seed43_arm_gate.py:1) to validate seed-43 artifacts for completed status, finite outputs, expected run parameters, expected task manifest, under-threshold VRAM, and active treatment projection evidence.
- Ran CPU, data, and GPU preflight. [`reasoning3_prepare_data_preflight.json`](nullspace_ablation_seed43/reasoning3_prepare_data_preflight.json:1) reports valid true, zero zero-label batches, and max sequence length 256. [`gpu_preflight.csv`](nullspace_ablation_seed43/gpu_preflight.csv:1) showed the RTX 4060 Ti idle enough to proceed, with 694 MiB used of 8188 MiB and 0 percent utilization.
- Completed the control arm under [`control/run_status.json`](nullspace_ablation_seed43/control/run_status.json:1): completed status, peak VRAM 6843.90966796875 MB, wall time 1497.7814581394196 seconds.
- Validated the control gate in [`control_gate.json`](nullspace_ablation_seed43/control_gate.json:1): valid true.
- Checked GPU before treatment in [`gpu_before_treatment.csv`](nullspace_ablation_seed43/gpu_before_treatment.csv:1): 707 MiB used of 8188 MiB and 1 percent utilization.
- Completed the treatment arm under [`treatment/run_status.json`](nullspace_ablation_seed43/treatment/run_status.json:1): completed status, peak VRAM 6843.90966796875 MB, wall time 1501.8400311470032 seconds.
- Validated active treatment execution in [`treatment_gate.json`](nullspace_ablation_seed43/treatment_gate.json:1): valid true, active projection, non-empty frozen basis, 3600 calls with frozen basis, and removed norm sum 3.100088352795865.
- Compared arms in [`comparison.json`](nullspace_ablation_seed43/comparison.json:1): valid true, projection active true, frozen basis non-empty true, both peaks under 7500 MB, delta BWT -0.008929480489536235 points, delta average accuracy -0.008046083930968173 points, old-task delta gaps -0.1564058367372001 and 0.13854687575812763 points, and continue false.
- Preserved the verdict-ready evidence in [`seed43_result_critic_packet.md`](seed43_result_critic_packet.md:1). The result-critic subtask was interrupted before returning, but the packet contains the necessary critic-ready evidence and decision boundary.

## 3. Files changed or inspected

### Changed or created during CF-cycle-4

- [`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md:1) — seed-43 protocol and gates.
- [`validate_seed43_arm_gate.py`](validate_seed43_arm_gate.py:1) — seed-43 control and treatment gate validator.
- [`seed43_result_critic_packet.md`](seed43_result_critic_packet.md:1) — evidence packet for the interrupted result-critic handoff.
- [`REPORT.md`](REPORT.md:1) — this durable closeout report.
- [`CONTEXT.md`](../../CONTEXT.md:1) — updated project memory and next-cycle seed.

### Evidence artifacts summarized

- [`reasoning3_prepare_data_preflight.json`](nullspace_ablation_seed43/reasoning3_prepare_data_preflight.json:1) — data preflight valid true, zero zero-label batches, max sequence length 256.
- [`gpu_preflight.csv`](nullspace_ablation_seed43/gpu_preflight.csv:1) — pre-control GPU availability.
- [`control/run_status.json`](nullspace_ablation_seed43/control/run_status.json:1), [`control/metrics.json`](nullspace_ablation_seed43/control/metrics.json:1), and [`control_gate.json`](nullspace_ablation_seed43/control_gate.json:1) — completed valid control evidence.
- [`gpu_before_treatment.csv`](nullspace_ablation_seed43/gpu_before_treatment.csv:1) — pre-treatment GPU availability.
- [`treatment/run_status.json`](nullspace_ablation_seed43/treatment/run_status.json:1), [`treatment/metrics.json`](nullspace_ablation_seed43/treatment/metrics.json:1), and [`treatment_gate.json`](nullspace_ablation_seed43/treatment_gate.json:1) — completed valid active-treatment evidence.
- [`comparison.json`](nullspace_ablation_seed43/comparison.json:1) — valid comparison with continue false.
- [`experiments/cf_cycle_3/REPORT.md`](../cf_cycle_3/REPORT.md:1) — seed-42 context and prior below-threshold result.

## 4. Commands/checks run

Report-writer note: this closeout did not rerun tests or GPU jobs. It records the confirmed command outcomes from the seed-43 experiment packet in [`seed43_result_critic_packet.md`](seed43_result_critic_packet.md:29).

- Isolation check: Git directory, common directory, branch, and superproject status were checked. Observed checkout was a normal repository checkout on master, not an isolated worktree. No manual worktree was created because this was an in-place loop artifact run.
- CPU/data/GPU preflight: bytecode compilation over [`train.py`](../../train.py:1), [`run_nullspace_ablation.py`](../cf_cycle_1/run_nullspace_ablation.py:1), [`compare_nullspace_ablation.py`](../cf_cycle_1/compare_nullspace_ablation.py:1), [`reasoning3_prepare_data_preflight.py`](../cf_cycle_3/reasoning3_prepare_data_preflight.py:1), and [`validate_seed43_arm_gate.py`](validate_seed43_arm_gate.py:1); pytest over [`tests/test_data.py`](../../tests/test_data.py:1) and [`tests/test_cf_cycle_2_guardrails.py`](../../tests/test_cf_cycle_2_guardrails.py:1); seed-43 data preflight; and GPU snapshot. Result: exit code 0, 19 tests passed, data preflight valid true, GPU idle enough.
- Control launch: [`run_nullspace_ablation.py`](../cf_cycle_1/run_nullspace_ablation.py:1) was run with control arm, reasoning3 suite, seed 43, two epochs, rank 4, max length 256, output root [`nullspace_ablation_seed43`](nullspace_ablation_seed43/), and 7500 MB threshold. Result: [`control/run_status.json`](nullspace_ablation_seed43/control/run_status.json:1) completed with peak VRAM 6843.90966796875 MB.
- Control gate: [`validate_seed43_arm_gate.py`](validate_seed43_arm_gate.py:1) was run against the control arm. Result: [`control_gate.json`](nullspace_ablation_seed43/control_gate.json:1) valid true.
- Pre-treatment GPU check: [`gpu_before_treatment.csv`](nullspace_ablation_seed43/gpu_before_treatment.csv:1) recorded 707 MiB used of 8188 MiB and 1 percent utilization.
- Treatment launch: [`run_nullspace_ablation.py`](../cf_cycle_1/run_nullspace_ablation.py:1) was run with treatment arm, reasoning3 suite, seed 43, two epochs, rank 4, max length 256, output root [`nullspace_ablation_seed43`](nullspace_ablation_seed43/), and 7500 MB threshold. Result: [`treatment/run_status.json`](nullspace_ablation_seed43/treatment/run_status.json:1) completed with peak VRAM 6843.90966796875 MB.
- Treatment gate and comparison: [`validate_seed43_arm_gate.py`](validate_seed43_arm_gate.py:1) was run against the treatment arm, then [`compare_nullspace_ablation.py`](../cf_cycle_1/compare_nullspace_ablation.py:1) was run over [`nullspace_ablation_seed43`](nullspace_ablation_seed43/). Result: [`treatment_gate.json`](nullspace_ablation_seed43/treatment_gate.json:1) valid true with active projection evidence, and [`comparison.json`](nullspace_ablation_seed43/comparison.json:1) valid true with continue false.

## 5. Evidence summary

- Confirmed: preflight was clean enough for the bounded seed-43 run. [`reasoning3_prepare_data_preflight.json`](nullspace_ablation_seed43/reasoning3_prepare_data_preflight.json:1) reports valid true and zero zero-label batches; [`gpu_preflight.csv`](nullspace_ablation_seed43/gpu_preflight.csv:1) reports low GPU memory use and zero utilization before launch.
- Confirmed: control and treatment both completed under the 7500 MB VRAM threshold. [`control/run_status.json`](nullspace_ablation_seed43/control/run_status.json:1) and [`treatment/run_status.json`](nullspace_ablation_seed43/treatment/run_status.json:1) both report completed status and peak VRAM 6843.90966796875 MB.
- Confirmed: treatment projection was actually active. [`treatment_gate.json`](nullspace_ablation_seed43/treatment_gate.json:1) reports active projection, non-empty frozen basis, 3600 calls with frozen basis, and removed norm sum 3.100088352795865.
- Confirmed: control metrics were finite and valid. [`control/metrics.json`](nullspace_ablation_seed43/control/metrics.json:1) reports average accuracy 0.5665482637215719, BWT -0.026559348750060024, and old-task deltas -0.04616356831829843 and -0.006955129181821618.
- Confirmed: treatment metrics were finite and valid. [`treatment/metrics.json`](nullspace_ablation_seed43/treatment/metrics.json:1) reports average accuracy 0.5664678028822622, BWT -0.026648643554955387, and old-task deltas -0.04772762668567043 and -0.005569660424240341.
- Contradicted: algorithmic success and escalation. [`comparison.json`](nullspace_ablation_seed43/comparison.json:1) reports delta BWT -0.008929480489536235 points, which misses the plus 1.5 point threshold; it also reports continue false.
- Likely: the current reduced-memory active frozen null-space treatment effect is weak or mixed across two seeds. Seed 42 in [`experiments/cf_cycle_3/REPORT.md`](../cf_cycle_3/REPORT.md:1) was valid but below threshold with a small positive BWT delta; seed 43 was valid but slightly negative.
- Unsupported: catastrophic forgetting is not solved. The seed-43 treatment still has negative absolute BWT in [`treatment/metrics.json`](nullspace_ablation_seed43/treatment/metrics.json:1), and no full current4, v10, Digital Twin task, or generative subset evaluation was run.
- Skipped or blocked checks: no full current4 run, no v10 run, no Digital Twin task, no generative subset evaluation, and no additional seeds beyond seed 43 in this cycle. Larger-run escalation remains blocked by continue false in [`comparison.json`](nullspace_ablation_seed43/comparison.json:1).

## 6. Decision: revise

Revise the treatment before spending larger-run GPU time.

Feasibility and active-projection execution are confirmed at high confidence, but algorithmic success is contradicted for seed 43 and remains below threshold across seed 42 and seed 43 together. CF-cycle-4 should therefore close with a pivot to revised treatment-strength hypotheses, not proceed to current4, v10, or Digital Twin escalation.

Do not claim catastrophic forgetting is solved. Do not lower the success threshold after seeing the result.

## 7. Confidence: medium

- High confidence for feasibility: preflight, control, treatment, gate validator outputs, and comparison all completed and point to valid under-threshold execution.
- High confidence that the active-projection condition was met: [`treatment_gate.json`](nullspace_ablation_seed43/treatment_gate.json:1) directly records active projection, non-empty frozen basis, repeated calls with frozen basis, and positive removed-norm sum.
- High confidence that seed 43 missed the predeclared BWT threshold: [`comparison.json`](nullspace_ablation_seed43/comparison.json:1) reports a negative delta BWT and continue false.
- Medium confidence that the treatment effect is weak or mixed: two reduced-memory seeds are enough to block escalation, but not enough to characterize all possible treatment variants.
- Low confidence that the current result generalizes to full current4, v10, or Digital Twin settings: those runs were intentionally not performed after the reduced proxy failed the gate.
- Unknown confidence for revised treatment-strength variants because they have not been planned or tested yet.

## 8. Next-cycle seed objective

CF-cycle-5 should pivot to treatment-strength redesign before any larger GPU run.

Recommended seed objective: produce ranked, falsifiable revised-treatment hypotheses that explain why active frozen null-space execution is feasible but below threshold across seed 42 and seed 43, then select the cheapest bounded experiment to test a stronger or less-confounded variant.

Candidate hypothesis areas for the next planner or hypothesis generator:

- Isolate or enable CL-LoRA reassignment rather than leaving the current treatment under-strength.
- Freeze earlier or use a stricter freeze criterion to preserve old-task directions sooner.
- Disable D-MoLE migration as a confound and test null-space behavior without migration interference.
- Adjust frozen-basis capacity, threshold, or update cadence to increase useful removed-gradient mass.
- Compare the default v10 or full null-space stack only after the revised reduced proxy hypothesis is explicit and bounded.

## 9. Handoff to LLM Loop Coordinator

handoff_target: llm-loop-coordinator

continue_pivot_or_stop: pivot. Continue the loop, but pivot away from direct larger-run escalation and toward treatment-strength hypothesis redesign.

next_cycle_seed: CF-cycle-5 should start with LLM Research Planner or LLM Hypothesis Generator to redesign the treatment-strength hypothesis set. The next cycle should not launch current4, v10, or Digital Twin GPU runs until it has a revised, bounded hypothesis and gate for why the treatment should now exceed the BWT threshold.

Coordinator routing recommendation: send CF-cycle-5 to LLM Research Planner if the loop wants a structured redesign packet first, or to LLM Hypothesis Generator if the coordinator accepts the current evidence summary and wants candidate interventions immediately.

Non-negotiable carry-forward: seed 43 confirms feasibility and active projection, but algorithmic success is contradicted; catastrophic forgetting is not solved.
