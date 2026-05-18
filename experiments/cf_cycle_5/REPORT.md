# CF-cycle-5 Report — CL-LoRA-active reduced reasoning3 treatment closeout

Date: 2026-05-17

## 1. Cycle objective

Close CF-cycle-5 durably after the CL-LoRA-active reduced reasoning3 treatment-only run, preserving what the cycle proved and what it did not prove.

Objective label: confirmed valid execution and active reassignment exercise; weak positive copied-control screen; below success threshold; unsupported catastrophic-forgetting solution claim.

## 2. What was done

- Followed the bounded treatment-only path from [`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md:175): copied the valid seed-43 control from [`experiments/cf_cycle_4/nullspace_ablation_seed43/control`](../cf_cycle_4/nullspace_ablation_seed43/control) into [`cllora_active_seed43/control`](cllora_active_seed43/control), then ran only the CL-LoRA-active treatment arm under reduced reasoning3, seed 43, rank 4, max length 256, two epochs, and a 7500 MB VRAM guardrail.
- Validated the completed treatment with the standard seed-43 arm gate in [`treatment_gate.json`](cllora_active_seed43/treatment_gate.json:1).
- Validated the active-treatment-specific conditions with [`treatment_active_gate.json`](cllora_active_seed43/treatment_active_gate.json:1), including active reassignment serialization, active reassignment path execution, finite active adjustment norm, frozen projection activity, and normalized frozen removal evidence.
- Compared the treatment against the copied seed-43 control in [`comparison.json`](cllora_active_seed43/comparison.json:1).
- Preserved the result-critic conclusion supplied for this closeout: valid execution; weak positive signal for the bounded copied-control screen; below threshold for success; revise toward Rank 2/3 hypotheses; no escalation and no catastrophic-forgetting solution claim.
- Wrote this durable closeout report and updated [`CONTEXT.md`](../../CONTEXT.md:1). No code was modified in this report-writer step.

## 3. Files changed or inspected

### Changed during report writer closeout

- [`REPORT.md`](REPORT.md:1) — durable CF-cycle-5 closeout report.
- [`CONTEXT.md`](../../CONTEXT.md:1) — project memory updated with the closeout decision and next-cycle seed.

### Evidence files inspected or summarized

- [`cllora_active_result_critic_packet.md`](cllora_active_result_critic_packet.md:1) — treatment-only evidence packet and command record.
- [`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md:1) — predeclared CL-LoRA-active protocol, success gate, and verdict labels.
- [`comparison.json`](cllora_active_seed43/comparison.json:1) — valid copied-control comparison with weak positive BWT/average-accuracy deltas but continue false.
- [`treatment_active_gate.json`](cllora_active_seed43/treatment_active_gate.json:1) — active reassignment and active-treatment config gate.
- [`treatment_gate.json`](cllora_active_seed43/treatment_gate.json:1) — standard treatment artifact and projection gate.
- [`treatment/run_status.json`](cllora_active_seed43/treatment/run_status.json:1) — treatment completion and VRAM status.
- [`treatment/instrumentation.json`](cllora_active_seed43/treatment/instrumentation.json:674) — frozen projection and active reassignment counters.
- [`hypothesis_handoff.md`](hypothesis_handoff.md:30) — Rank 2/3 follow-up hypotheses after Rank 1.

## 4. Commands/checks run

Report-writer note: this closeout did not run GPU jobs, tests, or code commands. It records the successful command outcomes from [`cllora_active_result_critic_packet.md`](cllora_active_result_critic_packet.md:18).

Recorded CF-cycle-5 execution commands and checks:

1. mkdir/copy root command: if not exist experiments\cf_cycle_5\cllora_active_seed43 mkdir experiments\cf_cycle_5\cllora_active_seed43 && xcopy /E /I /Y experiments\cf_cycle_4\nullspace_ablation_seed43\control experiments\cf_cycle_5\cllora_active_seed43\control
   - Exit 0; root already existed.
2. Explicit control copy command: xcopy /E /I /Y experiments\cf_cycle_4\nullspace_ablation_seed43\control experiments\cf_cycle_5\cllora_active_seed43\control
   - Exit 0; reported 8 files copied.
3. Treatment launch command: python experiments\cf_cycle_1\run_nullspace_ablation.py --arm treatment --task-suite reasoning3 --seed 43 --epochs 2 --rank 4 --max-length 256 --output-root experiments\cf_cycle_5\cllora_active_seed43 --treatment-variant cllora-active --vram-threshold-mb 7500 > experiments\cf_cycle_5\cllora_active_seed43\treatment.log 2>&1
   - Completed and wrote [`treatment/run_status.json`](cllora_active_seed43/treatment/run_status.json:1), [`treatment/metrics.json`](cllora_active_seed43/treatment/metrics.json:1), [`treatment/instrumentation.json`](cllora_active_seed43/treatment/instrumentation.json:1), and [`treatment/accuracy_matrix.npy`](cllora_active_seed43/treatment/accuracy_matrix.npy).
4. Standard gate command: python experiments\cf_cycle_4\validate_seed43_arm_gate.py --root experiments\cf_cycle_5\cllora_active_seed43 --arm treatment --vram-threshold-mb 7500 > experiments\cf_cycle_5\cllora_active_seed43\treatment_gate.json
   - Exit 0; wrote [`treatment_gate.json`](cllora_active_seed43/treatment_gate.json:1).
5. Active-treatment gate command: python experiments\cf_cycle_5\validate_active_treatment_gate.py --root experiments\cf_cycle_5\cllora_active_seed43 --arm treatment --out experiments\cf_cycle_5\cllora_active_seed43\treatment_active_gate.json
   - Exit 0; wrote [`treatment_active_gate.json`](cllora_active_seed43/treatment_active_gate.json:1).
6. Comparison command: python experiments\cf_cycle_1\compare_nullspace_ablation.py --root experiments\cf_cycle_5\cllora_active_seed43 --vram-threshold-mb 7500 > experiments\cf_cycle_5\cllora_active_seed43\comparison.json
   - Exit 0; wrote [`comparison.json`](cllora_active_seed43/comparison.json:1).

Skipped checks in this closeout: no paired-control retry, no additional seeds, no full current4, no v10 run, no Digital Twin run, no generative subset, and no broad settings fuzzer. These were out of scope or blocked by the below-threshold result.

## 5. Evidence summary

### Current evidence state

- Confirmed: the CL-LoRA-active treatment executed validly. [`treatment/run_status.json`](cllora_active_seed43/treatment/run_status.json:1) reports completed status, treatment variant cllora-active, seed 43, rank 4, max length 256, two epochs, and peak VRAM 6843.90966796875 MB under the 7500 MB threshold.
- Confirmed: the standard treatment gate passed. [`treatment_gate.json`](cllora_active_seed43/treatment_gate.json:1) reports valid true, under-threshold VRAM true, no finite metric or instrumentation bad paths, finite 3x3 accuracy matrix, non-empty frozen basis, 3600 calls with frozen basis, removed norm sum 2.195456126228237, projection active true, and freeze event count 38.
- Confirmed: the active reassignment path was exercised. [`treatment_active_gate.json`](cllora_active_seed43/treatment_active_gate.json:1) reports valid true; active-treatment config checks true; active instrumentation checks true; active adjustment positive true; 6900 calls with active reassignment enabled; 6900 calls with active reassignment path; active adjustment norm sum 49.50860421800462; active adjustment norm max 0.12196692824363708.
- Confirmed: frozen projection remained active alongside active reassignment. [`treatment/instrumentation.json`](cllora_active_seed43/treatment/instrumentation.json:674) reports 6900 total reassignment calls, 3600 calls with frozen basis, removed norm sum 2.195456126228237, and removed norm per frozen call 0.000609848923952288.
- Weak signal: the copied-control comparison moved in the right direction on BWT and average accuracy, but not enough. [`comparison.json`](cllora_active_seed43/comparison.json:1) reports valid true, delta BWT 0.03391994701978929, delta average accuracy 0.09627341372566711, and continue false.
- Mixed signal: old-task gaps split positive and negative. [`comparison.json`](cllora_active_seed43/comparison.json:6) reports old-task delta gaps 0.6678861145674175 and -0.600046220527839.
- Contradicted: success under the predeclared threshold. [`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md:234) required delta BWT at least 1.5, delta average accuracy at least -2.0, minimum old-task gap at least -3.0, and continue true. The observed delta BWT was only 0.03391994701978929 and continue was false.
- Blocked: full current4, v10, Digital Twin, generative subset, and broad settings-fuzzer escalation remain blocked until a revised reduced-suite treatment gives stronger evidence.
- Non-claim: catastrophic forgetting is not solved. This cycle only establishes valid CL-LoRA-active execution with a weak positive copied-control signal below the success threshold.

### Result critic verdict preserved

Result label: weak positive for the bounded copied-control screen; below threshold for success.

The result critic accepted execution validity and active reassignment evidence, but rejected escalation because the comparison remained far below the plus 1.5 BWT-point threshold and continue was false. The critic’s decision is revise, not proceed, not immediate paired retry, and not a catastrophic-forgetting solution claim.

## 6. Decision: revise

Revise the treatment design before spending larger-run GPU time.

The CF-cycle-5 Rank 1 hypothesis was useful because it removed a major ambiguity: active CL-LoRA reassignment can be enabled, serialized, instrumented, and run under the RTX 4060 Ti 8GB reduced reasoning3 guardrail. However, its measured improvement was too small to justify escalation. The next cycle should move to Rank 2/3 reduced-suite treatment redesign unless the coordinator explicitly chooses a narrow comparability cleanup first.

Do not claim catastrophic forgetting is solved. Do not lower the success threshold after seeing the result. Do not launch current4, v10, Digital Twin, or a broad fuzzer from this evidence.

## 7. Confidence: medium

- High confidence for valid execution: run status, standard gate, active-treatment gate, and comparison all exist and report valid or completed states.
- High confidence that active reassignment was actually exercised: active path calls are 6900 and active adjustment norm sum is positive.
- High confidence that the result missed the predeclared success threshold: delta BWT is 0.03391994701978929 and continue is false.
- Medium confidence in the algorithmic interpretation: the signal is from one reduced-suite seed and a copied-control comparison, so it is enough to block escalation and guide redesign, not enough to characterize all treatment variants.
- Low confidence that the result generalizes to full current4, v10, or Digital Twin settings because those runs were intentionally not performed.

## 8. Next-cycle seed objective

Seed the next coordinator cycle toward Rank 2 first, with Rank 3 kept as the explicit alternate if capacity/cadence analysis is less promising.

Recommended next-cycle seed objective: design a bounded reduced reasoning3 treatment redesign that adjusts frozen-basis capacity, threshold, or update cadence while preserving the validated CL-LoRA-active instrumentation and success gate; include a short decision checkpoint on whether freeze timing relative to D-MoLE migration should supersede that Rank 2 experiment.

Why this seed: Rank 1 proved active reassignment execution but only produced a weak positive signal. The highest-value next uncertainty is whether the frozen basis is too small, too conservatively thresholded, or updated at the wrong cadence before changing loop ordering. Freeze timing relative to D-MoLE migration is the next alternate because it is plausible but more invasive.

## 9. Handoff to LLM Loop Coordinator

handoff_target: llm-loop-coordinator

continue_pivot_or_stop: pivot. Continue the loop, but pivot away from immediate paired retry or larger-run escalation and toward Rank 2/3 reduced-suite treatment redesign.

next_cycle_seed: Ask LLM Research Planner to produce a concrete redesign packet for frozen-basis capacity, threshold, or update cadence under reduced reasoning3, seed 43 or a similarly bounded seed plan, with the validated CL-LoRA-active gate retained. The planner should also decide whether freeze timing relative to D-MoLE migration should become the experiment-design target instead of capacity/cadence.

Coordinator guardrails:

- Preserve confirmed evidence: CL-LoRA-active treatment executed validly and active reassignment was exercised.
- Preserve the critic conclusion: weak positive copied-control signal, below threshold, revise.
- Do not claim catastrophic forgetting is solved.
- Do not launch full current4, v10, Digital Twin, generative subset, or broad settings fuzzer without a new reduced-suite gate.
- Do not paired-retry immediately unless the coordinator explicitly prioritizes copied-control comparability cleanup over Rank 2/3 redesign.
