# CF-cycle-5 CL-LoRA-active treatment-only evidence packet

handoff_target: llm-result-critic

## 1. Hypothesis tested

The bounded test probed the CF-cycle-5 top hypothesis from [`hypothesis_handoff.md`](experiments/cf_cycle_5/hypothesis_handoff.md:47): prior reduced-suite frozen-only treatment may have been too weak because active CL-LoRA reassignment and soft EAR were disabled. The treatment should be artifact-distinguishable from prior frozen-only evidence by enabling COSO nullspace projection, CL-LoRA reassignment, soft EAR, and EAR gamma 1e-4 in [`config.json`](experiments/cf_cycle_5/cllora_active_seed43/treatment/config.json:15).

## 2. Experiment performed

Executed only the approved treatment-only path in [`EXPERIMENT_PROTOCOL.md`](experiments/cf_cycle_5/EXPERIMENT_PROTOCOL.md:175):

- Copied provisional seed-43 control from [`control`](experiments/cf_cycle_4/nullspace_ablation_seed43/control) to [`control`](experiments/cf_cycle_5/cllora_active_seed43/control).
- Ran one CL-LoRA-active treatment arm on reduced reasoning3, seed 43, rank 4, max length 256, two epochs, 7500 MB VRAM threshold.
- Ran the standard arm gate, the active-treatment gate, and the comparison against the copied seed-43 control.
- Stopped after treatment, gates, and comparison; no paired fallback or larger run was launched.

## 3. Commands or actions

Observed successful command exits unless otherwise noted.

1. mkdir/copy root command: if not exist experiments\cf_cycle_5\cllora_active_seed43 mkdir experiments\cf_cycle_5\cllora_active_seed43 && xcopy /E /I /Y experiments\cf_cycle_4\nullspace_ablation_seed43\control experiments\cf_cycle_5\cllora_active_seed43\control
   - Exit 0, no output because the root already existed.
2. Explicit control copy command: xcopy /E /I /Y experiments\cf_cycle_4\nullspace_ablation_seed43\control experiments\cf_cycle_5\cllora_active_seed43\control
   - Exit 0, output reported 8 files copied.
3. Treatment launch command: python experiments\cf_cycle_1\run_nullspace_ablation.py --arm treatment --task-suite reasoning3 --seed 43 --epochs 2 --rank 4 --max-length 256 --output-root experiments\cf_cycle_5\cllora_active_seed43 --treatment-variant cllora-active --vram-threshold-mb 7500 > experiments\cf_cycle_5\cllora_active_seed43\treatment.log 2>&1
   - Completed and wrote [`run_status.json`](experiments/cf_cycle_5/cllora_active_seed43/treatment/run_status.json:1), [`metrics.json`](experiments/cf_cycle_5/cllora_active_seed43/treatment/metrics.json:1), [`instrumentation.json`](experiments/cf_cycle_5/cllora_active_seed43/treatment/instrumentation.json:1), and [`accuracy_matrix.npy`](experiments/cf_cycle_5/cllora_active_seed43/treatment/accuracy_matrix.npy).
4. Standard gate command: python experiments\cf_cycle_4\validate_seed43_arm_gate.py --root experiments\cf_cycle_5\cllora_active_seed43 --arm treatment --vram-threshold-mb 7500 > experiments\cf_cycle_5\cllora_active_seed43\treatment_gate.json
   - Exit 0, wrote [`treatment_gate.json`](experiments/cf_cycle_5/cllora_active_seed43/treatment_gate.json:1).
5. Active-treatment gate command: python experiments\cf_cycle_5\validate_active_treatment_gate.py --root experiments\cf_cycle_5\cllora_active_seed43 --arm treatment --out experiments\cf_cycle_5\cllora_active_seed43\treatment_active_gate.json
   - Exit 0, wrote [`treatment_active_gate.json`](experiments/cf_cycle_5/cllora_active_seed43/treatment_active_gate.json:1), and printed the same valid JSON to stdout.
6. Comparison command: python experiments\cf_cycle_1\compare_nullspace_ablation.py --root experiments\cf_cycle_5\cllora_active_seed43 --vram-threshold-mb 7500 > experiments\cf_cycle_5\cllora_active_seed43\comparison.json
   - Exit 0, wrote [`comparison.json`](experiments/cf_cycle_5/cllora_active_seed43/comparison.json:1).

## 4. Files changed or inspected

Changed:

- Added this packet: [`cllora_active_result_critic_packet.md`](experiments/cf_cycle_5/cllora_active_result_critic_packet.md:1).
- Updated project context in [`CONTEXT.md`](CONTEXT.md:17) with the bounded treatment-only run summary and decision boundary.
- Created or refreshed the experiment output root under [`cllora_active_seed43`](experiments/cf_cycle_5/cllora_active_seed43).

Inspected or used as evidence:

- Protocol: [`EXPERIMENT_PROTOCOL.md`](experiments/cf_cycle_5/EXPERIMENT_PROTOCOL.md:175).
- Treatment config: [`config.json`](experiments/cf_cycle_5/cllora_active_seed43/treatment/config.json:1).
- Treatment status: [`run_status.json`](experiments/cf_cycle_5/cllora_active_seed43/treatment/run_status.json:1).
- Treatment metrics: [`metrics.json`](experiments/cf_cycle_5/cllora_active_seed43/treatment/metrics.json:1).
- Treatment instrumentation: [`instrumentation.json`](experiments/cf_cycle_5/cllora_active_seed43/treatment/instrumentation.json:674).
- Standard gate: [`treatment_gate.json`](experiments/cf_cycle_5/cllora_active_seed43/treatment_gate.json:1).
- Active-treatment gate: [`treatment_active_gate.json`](experiments/cf_cycle_5/cllora_active_seed43/treatment_active_gate.json:1).
- Comparison: [`comparison.json`](experiments/cf_cycle_5/cllora_active_seed43/comparison.json:1).

## 5. Observed facts

### Treatment config and status

- [`config.json`](experiments/cf_cycle_5/cllora_active_seed43/treatment/config.json:14) reports treatment variant cllora-active.
- [`config.json`](experiments/cf_cycle_5/cllora_active_seed43/treatment/config.json:19) reports COSO nullspace enabled.
- [`config.json`](experiments/cf_cycle_5/cllora_active_seed43/treatment/config.json:20) reports CL-LoRA reassignment enabled.
- [`config.json`](experiments/cf_cycle_5/cllora_active_seed43/treatment/config.json:25) reports EAR gamma 0.0001.
- [`config.json`](experiments/cf_cycle_5/cllora_active_seed43/treatment/config.json:26) reports soft EAR enabled.
- [`run_status.json`](experiments/cf_cycle_5/cllora_active_seed43/treatment/run_status.json:3) reports completed status.
- [`run_status.json`](experiments/cf_cycle_5/cllora_active_seed43/treatment/run_status.json:14) reports peak VRAM 6843.90966796875 MB, below the 7500 MB threshold.
- [`run_status.json`](experiments/cf_cycle_5/cllora_active_seed43/treatment/run_status.json:15) reports wall time 1126.0758669376373 seconds.

### Treatment metrics

- [`metrics.json`](experiments/cf_cycle_5/cllora_active_seed43/treatment/metrics.json:8) reports average accuracy 0.5675109978588285.
- [`metrics.json`](experiments/cf_cycle_5/cllora_active_seed43/treatment/metrics.json:9) reports BWT -0.02622014927986213.
- [`metrics.json`](experiments/cf_cycle_5/cllora_active_seed43/treatment/metrics.json:10) reports final accuracies [0.39242165663305134, 0.5440763788096135, 0.7660349581338208].
- [`metrics.json`](experiments/cf_cycle_5/cllora_active_seed43/treatment/metrics.json:20) reports old-task deltas [-0.039484707172624256, -0.012955591387100007].

### Projection and active reassignment instrumentation

- [`instrumentation.json`](experiments/cf_cycle_5/cllora_active_seed43/treatment/instrumentation.json:674) records reassignment instrumentation.
- [`instrumentation.json`](experiments/cf_cycle_5/cllora_active_seed43/treatment/instrumentation.json:675) reports 6900 total reassignment calls.
- [`instrumentation.json`](experiments/cf_cycle_5/cllora_active_seed43/treatment/instrumentation.json:677) reports 3600 calls with frozen basis.
- [`instrumentation.json`](experiments/cf_cycle_5/cllora_active_seed43/treatment/instrumentation.json:679) reports removed norm sum 2.195456126228237.
- [`instrumentation.json`](experiments/cf_cycle_5/cllora_active_seed43/treatment/instrumentation.json:680) reports removed norm per frozen call 0.000609848923952288.
- [`instrumentation.json`](experiments/cf_cycle_5/cllora_active_seed43/treatment/instrumentation.json:681) reports 6900 calls with active reassignment enabled.
- [`instrumentation.json`](experiments/cf_cycle_5/cllora_active_seed43/treatment/instrumentation.json:682) reports 6900 calls with active reassignment path.
- [`instrumentation.json`](experiments/cf_cycle_5/cllora_active_seed43/treatment/instrumentation.json:683) reports active adjustment norm sum 49.50860421800462.
- [`instrumentation.json`](experiments/cf_cycle_5/cllora_active_seed43/treatment/instrumentation.json:684) reports active adjustment norm max 0.12196692824363708.

### Gates

- [`treatment_gate.json`](experiments/cf_cycle_5/cllora_active_seed43/treatment_gate.json:2) reports valid true.
- [`treatment_gate.json`](experiments/cf_cycle_5/cllora_active_seed43/treatment_gate.json:44) reports under VRAM threshold true.
- [`treatment_gate.json`](experiments/cf_cycle_5/cllora_active_seed43/treatment_gate.json:45) and [`treatment_gate.json`](experiments/cf_cycle_5/cllora_active_seed43/treatment_gate.json:46) report no finite metric or instrumentation bad paths.
- [`treatment_gate.json`](experiments/cf_cycle_5/cllora_active_seed43/treatment_gate.json:47) reports finite accuracy matrix true.
- [`treatment_gate.json`](experiments/cf_cycle_5/cllora_active_seed43/treatment_gate.json:52) reports frozen basis nonempty, calls with frozen basis 3600, removed norm sum 2.195456126228237, projection active true, and freeze event count 38.
- [`treatment_active_gate.json`](experiments/cf_cycle_5/cllora_active_seed43/treatment_active_gate.json:2) reports valid true.
- [`treatment_active_gate.json`](experiments/cf_cycle_5/cllora_active_seed43/treatment_active_gate.json:6) reports all required active-treatment config checks true.
- [`treatment_active_gate.json`](experiments/cf_cycle_5/cllora_active_seed43/treatment_active_gate.json:14) reports all required instrumentation checks true.
- [`treatment_active_gate.json`](experiments/cf_cycle_5/cllora_active_seed43/treatment_active_gate.json:21) reports active adjustment positive true.

### Comparison

- [`comparison.json`](experiments/cf_cycle_5/cllora_active_seed43/comparison.json:2) reports valid true with no failures.
- [`comparison.json`](experiments/cf_cycle_5/cllora_active_seed43/comparison.json:4) reports delta BWT 0.03391994701978929.
- [`comparison.json`](experiments/cf_cycle_5/cllora_active_seed43/comparison.json:5) reports delta average accuracy 0.09627341372566711.
- [`comparison.json`](experiments/cf_cycle_5/cllora_active_seed43/comparison.json:6) reports old-task delta gaps [0.6678861145674175, -0.600046220527839].
- [`comparison.json`](experiments/cf_cycle_5/cllora_active_seed43/comparison.json:10) and [`comparison.json`](experiments/cf_cycle_5/cllora_active_seed43/comparison.json:11) report control and treatment peak VRAM 6843.90966796875 MB.
- [`comparison.json`](experiments/cf_cycle_5/cllora_active_seed43/comparison.json:13) reports projection active true.
- [`comparison.json`](experiments/cf_cycle_5/cllora_active_seed43/comparison.json:14) reports frozen basis nonempty true.
- [`comparison.json`](experiments/cf_cycle_5/cllora_active_seed43/comparison.json:15) reports continue false.

## 6. Interpretations separated from facts

- Confirmed: the active-treatment configuration was serialized and gate-validated.
- Confirmed: the active reassignment path executed, not merely frozen-only projection.
- Confirmed: the treatment run completed under the VRAM guardrail with finite metrics and finite instrumentation according to gates.
- Weak signal: compared with the copied seed-43 control, the active treatment improved delta BWT and average accuracy, but the BWT lift is small and the comparison still reports continue false.
- Mixed signal: old-task delta gaps are split, with one positive and one negative gap.
- Not supported: a claim that catastrophic forgetting is solved.

## 7. Checks not run and why

- Paired fallback was not run because the approved instruction explicitly said not to launch it unless the treatment-only path was blocked and to return for coordination instead.
- Full current4, v10, Digital Twin, generative subset, and broad fuzzer runs were not run because they were explicitly out of scope for this bounded treatment-only experiment.
- Additional seeds were not run because this task approved seed 43 only.

## 8. Hypothesis status map

- H1: Active CL-LoRA reassignment path can be enabled and measured under reduced reasoning3 seed 43: confirmed.
- H2: Active CL-LoRA treatment improves the copied-control comparison enough to justify escalation: weak signal, not sufficient under the current continue gate.
- H3: Treatment resolves catastrophic forgetting: not tested and not supported.

## 9. Result label and confidence

- Result label: weak signal for algorithmic improvement; confirmed for execution/gate validity.
- Confidence: medium for the bounded decision. Confidence is limited by one seed, reduced reasoning3 proxy scope, copied-control comparability, and mixed old-task gaps.

## 10. Recommended next action

Hand off to llm-result-critic to judge whether the copied-control comparison is acceptable and whether delta BWT 0.03391994701978929 with continue false warrants any paired-control retry. If the critic rejects escalation, the next design should revise treatment strength or isolate confounds rather than running full current4, v10, or Digital Twin.
