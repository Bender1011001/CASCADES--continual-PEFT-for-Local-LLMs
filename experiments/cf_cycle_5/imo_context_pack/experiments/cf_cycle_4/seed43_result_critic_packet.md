# CF-cycle-4 Seed-43 Result Critic Packet

handoff_target: llm-result-critic

## 1. Hypothesis tested

Retest whether the active frozen null-space treatment improves backward transfer on the reduced-memory `reasoning3` proxy suite versus a matched control, using seed 43 after seed 42 produced valid feasibility evidence but missed the +1.5 BWT-point success threshold.

## 2. Experiment performed

Protocol was written to [`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md:1). Execution preserved CF-cycle-3 artifacts by writing seed-43 outputs only under [`nullspace_ablation_seed43`](nullspace_ablation_seed43/).

Fixed parameters:

- Arm order: control first, treatment second, no `--arm both`.
- Task suite: `reasoning3` only.
- Seed: 43.
- Epochs: 2.
- Rank: 4.
- Max length: 256.
- VRAM threshold: 7500 MB.

## 3. Files changed or created

- Created [`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md:1) to preserve the concise seed-43 protocol and gates.
- Created [`validate_seed43_arm_gate.py`](validate_seed43_arm_gate.py:1) to validate seed-43 control/treatment artifacts for completed status, finite artifacts, expected parameters, expected manifest, under-threshold VRAM, and treatment projection evidence.
- Created this handoff packet: [`seed43_result_critic_packet.md`](seed43_result_critic_packet.md:1).

## 4. Commands/checks run

### Isolation check

Observed current checkout is a normal repo checkout on `master`, not an isolated worktree:

```cmd
git rev-parse --git-dir && git rev-parse --git-common-dir && git branch --show-current && git rev-parse --show-superproject-working-tree
```

Observed output:

```text
.git
.git
master
```

No manual worktree was created because this was an in-place experiment artifact run requested by the loop handoff.

### CPU/data/GPU preflight

```cmd
python -m py_compile train.py experiments\cf_cycle_1\run_nullspace_ablation.py experiments\cf_cycle_1\compare_nullspace_ablation.py experiments\cf_cycle_3\reasoning3_prepare_data_preflight.py experiments\cf_cycle_4\validate_seed43_arm_gate.py && python -m pytest tests\test_data.py tests\test_cf_cycle_2_guardrails.py -q && python experiments\cf_cycle_3\reasoning3_prepare_data_preflight.py --seed 43 --max-length 256 --out experiments\cf_cycle_4\nullspace_ablation_seed43\reasoning3_prepare_data_preflight.json --fail-on-invalid > experiments\cf_cycle_4\nullspace_ablation_seed43\reasoning3_prepare_data_preflight.stdout.json && nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv > experiments\cf_cycle_4\nullspace_ablation_seed43\gpu_preflight.csv && type experiments\cf_cycle_4\nullspace_ablation_seed43\gpu_preflight.csv
```

Observed result:

- Exit code 0.
- 19 tests passed.
- Data preflight valid true in [`reasoning3_prepare_data_preflight.json`](nullspace_ablation_seed43/reasoning3_prepare_data_preflight.json:1): all three tasks had zero zero-label batches, max sequence length 256, and valid true.
- GPU preflight in [`gpu_preflight.csv`](nullspace_ablation_seed43/gpu_preflight.csv:1): RTX 4060 Ti, 694 MiB used / 8188 MiB total, 0% utilization.

### Control launch

```cmd
python experiments\cf_cycle_1\run_nullspace_ablation.py --arm control --task-suite reasoning3 --seed 43 --epochs 2 --rank 4 --max-length 256 --output-root experiments\cf_cycle_4\nullspace_ablation_seed43 --vram-threshold-mb 7500 > experiments\cf_cycle_4\nullspace_ablation_seed43\control.log 2>&1
```

Observed result:

- Completed with status `completed` in [`control/run_status.json`](nullspace_ablation_seed43/control/run_status.json:1).
- Peak VRAM: 6843.90966796875 MB.
- Wall time: 1497.7814581394196 seconds.

### Control gate

```cmd
python experiments\cf_cycle_4\validate_seed43_arm_gate.py --root experiments\cf_cycle_4\nullspace_ablation_seed43 --arm control > experiments\cf_cycle_4\nullspace_ablation_seed43\control_gate.json
```

Observed result:

- Exit code 0.
- [`control_gate.json`](nullspace_ablation_seed43/control_gate.json:1) reports `valid: true`, completed status, matching seed/rank/epochs/max length, expected reasoning3 manifest, finite metrics/instrumentation/matrix, and peak VRAM under threshold.

### Treatment launch

GPU was checked after control before treatment:

```cmd
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv > experiments\cf_cycle_4\nullspace_ablation_seed43\gpu_before_treatment.csv && type experiments\cf_cycle_4\nullspace_ablation_seed43\gpu_before_treatment.csv
```

Observed output in [`gpu_before_treatment.csv`](nullspace_ablation_seed43/gpu_before_treatment.csv:1): 707 MiB used / 8188 MiB total, 1% utilization.

```cmd
python experiments\cf_cycle_1\run_nullspace_ablation.py --arm treatment --task-suite reasoning3 --seed 43 --epochs 2 --rank 4 --max-length 256 --output-root experiments\cf_cycle_4\nullspace_ablation_seed43 --vram-threshold-mb 7500 > experiments\cf_cycle_4\nullspace_ablation_seed43\treatment.log 2>&1
```

Observed result:

- Completed with status `completed` in [`treatment/run_status.json`](nullspace_ablation_seed43/treatment/run_status.json:1).
- Peak VRAM: 6843.90966796875 MB.
- Wall time: 1501.8400311470032 seconds.

### Treatment gate and comparison

```cmd
python experiments\cf_cycle_4\validate_seed43_arm_gate.py --root experiments\cf_cycle_4\nullspace_ablation_seed43 --arm treatment > experiments\cf_cycle_4\nullspace_ablation_seed43\treatment_gate.json && python experiments\cf_cycle_1\compare_nullspace_ablation.py --root experiments\cf_cycle_4\nullspace_ablation_seed43 --vram-threshold-mb 7500 > experiments\cf_cycle_4\nullspace_ablation_seed43\comparison.json && type experiments\cf_cycle_4\nullspace_ablation_seed43\treatment_gate.json && type experiments\cf_cycle_4\nullspace_ablation_seed43\comparison.json
```

Observed result:

- Exit code 0.
- [`treatment_gate.json`](nullspace_ablation_seed43/treatment_gate.json:1) reports `valid: true`, completed status, finite artifacts, expected manifest, under-threshold VRAM, non-empty frozen basis, 3600 calls with frozen basis, and `removed_norm_sum=3.100088352795865`.
- [`comparison.json`](nullspace_ablation_seed43/comparison.json:1) reports `valid: true`, `projection_active: true`, `frozen_basis_nonempty: true`, both peaks under 7500 MB, `delta_bwt_points=-0.008929480489536235`, `delta_avg_acc_points=-0.008046083930968173`, old-task delta gaps `[-0.1564058367372001, 0.13854687575812763]`, and `continue: false`.

## 5. Raw or summarized results

Control metrics in [`control/metrics.json`](nullspace_ablation_seed43/control/metrics.json:1):

- `avg_acc=0.5665482637215719`.
- `bwt=-0.026559348750060024`.
- `old_task_deltas=[-0.04616356831829843, -0.006955129181821618]`.
- `peak_vram_mb=6843.90966796875`.

Treatment metrics in [`treatment/metrics.json`](nullspace_ablation_seed43/treatment/metrics.json:1):

- `avg_acc=0.5664678028822622`.
- `bwt=-0.026648643554955387`.
- `old_task_deltas=[-0.04772762668567043, -0.005569660424240341]`.
- `peak_vram_mb=6843.90966796875`.

Comparison facts in [`comparison.json`](nullspace_ablation_seed43/comparison.json:1):

- Feasibility: confirmed valid.
- Projection: confirmed active and frozen basis non-empty.
- VRAM: confirmed both arms below 7500 MB.
- Algorithmic gate: contradicted because BWT did not improve by +1.5 points; it was slightly negative at -0.008929480489536235 points.
- Average accuracy gate: passed, no meaningful degradation versus the -2.0 point allowance.
- Old-task gap gate: passed, no gap worse than -3.0 points.

## 6. Result label

Contradicted for algorithmic success / escalation. Confirmed for feasibility and active-projection execution.

## 7. Confidence level

High confidence for bounded feasibility and gate interpretation because control, treatment, treatment projection, and comparison all completed with explicit gate artifacts.

Medium confidence for the negative/near-zero treatment-effect interpretation because this is still one seed on a reduced 3-task proxy suite, not a full current4/v10/Digital Twin run.

## 8. Recommended next action

Send this packet to llm-result-critic. If the critic accepts the evidence, pivot to revised treatment-strength hypotheses before any larger current4, v10, or Digital Twin escalation. Do not claim catastrophic forgetting is solved.

## 9. Checks not run

- No full `current4` run.
- No v10 run.
- No Digital Twin task.
- No generative subset evaluation.
- No additional seeds beyond seed 43 in this cycle.

