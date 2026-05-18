# CF-cycle-4 Seed-43 Reasoning3 Replication Protocol

## Objective

Run one additional reduced-memory `reasoning3` null-space ablation replication with seed 43 before any larger `current4`, v10, or Digital Twin escalation. Preserve CF-cycle-3 artifacts by writing only under `experiments/cf_cycle_4/nullspace_ablation_seed43`.

## Hypothesis under retest

The active frozen null-space treatment may improve backward transfer on the reduced `reasoning3` proxy suite relative to the matched control. Seed 42 was valid feasibility evidence but missed the predeclared mitigation threshold, so seed 43 is a bounded retest, not proof that catastrophic forgetting is solved.

## Fixed parameters

- Task suite: `reasoning3` only: GSM8K, ARC, CSQA.
- Seed: `43`.
- Epochs: `2`.
- Rank: `4`.
- Max length: `256`.
- VRAM threshold: `7500` MB.
- Output root: `experiments\cf_cycle_4\nullspace_ablation_seed43`.
- Launch policy: sequential arm execution only; do not use `--arm both`.

## Preflight checks

Run CPU checks and inspect GPU idleness before spending GPU time:

```cmd
python -m py_compile train.py experiments\cf_cycle_1\run_nullspace_ablation.py experiments\cf_cycle_1\compare_nullspace_ablation.py experiments\cf_cycle_3\reasoning3_prepare_data_preflight.py experiments\cf_cycle_4\validate_seed43_arm_gate.py && python -m pytest tests\test_data.py tests\test_cf_cycle_2_guardrails.py -q && python experiments\cf_cycle_3\reasoning3_prepare_data_preflight.py > experiments\cf_cycle_4\nullspace_ablation_seed43\reasoning3_prepare_data_preflight.json
```

```cmd
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
```

Proceed only if the data preflight is valid and GPU memory/use look idle enough for a run expected to peak below 7500 MB.

## Launch sequence

Create the output root:

```cmd
if not exist experiments\cf_cycle_4\nullspace_ablation_seed43 mkdir experiments\cf_cycle_4\nullspace_ablation_seed43
```

### Step 1: control arm only

```cmd
python experiments\cf_cycle_1\run_nullspace_ablation.py --arm control --task-suite reasoning3 --seed 43 --epochs 2 --rank 4 --max-length 256 --output-root experiments\cf_cycle_4\nullspace_ablation_seed43 --vram-threshold-mb 7500 > experiments\cf_cycle_4\nullspace_ablation_seed43\control.log 2>&1
```

Validate the completed control before launching treatment:

```cmd
python experiments\cf_cycle_4\validate_seed43_arm_gate.py --root experiments\cf_cycle_4\nullspace_ablation_seed43 --arm control > experiments\cf_cycle_4\nullspace_ablation_seed43\control_gate.json
```

Stop if the control command exits nonzero or `control_gate.json` reports `valid: false`.

### Step 2: treatment arm only, conditional on valid control

```cmd
python experiments\cf_cycle_1\run_nullspace_ablation.py --arm treatment --task-suite reasoning3 --seed 43 --epochs 2 --rank 4 --max-length 256 --output-root experiments\cf_cycle_4\nullspace_ablation_seed43 --vram-threshold-mb 7500 > experiments\cf_cycle_4\nullspace_ablation_seed43\treatment.log 2>&1
```

Validate the completed treatment before comparison:

```cmd
python experiments\cf_cycle_4\validate_seed43_arm_gate.py --root experiments\cf_cycle_4\nullspace_ablation_seed43 --arm treatment > experiments\cf_cycle_4\nullspace_ablation_seed43\treatment_gate.json
```

Stop if the treatment command exits nonzero, if finite/status/manifest/VRAM gates fail, or if projection evidence is inactive.

### Step 3: comparison, conditional on valid control and treatment

```cmd
python experiments\cf_cycle_1\compare_nullspace_ablation.py --root experiments\cf_cycle_4\nullspace_ablation_seed43 --vram-threshold-mb 7500 > experiments\cf_cycle_4\nullspace_ablation_seed43\comparison.json
```

## Gate criteria

Feasibility requires:

1. Both arms completed with finite metrics, finite instrumentation, finite accuracy matrices, and matching `reasoning3` manifests.
2. Both peak VRAM values are finite and at or below 7500 MB.
3. Treatment projection evidence is active: non-empty frozen basis, calls with frozen basis, and positive removed norm sum.
4. `comparison.json` reports `valid: true`.

Algorithmic success can be considered only after feasibility succeeds and requires:

1. `delta_bwt_points >= 1.5`.
2. `delta_avg_acc_points >= -2.0`.
3. Every old-task delta gap is at least `-3.0` points.

## Decision rule

- If seed 43 clears feasibility and algorithmic gates, hand the evidence to result critic before any escalation.
- If seed 43 is valid but misses +1.5 BWT improvement or flips sign, recommend pivoting to revised treatment-strength hypotheses before larger `current4`, v10, or Digital Twin runs.
- If any guardrail fails, stop immediately, preserve artifacts, and hand the failed-run packet to debug/result critic.

Do not claim catastrophic forgetting is solved from this reduced proxy replication.

