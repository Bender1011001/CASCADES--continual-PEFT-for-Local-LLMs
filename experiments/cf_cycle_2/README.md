# CF-cycle-2 Reasoning3 Guardrail Runbook

CF-cycle-2 is a code-first guardrail repair cycle for the direct null-space ablation harness. GPU runs remain blocked until CPU-light verification passes and the experiment-design flow explicitly approves the reduced-memory launch.

## CPU-light verification required before GPU work

Run these commands from the repository root:

```cmd
python -m py_compile train.py experiments\cf_cycle_1\run_nullspace_ablation.py experiments\cf_cycle_1\compare_nullspace_ablation.py
python -m pytest tests/test_data.py tests/test_cf_cycle_2_guardrails.py -q
```

The checks verify that:

- `train_cascades()` keeps normal defaults backward compatible while the direct wrapper opts into non-finite loss aborts and hard peak-VRAM checks.
- Each arm writes `run_status.json` so failed guardrail runs become durable evidence instead of partial success artifacts.
- Comparison rejects missing artifacts, failed statuses, non-finite metrics, peak VRAM above 7500 MB for either arm, and inactive treatment projection evidence.

## Later explicit-approval GPU commands only

Do not run these commands from code mode without explicit experiment-design approval. The invalid CF-cycle-1 `current4` artifacts are diagnostic only and must not be reused as algorithmic evidence.

```cmd
if not exist experiments\cf_cycle_2\nullspace_ablation mkdir experiments\cf_cycle_2\nullspace_ablation
python experiments\cf_cycle_1\run_nullspace_ablation.py --arm control --task-suite reasoning3 --seed 42 --epochs 2 --rank 4 --max-length 256 --output-root experiments\cf_cycle_2\nullspace_ablation --vram-threshold-mb 7500 > experiments\cf_cycle_2\nullspace_ablation\control.log 2>&1
python experiments\cf_cycle_1\run_nullspace_ablation.py --arm treatment --task-suite reasoning3 --seed 42 --epochs 2 --rank 4 --max-length 256 --output-root experiments\cf_cycle_2\nullspace_ablation --vram-threshold-mb 7500 > experiments\cf_cycle_2\nullspace_ablation\treatment.log 2>&1
python experiments\cf_cycle_1\compare_nullspace_ablation.py --root experiments\cf_cycle_2\nullspace_ablation --vram-threshold-mb 7500 > experiments\cf_cycle_2\nullspace_ablation\comparison.json
```

Success after approved GPU execution means both `reasoning3` arms complete with finite metrics, `run_status.json` status `completed`, peak VRAM at or below 7500 MB, matching manifests, and active treatment frozen-basis projection evidence. This runbook does not claim catastrophic forgetting is solved.
