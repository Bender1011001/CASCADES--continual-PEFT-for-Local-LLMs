# CASCADES CF Context Pack for Advanced Math/Science Review

This directory is a compact, self-contained handoff bundle for an external advanced math/science LLM reviewer. Its purpose is to help solve the current CASCADES catastrophic-forgetting problem without forcing the reviewer to ingest the entire repository.

## How to use this pack

1. Give the reviewer the contents of this directory, preserving relative paths where possible.
2. Start with [`MASTER_PROMPT.md`](MASTER_PROMPT.md).
3. Ask the reviewer to read [`CONTEXT.md`](CONTEXT.md), then the core code files, then the experiment evidence files.
4. Ask for a concrete next intervention and an experiment design that can run on the RTX 4060 Ti 8GB constraint.

## Main question for the reviewer

The current isolated frozen-nullspace intervention is validly implemented and activated, but it does not robustly reduce forgetting on the reduced `reasoning3` suite:

- Seed 42: small positive delta BWT, below success threshold.
- Seed 43: nearly flat/slightly negative delta BWT.

The reviewer should identify whether the failure is likely mathematical, architectural, optimization-related, evaluation-related, or experimental-design-related, then propose the next highest-information treatment.

## Bundle layout

Most files are copied under this directory using their original repository-relative paths. The original project [`README.md`](../../../README.md) copy is stored as [`PROJECT_README.md`](PROJECT_README.md) so this pack can use [`README.md`](README.md) as the bundle guide.

### Project memory and prior hypotheses

- [`CONTEXT.md`](CONTEXT.md)
- [`PROJECT_README.md`](PROJECT_README.md)
- [`experiments/v10_hypotheses.md`](experiments/v10_hypotheses.md)
- [`reports/parametric_memory_issue/REPORT.md`](reports/parametric_memory_issue/REPORT.md)

### Core implementation files

- [`cascades/config.py`](cascades/config.py)
- [`cascades/adapters.py`](cascades/adapters.py)
- [`cascades/math_ops.py`](cascades/math_ops.py)
- [`cascades/injection.py`](cascades/injection.py)
- [`cascades/sleep.py`](cascades/sleep.py)
- [`cascades/data.py`](cascades/data.py)
- [`cascades/metrics.py`](cascades/metrics.py)
- [`cascades/eval.py`](cascades/eval.py)
- [`train.py`](train.py)
- [`experiment_matrix.py`](experiment_matrix.py)

### Experiment harness and validation

- [`experiments/cf_cycle_1/run_nullspace_ablation.py`](experiments/cf_cycle_1/run_nullspace_ablation.py)
- [`experiments/cf_cycle_1/compare_nullspace_ablation.py`](experiments/cf_cycle_1/compare_nullspace_ablation.py)
- [`experiments/cf_cycle_2/EXPERIMENT_PROTOCOL.md`](experiments/cf_cycle_2/EXPERIMENT_PROTOCOL.md)
- [`experiments/cf_cycle_2/REPORT.md`](experiments/cf_cycle_2/REPORT.md)
- [`tests/test_data.py`](tests/test_data.py)
- [`tests/test_cf_cycle_2_guardrails.py`](tests/test_cf_cycle_2_guardrails.py)

### Recent evidence

- [`experiments/cf_cycle_3/root_cause_packet.md`](experiments/cf_cycle_3/root_cause_packet.md)
- [`experiments/cf_cycle_3/REPORT.md`](experiments/cf_cycle_3/REPORT.md)
- [`experiments/cf_cycle_3/nullspace_ablation_retry/comparison.json`](experiments/cf_cycle_3/nullspace_ablation_retry/comparison.json)
- [`experiments/cf_cycle_3/control_gate_validation.json`](experiments/cf_cycle_3/control_gate_validation.json)
- [`experiments/cf_cycle_3/reasoning3_prepare_data_preflight.json`](experiments/cf_cycle_3/reasoning3_prepare_data_preflight.json)
- [`experiments/cf_cycle_4/EXPERIMENT_PROTOCOL.md`](experiments/cf_cycle_4/EXPERIMENT_PROTOCOL.md)
- [`experiments/cf_cycle_4/seed43_result_critic_packet.md`](experiments/cf_cycle_4/seed43_result_critic_packet.md)
- [`experiments/cf_cycle_4/REPORT.md`](experiments/cf_cycle_4/REPORT.md)
- [`experiments/cf_cycle_4/nullspace_ablation_seed43/comparison.json`](experiments/cf_cycle_4/nullspace_ablation_seed43/comparison.json)
- [`experiments/cf_cycle_4/nullspace_ablation_seed43/control_gate.json`](experiments/cf_cycle_4/nullspace_ablation_seed43/control_gate.json)
- [`experiments/cf_cycle_4/nullspace_ablation_seed43/treatment_gate.json`](experiments/cf_cycle_4/nullspace_ablation_seed43/treatment_gate.json)

## Important caveats

- Catastrophic forgetting is not solved.
- Current evidence supports harness validity and projection activation, not algorithmic success.
- The Digital Twin task and `current4` suite remain blocked until a stronger reduced-suite treatment exists.
- Do not propose a large exhaustive grid search as the first answer; this project is GPU-budget constrained.
- `num_samples` is known to be an unreliable experimental knob until it is wired through the data path.

