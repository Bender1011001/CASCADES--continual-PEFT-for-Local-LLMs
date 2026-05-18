# CF-cycle-9 Implementation Report — utility-veto active-freeze CPU closeout

Date: 2026-05-18

## 1. Cycle objective

Close CF-cycle-9 implementation durably after code mode implemented the CPU-preflight-ready treatment variant cllora-active-freeze-utility-veto-topk-2 from [`EXPERIMENT_DESIGN_PACKET.md`](experiments/cf_cycle_9/EXPERIMENT_DESIGN_PACKET.md:1), preserving the implementation evidence, critic verdict, caveats, non-claims, and next-cycle seed without launching GPU work or overstating the result.

Objective label: CPU implementation readiness likely supported; direct utility-probe validity weak; structural proxy utility-veto active-freeze treatment; catastrophic forgetting is not solved.

## 2. What was done

- Implemented only the bounded treatment variant cllora-active-freeze-utility-veto-topk-2. The starting mechanism remains cllora-active-freeze-topk-2 in [`config_for_arm()`](experiments/cf_cycle_1/run_nullspace_ablation.py:85), with top-k 2, frozen-basis variance threshold 0.05, active reassignment, soft EAR, and CL-LoRA active-freeze behavior preserved.
- Added default-off utility admission configuration fields in [`AblationConfig`](cascades/config.py:60), so historical/default salience admission remains the default path while the new variant explicitly selects utility-veto admission.
- Added pure CPU-testable utility-veto decision logic in [`UtilityAdmissionResult`](cascades/adapters.py:37) and its utility admission helper in [`adapters.py`](cascades/adapters.py:37). The rule admits candidates only when their mean utility is nonnegative and no old-task proxy delta falls below the veto threshold; all-veto cases do not fall back to salience-only admission.
- Threaded utility-veto admission into the adapter freeze path in [`adapters.py`](cascades/adapters.py:37) while preserving prior salience/top-k behavior for default and legacy variants.
- Extended variant routing and instrumentation in [`run_nullspace_ablation.py`](experiments/cf_cycle_1/run_nullspace_ablation.py:85) so freeze events and run artifacts can record admission policy, utility-probe status, considered/admitted/vetoed counts, zero-admission reasons, utility deltas, and per-old-task veto counts.
- Extended the active treatment validator in [`validate_active_treatment_gate.py`](experiments/cf_cycle_5/validate_active_treatment_gate.py:39) with optional expected admission-policy and utility-probe checks while keeping zero admission diagnostic rather than automatically fatal.
- Added CPU regression and fixture coverage in [`test_cf_cycle_9_utility_veto.py`](tests/test_cf_cycle_9_utility_veto.py:1), including config defaults/serialization, variant routing, legacy preservation, all-veto no-fallback behavior, small-tensor shape/orthogonality checks, and validator fixture coverage.
- Created CPU evidence artifacts in [`treatment_active_gate.json`](experiments/cf_cycle_9/validator_fixture/treatment_active_gate.json:1) and [`reasoning3_prepare_data_preflight.json`](experiments/cf_cycle_9/cllora_active_freeze_utility_veto_topk2_seed43_paired/reasoning3_prepare_data_preflight.json:1).
- Updated project memory in [`CONTEXT.md`](CONTEXT.md:1). This report-writer step did not run commands, did not edit algorithm code, did not request GPU approval, and did not launch training.

## 3. Files changed or inspected

### Changed during implementation

- [`cascades/config.py`](cascades/config.py:60) — added default-off utility admission configuration fields.
- [`cascades/adapters.py`](cascades/adapters.py:37) — added CPU-testable utility-veto admission logic and integrated it into frozen-basis admission.
- [`experiments/cf_cycle_1/run_nullspace_ablation.py`](experiments/cf_cycle_1/run_nullspace_ablation.py:85) — added the cllora-active-freeze-utility-veto-topk-2 treatment routing and instrumentation fields.
- [`experiments/cf_cycle_5/validate_active_treatment_gate.py`](experiments/cf_cycle_5/validate_active_treatment_gate.py:39) — added utility/admission-policy validator checks.
- [`tests/test_cf_cycle_9_utility_veto.py`](tests/test_cf_cycle_9_utility_veto.py:1) — added CPU tests for the new utility-veto treatment surface.
- [`experiments/cf_cycle_9/validator_fixture/treatment_active_gate.json`](experiments/cf_cycle_9/validator_fixture/treatment_active_gate.json:1) — added synthetic active-gate fixture for validator preflight.
- [`experiments/cf_cycle_9/cllora_active_freeze_utility_veto_topk2_seed43_paired/reasoning3_prepare_data_preflight.json`](experiments/cf_cycle_9/cllora_active_freeze_utility_veto_topk2_seed43_paired/reasoning3_prepare_data_preflight.json:1) — added reduced reasoning3 data preflight artifact.

### Changed during report-writer closeout

- [`experiments/cf_cycle_9/IMPLEMENTATION_REPORT.md`](experiments/cf_cycle_9/IMPLEMENTATION_REPORT.md:1) — durable CF-cycle-9 implementation closeout report.
- [`CONTEXT.md`](CONTEXT.md:1) — project memory updated with closeout decision and next-cycle seed.

### Evidence files inspected or summarized

- [`EXPERIMENT_DESIGN_PACKET.md`](experiments/cf_cycle_9/EXPERIMENT_DESIGN_PACKET.md:1) — original bounded design packet and stop conditions.
- [`treatment_active_gate.json`](experiments/cf_cycle_9/validator_fixture/treatment_active_gate.json:1) — synthetic validator fixture with active utility-admission fields.
- [`reasoning3_prepare_data_preflight.json`](experiments/cf_cycle_9/cllora_active_freeze_utility_veto_topk2_seed43_paired/reasoning3_prepare_data_preflight.json:1) — reduced reasoning3 data preflight.
- [`CONTEXT.md`](CONTEXT.md:1) — project memory state before and after closeout.

## 4. Commands/checks run

Report-writer closeout: no new shell commands, tests, GPU jobs, training runs, or code-modifying commands were run.

Recorded CPU implementation checks from code mode:

1. Python bytecode compilation over the touched code and test files exited 0.
2. Targeted pytest command over [`test_cf_cycle_9_utility_veto.py`](tests/test_cf_cycle_9_utility_veto.py:1), [`test_adapters_v9.py`](tests/test_adapters_v9.py:1), and [`test_cf_cycle_2_guardrails.py`](tests/test_cf_cycle_2_guardrails.py:1) passed 44 tests with 0 failures.
3. The active treatment validator CLI against the synthetic fixture returned valid true in [`treatment_active_gate.json`](experiments/cf_cycle_9/validator_fixture/treatment_active_gate.json:2).
4. Reduced reasoning3 data preflight returned valid true in [`reasoning3_prepare_data_preflight.json`](experiments/cf_cycle_9/cllora_active_freeze_utility_veto_topk2_seed43_paired/reasoning3_prepare_data_preflight.json:6), with no zero-label batches.
5. The only observed warning was the pre-existing pytest-asyncio loop-scope deprecation warning associated with [`pyproject.toml`](pyproject.toml:1).

Skipped checks and non-executed work:

- No GPU job was launched.
- No training run was launched.
- No fresh control/treatment paired comparison was produced for cllora-active-freeze-utility-veto-topk-2.
- No full current4 run, v10 run, Digital Twin run, generative subset, broad settings fuzzer, cadence redesign, rank redesign, or D-MoLE timing change was performed.
- No live held-out old-task model-loss probe was implemented inside the training loop.

## 5. Evidence summary

### Confirmed

- CPU implementation surface exists for cllora-active-freeze-utility-veto-topk-2 across config, adapter admission, runner routing/instrumentation, validator checks, tests, and fixtures.
- CPU checks passed: bytecode compilation exited 0, targeted pytest passed 44 tests with 0 failures, validator fixture returned valid true, and reduced reasoning3 data preflight returned valid true with no zero-label batches.
- Prior variants remain intended to use salience/default-off admission unless the new treatment explicitly selects utility-veto admission.
- The all-veto case is intentionally no-fallback: if every candidate is vetoed, the freeze event records zero admission instead of silently reverting to salience-only top-k admission.

### Likely

- The implementation is ready for protocol review because the CPU-facing surfaces, serialization, routing, and diagnostic artifacts are in place and regression-tested.
- The structural proxy is sufficient for a bounded diagnostic GPU protocol only if the next protocol explicitly checks no-op, all-veto, over-veto, and admitted-direction behavior after old tasks exist.

### Weak signal

- Direct utility-probe validity is weak. The current utility decision does not measure live held-out old-task model loss during training.
- The proxy may behave differently once real frozen bases and old tasks accumulate in a GPU run; it could admit too few directions, admit none, or veto directions that would have helped retained-task accuracy.

### Contradicted or unsupported

- No claim is supported that catastrophic forgetting is solved.
- No claim is supported that cllora-active-freeze-utility-veto-topk-2 improves BWT, average accuracy, old-task gaps, or generative retention.
- No full-suite, current4, v10, Digital Twin, generative, or broad escalation claim is supported.
- No true held-out old-task utility-probe claim is supported because the implemented utility decision is a deterministic tensor proxy.

### Critical caveat and proxy reframe

The largest material design deviation is that no live held-out model-loss probe was added inside training. Old-task replay/evaluation data is not available in the current adapter freeze path without invasive model/data plumbing. The implemented utility decision is therefore a deterministic CPU-testable tensor proxy based on overlap with existing frozen bases, not a true held-out old-task utility probe.

The treatment should be framed as a structural proxy utility-veto active-freeze treatment, not as faithful held-out old-task utility preservation. The next protocol must diagnose proxy behavior directly, including no-op behavior, all-veto behavior, over-veto risk, zero-admission frequency, and whether any directions are admitted after old tasks exist.

## 6. Decision: proceed with constraints

Proceed with constraints: close out implementation and route to GPU protocol review, not immediate GPU execution.

The expected value of a protocol-review cycle is positive because the CPU implementation is likely ready and the remaining uncertainty is now about GPU diagnostic protocol design rather than basic import/test failure. The downside is contained if the next cycle requires explicit proxy diagnostics and explicit user approval before any GPU run.

Do not proceed directly to GPU execution from this closeout. Do not claim the utility veto is faithful to held-out old-task loss. Do not claim catastrophic forgetting is solved.

## 7. Confidence: medium

- High confidence that the CPU-facing implementation artifacts exist and passed the recorded CPU checks.
- Medium confidence that CPU implementation readiness is supported for protocol review.
- Low-to-medium confidence that the structural proxy will act like a useful old-task utility signal in training.
- High confidence that direct held-out-utility validity, BWT improvement, and catastrophic-forgetting solution claims are unsupported.

Overall confidence: medium.

## 8. Next-cycle seed objective

Recommended next-cycle seed objective: route to experiment-designer GPU protocol review for cllora-active-freeze-utility-veto-topk-2, requiring explicit proxy diagnostics and explicit user approval before any GPU run.

Required protocol-review constraints:

- Treat the variant as structural proxy utility-veto active-freeze, not true held-out old-task utility probing.
- Require diagnostics for no-op, all-veto, over-veto, zero-admission frequency, admitted-direction counts, per-old-task veto counts, utility delta summaries, and whether directions are admitted after old tasks exist.
- Preserve reduced reasoning3, seed 43, rank 4, max length 256, two epochs, 7500 MB VRAM threshold, paired fresh-control/fresh-treatment discipline, standard gates, active/utility gate, same-envelope check, and final comparison only after artifacts are valid.
- Require explicit user approval before any GPU launch.
- Continue blocking full current4, v10, Digital Twin, generative subset, broad settings fuzzer, broad escalation, and catastrophic-forgetting solution language.

## 9. Handoff to LLM Loop Coordinator

handoff_target: llm-loop-coordinator

continue_pivot_or_stop: continue to protocol review, not immediate GPU execution.

Current objective: close CF-cycle-9 implementation and seed the next coordinator cycle for permission-gated GPU protocol review.

Current evidence state:

- Confirmed: CPU implementation artifacts are present for config, adapter admission, runner routing/instrumentation, validator checks, tests, and fixtures.
- Confirmed: recorded CPU checks passed, including bytecode compilation, 44 targeted pytest tests, synthetic validator valid true, and reduced reasoning3 data preflight valid true with no zero-label batches.
- Likely: CPU implementation readiness is sufficient for experiment-designer protocol review.
- Weak signal: direct utility-probe validity, because the implementation uses a deterministic structural tensor proxy rather than live held-out old-task model-loss evaluation.
- Unsupported: GPU/training utility, BWT improvement, average-accuracy improvement, old-task gap improvement, catastrophic-forgetting solution claims, and full-suite escalation.
- Blocked: any GPU run until explicit user approval after protocol review.

Next mode to use: llm-loop-coordinator.

Specific task for that mode: route the next cycle to LLM Experiment Designer for a permission-gated GPU protocol review of cllora-active-freeze-utility-veto-topk-2. The protocol must include explicit proxy diagnostics for no-op, all-veto, over-veto, zero-admission, admitted directions after old tasks exist, and per-old-task veto behavior. The coordinator must not request or launch GPU work in its own step, and no GPU execution should occur without explicit user approval.

Stop condition for coordinator: produce the next routing handoff only; do not run GPU jobs, do not edit algorithm code, do not request approval prematurely, and do not claim catastrophic forgetting is solved.
