# CF-cycle-9 Experiment Design Packet — utility-coupled frozen-basis admission

Date: 2026-05-18

## 1. Hypothesis tested

Utility-coupled frozen-basis admission with a per-old-task veto will convert the strong CF-cycle-8 active/top-k projection mechanism into larger old-task retention by admitting frozen basis columns only when a cheap old-task utility probe predicts benefit and no old task is materially harmed.

Prior evidence anchor:

- CF-cycle-8 report: [`experiments/cf_cycle_8/REPORT.md`](experiments/cf_cycle_8/REPORT.md:1).
- CF-cycle-8 treatment gate: [`experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/treatment_active_gate.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/treatment_active_gate.json:1).
- CF-cycle-8 comparison: [`experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/comparison.json`](experiments/cf_cycle_8/cllora_active_freeze_topk2_seed43_paired/comparison.json:1).

Interpretation to preserve: CF-cycle-8 is evidence-usable, mechanically strong, and weak-positive, but not promotion-passing; catastrophic forgetting is not solved.

## 2. Experiment family and treatment variant

Run exactly one bounded treatment variant against a fresh same-seed control under the CF-cycle-8 reduced envelope.

Recommended treatment variant name:

- `cllora-active-freeze-utility-veto-topk-2`

Recommended config flags to add in a later code-mode handoff:

- `frozen_basis_top_k_per_freeze = 2`
- `frozen_basis_variance_threshold = 0.05`
- `frozen_basis_admission_policy = "utility_veto"`
- `frozen_basis_utility_probe_enabled = True`
- `frozen_basis_utility_probe_batch_size = 2`
- `frozen_basis_utility_probe_batches_per_old_task = 1`
- `frozen_basis_utility_min_mean_delta = 0.0`
- `frozen_basis_utility_old_task_veto_drop = 0.0`
- `frozen_basis_utility_max_probe_examples_per_old_task = 2`
- `frozen_basis_utility_metric = "heldout_loss_proxy"`

The thresholds are intentionally conservative for the first run: require non-negative mean utility and no negative per-old-task predicted change. If code-mode implementation discovers the probe is too noisy at zero margin, it may set `frozen_basis_utility_old_task_veto_drop = -0.001` only if the CPU unit tests document the rationale and the final artifact records it.

## 3. Concrete protocol for implementing the variant

### 3.1 Admission point

Start from the existing top-k admission path in [`cascades/adapters.py`](cascades/adapters.py:366), especially the current `freeze_current_subspace()` flow in [`cascades/adapters.py`](cascades/adapters.py:380) through [`cascades/adapters.py`](cascades/adapters.py:493).

Keep the current structural candidate generation:

1. Build normalized U sketch directions from `streaming_sketch_U`.
2. Score candidates using normalized eigenvalue share.
3. Apply `frozen_basis_variance_threshold`.
4. Apply top-k capacity with `frozen_basis_top_k_per_freeze = 2`.
5. Repeat analogous V-side admission from `streaming_sketch_V`.

Then insert the utility admission step between candidate generation and appending to the frozen basis:

1. Produce candidate U and V directions as the existing mechanism does.
2. For each candidate direction or aligned U/V candidate pair, run a cheap old-task utility probe.
3. Admit the candidate only if all utility criteria pass.
4. If no candidate passes, do not force-admit the top structural direction; log this as `utility_gate_admitted_n = 0` and treat it as interpretable mechanism-under-conversion evidence.

### 3.2 Utility probe design

The probe should be CPU-testable and GPU-bounded. It does not need to prove true downstream utility; it only needs to be a consistent admission signal.

Preferred probe abstraction:

- Add a pure helper that accepts old-task probe summaries and candidate scores, then returns an admission decision. This helper should be unit-tested on CPU without loading a model.
- Keep model-dependent probe collection behind a small callable or hook so the decision logic is not entangled with training.

Minimum useful signal:

- For every old task already completed before the freeze call, collect one tiny held-out or replay mini-batch summary.
- Estimate predicted old-task harm or benefit for a candidate by comparing a proxy score before and after applying the candidate projection in a no-gradient evaluation context.
- Use loss delta if available. If exact loss replay is too invasive, use a deterministic tensor proxy derived from old-task gradients already available in the adapter instrumentation.

Decision rule:

- Let `utility_delta_by_old_task` be positive when the candidate is predicted to improve retention and negative when predicted to harm retention.
- Admit only if `mean(utility_delta_by_old_task) >= frozen_basis_utility_min_mean_delta`.
- Veto if any `utility_delta_by_old_task < frozen_basis_utility_old_task_veto_drop`.
- Break ties by structural score so the cap remains top-k among utility-passing candidates, not top-k before utility filtering.

Important design constraint: the utility gate must not silently fall back to salience-only admission. If all candidates fail the utility gate, the treatment should continue with an empty new admission for that freeze event and record the reason.

### 3.3 Per-old-task veto logging

Add instrumentation fields under the existing `freeze_events` entries produced by [`experiments/cf_cycle_1/run_nullspace_ablation.py`](experiments/cf_cycle_1/run_nullspace_ablation.py:262):

- `admission_policy`
- `utility_probe_enabled`
- `utility_candidates_considered_u`
- `utility_candidates_admitted_u`
- `utility_candidates_vetoed_u`
- `utility_candidates_considered_v`
- `utility_candidates_admitted_v`
- `utility_candidates_vetoed_v`
- `utility_mean_delta_by_candidate`
- `utility_min_old_task_delta_by_candidate`
- `utility_veto_task_indices_by_candidate`
- `utility_gate_admitted_any`
- `utility_gate_zero_admission_reason`

Also add aggregate instrumentation under a new top-level `utility_admission` object in `instrumentation.json`:

- `freeze_calls_with_utility_probe`
- `candidates_considered_total`
- `candidates_admitted_total`
- `candidates_vetoed_total`
- `zero_admission_freeze_calls`
- `mean_utility_delta_sum`
- `min_old_task_delta_min`
- `per_old_task_veto_counts`

These fields make the next cycle actionable: if BWT under-converts despite admissions, cadence or probe magnitude becomes the likely next lever; if admission is near-zero, utility gating itself is too strict or the candidate directions are not retention-useful.

## 4. Expected files/code areas for a later code-mode handoff

Minimal implementation targets:

1. [`cascades/config.py`](cascades/config.py:15)
   - Add frozen-basis utility admission flags to `AblationConfig` with salience-only defaults preserving existing behavior.

2. [`cascades/adapters.py`](cascades/adapters.py:366)
   - Add pure admission helpers near `freeze_current_subspace()` or in a small adjacent helper section.
   - Thread the config-controlled admission policy into U and V candidate filtering.
   - Preserve current behavior for existing variants when `frozen_basis_admission_policy` is absent or set to `"salience"`.

3. [`experiments/cf_cycle_1/run_nullspace_ablation.py`](experiments/cf_cycle_1/run_nullspace_ablation.py:79)
   - Add `cllora-active-freeze-utility-veto-topk-2` to supported treatment variants and CLI choices.
   - Configure the new flags only for treatment arm and only for the new variant.
   - Extend instrumentation wrapping in [`install_instrumentation()`](experiments/cf_cycle_1/run_nullspace_ablation.py:184) and `wrapped_freeze()` in [`experiments/cf_cycle_1/run_nullspace_ablation.py`](experiments/cf_cycle_1/run_nullspace_ablation.py:254) to capture utility admission fields.

4. [`experiments/cf_cycle_5/validate_active_treatment_gate.py`](experiments/cf_cycle_5/validate_active_treatment_gate.py:39)
   - Keep existing active/capacity checks.
   - Add optional expected-admission-policy and utility-probe checks.
   - Report `utility_gate_admitted_any`, admitted/vetoed counts, and zero-admission freeze calls without making zero admission an automatic hard failure.

5. Optional small test file under [`tests`](tests)
   - Add CPU-only unit tests for admission decision logic if this repository already runs lightweight tests. If test discovery is unclear, create a narrow script under [`experiments/cf_cycle_9`](experiments/cf_cycle_9) instead of broadening the test framework.

## 5. CPU-only preflight and unit checks before any GPU request

These checks are required before asking for a GPU run. They are designed to run without model loading and without CUDA.

1. Config serialization check.
   - Instantiate the new treatment config through `config_for_arm("treatment", "cllora-active-freeze-utility-veto-topk-2")`.
   - Assert `enable_coso_nullspace`, `enable_cllora_reassign`, `enable_soft_ear`, and top-k 2 are preserved.
   - Assert the new utility admission flags are present in `asdict(cfg)` and JSON-serializable.

2. Variant routing check.
   - Assert control arm config is unchanged for the new treatment variant.
   - Assert existing variants still produce the same values for `frozen_basis_variance_threshold` and `frozen_basis_top_k_per_freeze`.

3. Pure utility admission tests.
   - Case A: two candidates, both old-task deltas nonnegative, admit top structural candidate up to k=2.
   - Case B: one candidate improves task 0 but harms task 1, veto it even if its structural score is highest.
   - Case C: all candidates vetoed, return zero admissions and `utility_gate_zero_admission_reason` without fallback.
   - Case D: no old tasks yet, bypass utility probe and admit by structural top-k only for the first task boundary if and only if no old-task veto is meaningful.

4. Shape and orthogonality smoke test.
   - Use small random tensors to exercise U-side and V-side candidate selection on CPU.
   - Assert admitted basis column counts never exceed 2 per freeze call.
   - Assert merged basis columns remain finite and approximately orthonormal.

5. Validator dry-run fixture.
   - Create a tiny synthetic `config.json`, `instrumentation.json`, and `run_status.json` fixture under [`experiments/cf_cycle_9`](experiments/cf_cycle_9).
   - Run the active treatment validator against the fixture after code-mode implementation.
   - Expected: valid when active/capacity evidence is positive and utility fields are internally consistent; valid with a warning-like summary when zero admission is recorded, but not valid if projection materiality is claimed without frozen-basis calls.

6. Data preflight command, CPU-only, later execution only:

```cmd
python experiments\cf_cycle_3\reasoning3_prepare_data_preflight.py --seed 43 --max-length 256 --out experiments\cf_cycle_9\cllora_active_freeze_utility_veto_topk2_seed43_paired\reasoning3_prepare_data_preflight.json --fail-on-invalid
```

Expected preflight result:

- valid true.
- no zero-label batches across reduced reasoning3 tasks.
- same task files as [`REASONING3`](experiments/cf_cycle_1/run_nullspace_ablation.py:32).

## 6. Permission-gated GPU protocol shape, not to be run now

Do not run these commands from experiment-designer mode. They are the exact later protocol shape after code-mode implementation and CPU preflight pass, and only after explicit user GPU approval.

Output root:

- [`experiments/cf_cycle_9/cllora_active_freeze_utility_veto_topk2_seed43_paired`](experiments/cf_cycle_9/cllora_active_freeze_utility_veto_topk2_seed43_paired)

Fresh control arm:

```cmd
python experiments\cf_cycle_1\run_nullspace_ablation.py --arm control --task-suite reasoning3 --seed 43 --epochs 2 --rank 4 --max-length 256 --output-root experiments\cf_cycle_9\cllora_active_freeze_utility_veto_topk2_seed43_paired --vram-threshold-mb 7500 > experiments\cf_cycle_9\cllora_active_freeze_utility_veto_topk2_seed43_paired\control.log 2>&1
```

Control standard gate:

```cmd
python experiments\cf_cycle_4\validate_seed43_arm_gate.py --root experiments\cf_cycle_9\cllora_active_freeze_utility_veto_topk2_seed43_paired --arm control --vram-threshold-mb 7500 > experiments\cf_cycle_9\cllora_active_freeze_utility_veto_topk2_seed43_paired\control_gate.json
```

Fresh treatment arm, only after control gate is valid:

```cmd
python experiments\cf_cycle_1\run_nullspace_ablation.py --arm treatment --task-suite reasoning3 --seed 43 --epochs 2 --rank 4 --max-length 256 --treatment-variant cllora-active-freeze-utility-veto-topk-2 --output-root experiments\cf_cycle_9\cllora_active_freeze_utility_veto_topk2_seed43_paired --vram-threshold-mb 7500 > experiments\cf_cycle_9\cllora_active_freeze_utility_veto_topk2_seed43_paired\treatment.log 2>&1
```

Treatment standard gate:

```cmd
python experiments\cf_cycle_4\validate_seed43_arm_gate.py --root experiments\cf_cycle_9\cllora_active_freeze_utility_veto_topk2_seed43_paired --arm treatment --vram-threshold-mb 7500 > experiments\cf_cycle_9\cllora_active_freeze_utility_veto_topk2_seed43_paired\treatment_gate.json
```

Treatment active/capacity/utility gate:

```cmd
python experiments\cf_cycle_5\validate_active_treatment_gate.py --root experiments\cf_cycle_9\cllora_active_freeze_utility_veto_topk2_seed43_paired --arm treatment --out experiments\cf_cycle_9\cllora_active_freeze_utility_veto_topk2_seed43_paired\treatment_active_gate.json --expected-variant cllora-active-freeze-utility-veto-topk-2 --expected-frozen-basis-variance-threshold 0.05 --expected-frozen-basis-top-k-per-freeze 2 --expected-admission-policy utility_veto --expect-utility-probe
```

Final paired comparison, only after all gates are accepted:

```cmd
python experiments\cf_cycle_1\compare_nullspace_ablation.py --root experiments\cf_cycle_9\cllora_active_freeze_utility_veto_topk2_seed43_paired --vram-threshold-mb 7500 > experiments\cf_cycle_9\cllora_active_freeze_utility_veto_topk2_seed43_paired\comparison.json
```

If the active validator CLI is not extended with `--expected-admission-policy` and `--expect-utility-probe`, use the existing validator flags plus a separate utility-admission validator under [`experiments/cf_cycle_9`](experiments/cf_cycle_9). Do not weaken the active/capacity gate to accommodate the new utility fields.

## 7. Required artifacts and gate expectations

Required artifacts:

- [`reasoning3_prepare_data_preflight.json`](experiments/cf_cycle_9/cllora_active_freeze_utility_veto_topk2_seed43_paired/reasoning3_prepare_data_preflight.json)
- [`control.log`](experiments/cf_cycle_9/cllora_active_freeze_utility_veto_topk2_seed43_paired/control.log)
- [`control/config.json`](experiments/cf_cycle_9/cllora_active_freeze_utility_veto_topk2_seed43_paired/control/config.json)
- [`control/instrumentation.json`](experiments/cf_cycle_9/cllora_active_freeze_utility_veto_topk2_seed43_paired/control/instrumentation.json)
- [`control/metrics.json`](experiments/cf_cycle_9/cllora_active_freeze_utility_veto_topk2_seed43_paired/control/metrics.json)
- [`control/run_status.json`](experiments/cf_cycle_9/cllora_active_freeze_utility_veto_topk2_seed43_paired/control/run_status.json)
- [`control_gate.json`](experiments/cf_cycle_9/cllora_active_freeze_utility_veto_topk2_seed43_paired/control_gate.json)
- [`treatment.log`](experiments/cf_cycle_9/cllora_active_freeze_utility_veto_topk2_seed43_paired/treatment.log)
- [`treatment/config.json`](experiments/cf_cycle_9/cllora_active_freeze_utility_veto_topk2_seed43_paired/treatment/config.json)
- [`treatment/instrumentation.json`](experiments/cf_cycle_9/cllora_active_freeze_utility_veto_topk2_seed43_paired/treatment/instrumentation.json)
- [`treatment/metrics.json`](experiments/cf_cycle_9/cllora_active_freeze_utility_veto_topk2_seed43_paired/treatment/metrics.json)
- [`treatment/run_status.json`](experiments/cf_cycle_9/cllora_active_freeze_utility_veto_topk2_seed43_paired/treatment/run_status.json)
- [`treatment_gate.json`](experiments/cf_cycle_9/cllora_active_freeze_utility_veto_topk2_seed43_paired/treatment_gate.json)
- [`treatment_active_gate.json`](experiments/cf_cycle_9/cllora_active_freeze_utility_veto_topk2_seed43_paired/treatment_active_gate.json)
- [`comparison.json`](experiments/cf_cycle_9/cllora_active_freeze_utility_veto_topk2_seed43_paired/comparison.json)

Hard validity gates:

1. Data preflight valid.
2. Fresh paired control and treatment arms; no copied control.
3. Standard gates pass for both arms.
4. Active/capacity gate passes for treatment.
5. Same-envelope pass: seed 43, reduced reasoning3, rank 4, max length 256, two epochs, same code revision, both arms under 7500 MB.
6. Final comparison produced and valid.

Mechanism gates:

1. If utility gate admits frozen basis columns, active/projection materiality must remain positive: frozen-basis calls greater than zero, removed norm sum positive, removed norm per frozen call finite, active reassignment path greater than zero, and active adjustment norm sum positive.
2. If utility gate admits almost nothing, mark the result as an interpretable failure signal, not success. In that case, preserve the zero-admission utility logs and do not claim mechanism success.
3. Capacity must remain bounded: max frozen columns should remain no greater than 2 per freeze admission event unless the implementation reports a specific unavoidable reason and the result critic accepts it.

## 8. Success and falsification thresholds

Utility success threshold:

- `delta_bwt_points >= 0.75` versus fresh paired control.
- `delta_avg_acc_points >= -0.25`.
- Prefer both old-task gaps nonnegative.
- If one old-task gap remains negative, require clear evidence it materially improves from the CF-cycle-8 negative gap of -0.20283620228340737 points and explain why the remaining tradeoff is acceptable for the next cycle.

Evidence-usable but not successful:

- All hard validity gates pass, active/capacity and same-envelope pass, and final comparison is valid, but utility thresholds are not met.
- Send to result critic with the utility admission logs to decide whether cadence, probe metric, or veto margin is the next bounded lever.

Falsification thresholds:

- Gates pass but `delta_bwt_points < 0.5`.
- `delta_avg_acc_points < -0.25`.
- Old-task gaps remain mixed without improvement in the negative gap relative to CF-cycle-8.
- Active/projection materiality disappears unexpectedly despite admitted frozen basis columns.
- Utility gate admits near-zero candidates and final utility does not improve; label as utility-gate over-filtering or candidate-not-retention-critical, not as treatment success.

## 9. Result labels to use

- Confirmed: hard validity gates pass and utility success thresholds pass.
- Likely: hard validity gates pass, BWT is at least 0.5 points, average accuracy is no worse than -0.25 points, and old-task gaps improve but do not fully clear success.
- Plausible: mechanism logs are coherent but utility improvement is below 0.5 points or mixed.
- Weak signal: one metric improves but old-task gaps remain mixed or probe admission is too sparse.
- Contradicted: hard gates pass and falsification thresholds trigger.
- Blocked: implementation, preflight, GPU guardrail, or artifact validity prevents comparison.
- Skipped: any explicitly omitted check, with reason.

## 10. Recommended next action and handoff

Recommended handoff target: `code` mode, then `llm-result-critic` after evidence collection.

Code-mode acceptance criteria:

1. Implement only the new utility-veto top-k treatment variant and CPU tests/preflight helpers.
2. Preserve CF-cycle-8 behavior for existing variants.
3. Do not run GPU commands without explicit user approval.
4. Produce a short implementation note listing changed files and CPU-only check results.
5. Stop before GPU execution unless the user explicitly grants permission.

After permission-gated GPU execution, hand off to `llm-result-critic` with:

- experiments_run
- commands_or_actions
- files_changed_or_inspected
- artifacts_and_outputs
- hypothesis_status_map
- observed facts separated from interpretations
- skipped or blocked checks and reasons

## 11. Checks not run in this design step

No shell commands, tests, GPU jobs, or algorithm-code modifications were run in this experiment-design step. The only intended output is this bounded design packet.

## 12. Task record

I inspected the CF-cycle-8 report and the relevant config, adapter, runner, active-gate validator, and comparison files to anchor this protocol in the existing implementation. I wrote this packet under [`experiments/cf_cycle_9`](experiments/cf_cycle_9) because the task requested a design-only CF-cycle-9 handoff and prohibited GPU execution or algorithm-code changes from experiment-designer mode.
