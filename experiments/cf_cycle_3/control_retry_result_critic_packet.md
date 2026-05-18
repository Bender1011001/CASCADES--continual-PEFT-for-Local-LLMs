# CF-cycle-3 Result Critic Packet — reduced-memory reasoning3 control retry

## 1. Claim reviewed

The reduced-memory `reasoning3` control retry after the data-loader all-ignored-label fix is acceptable control-gate evidence: the fixed data path removes the diagnosed zero-supervision failure mode for the bounded protocol, and the control arm completed with finite artifacts under the 7500 MB VRAM threshold. This would permit a treatment-only retry in the same new output root, but does not support any catastrophic-forgetting or null-space-mitigation claim yet.

## 2. Evidence quality

**Moderate to strong for the bounded control-gate decision.** Evidence is stronger than CF-cycle-2 because it combines root-cause diagnosis, targeted data-loader tests, all-task CPU preflight, a completed GPU control run, and a fresh post-run validator.

**Unavailable for treatment projection and algorithmic mitigation.** Treatment was intentionally not launched, so projection activation, comparison validity, BWT improvement, and catastrophic-forgetting claims remain untested.

## 3. Main support

- The original failure mechanism is well supported: Task 0 batch 93 previously mapped to a sample whose prompt alone exceeded max length, producing all ignored labels and NaN mean cross-entropy in [`experiments/cf_cycle_3/root_cause_packet.md`](experiments/cf_cycle_3/root_cause_packet.md:31) and [`experiments/cf_cycle_3/root_cause_packet.md`](experiments/cf_cycle_3/root_cause_packet.md:54).
- The fixed supervised sequence construction now reserves response-label budget and rejects zero-supervision samples in [`python._build_supervised_sequence()`](cascades/data.py:80), with [`python.prepare_data()`](cascades/data.py:107) using that helper for the training loader.
- The code-mode tests include the targeted no-all-ignored regression and synthetic long-prompt regression in [`tests/test_data.py`](tests/test_data.py:147), and the handoff records 14 passing data tests in [`experiments/cf_cycle_3/control_retry_handoff_packet.md`](experiments/cf_cycle_3/control_retry_handoff_packet.md:27).
- The targeted real-tokenizer verification shows the former batch-93 failure point now has one valid label, no all-masked labels, and zero zero-label batches in [`experiments/cf_cycle_3/batch93_static_probe.verify.json`](experiments/cf_cycle_3/batch93_static_probe.verify.json:1).
- The full `reasoning3` preflight scanned all three tasks at seed 42 and max length 256 and found `valid=true`, zero zero-label batches, and max sequence length 256 in [`experiments/cf_cycle_3/reasoning3_prepare_data_preflight.json`](experiments/cf_cycle_3/reasoning3_prepare_data_preflight.json:1).
- The retried control run completed with the expected bounded parameters in [`experiments/cf_cycle_3/nullspace_ablation_retry/control/run_status.json`](experiments/cf_cycle_3/nullspace_ablation_retry/control/run_status.json:1): control arm, `reasoning3`, seed 42, rank 4, max length 256, epochs 2, nonfinite disallowed, and VRAM-over-threshold disallowed.
- The retried control stayed under the VRAM threshold: peak 6848.810546875 MB versus threshold 7500 MB in [`experiments/cf_cycle_3/nullspace_ablation_retry/control/run_status.json`](experiments/cf_cycle_3/nullspace_ablation_retry/control/run_status.json:13).
- Metrics are finite and have the expected shape for a completed three-task control: average accuracy 0.5652729740720018, BWT -0.03614532252823527, three final accuracies, and two old-task deltas in [`experiments/cf_cycle_3/nullspace_ablation_retry/control/metrics.json`](experiments/cf_cycle_3/nullspace_ablation_retry/control/metrics.json:1).
- The manifest exactly matches the intended reduced three-task suite and excludes Digital Twin in [`experiments/cf_cycle_3/nullspace_ablation_retry/control/task_manifest.json`](experiments/cf_cycle_3/nullspace_ablation_retry/control/task_manifest.json:1).
- The post-run validator reports `valid=true`, required artifacts present, no status or metrics parameter mismatches, finite metrics/instrumentation/matrix, a 3 by 3 accuracy matrix, and under-threshold VRAM in [`experiments/cf_cycle_3/control_gate_validation.json`](experiments/cf_cycle_3/control_gate_validation.json:1).
- The control log independently shows the run reached final metrics, saved results, ran diagnostics, and saved weights in [`experiments/cf_cycle_3/nullspace_ablation_retry/control.log`](experiments/cf_cycle_3/nullspace_ablation_retry/control.log:167) and [`experiments/cf_cycle_3/nullspace_ablation_retry/control.log`](experiments/cf_cycle_3/nullspace_ablation_retry/control.log:311).

## 4. Main confounders or failure modes

- This is one seed, one reduced three-task suite, one model, and one run. It is enough for the bounded control gate; it is not broad reproducibility evidence.
- The data-loader fix changes truncation behavior by preserving response supervision. That is acceptable because treatment should run under the same fixed code, but metrics should not be compared to old failed CF-cycle-2 artifacts as algorithmic evidence.
- The artifact `git_revision` remains `7c4e01d` in both the failed and retried control artifacts. If the data-loader fix was uncommitted, source provenance is not fully captured by the revision alone. The next GPU step should avoid additional code changes before treatment or record a dirty-tree/file-hash note.
- The custom validator is appropriate for the control gate, but it does not parse every CSV field and does not itself prove the shell command exit code. The control log, status, metrics, and user-reported fresh validation exit 0 cover the practical decision.
- The control log still shows nonfatal D-MoLE variance warnings in [`experiments/cf_cycle_3/nullspace_ablation_retry/control.log`](experiments/cf_cycle_3/nullspace_ablation_retry/control.log:44) and [`experiments/cf_cycle_3/nullspace_ablation_retry/control.log`](experiments/cf_cycle_3/nullspace_ablation_retry/control.log:110). They did not invalidate this run but should be monitored if treatment fails.
- Treatment-specific projection evidence is absent by design. Active frozen-basis and reassignment evidence remains a hard treatment-gate requirement under [`experiments/cf_cycle_2/EXPERIMENT_PROTOCOL.md`](experiments/cf_cycle_2/EXPERIMENT_PROTOCOL.md:93) and [`python.compare_runs()`](experiments/cf_cycle_1/compare_nullspace_ablation.py:105).
- Comparison is invalid until treatment artifacts exist. The comparison script requires both control and treatment arms in [`python.compare_runs()`](experiments/cf_cycle_1/compare_nullspace_ablation.py:112), so running comparison now would only test missing-artifact rejection.

## 5. Corrected conclusion

The control gate is acceptable for the bounded CF-cycle-3 decision. The fixed data-loader path is strongly supported as sufficient to remove the diagnosed all-ignored-label failure for the `reasoning3`, seed 42, rank 4, max length 256, two-epoch control protocol. The completed control can serve as the baseline artifact for the next treatment-only run in the same output root.

This evidence does not show that catastrophic forgetting is solved, does not show that frozen null-space projection is active, and does not show that treatment improves BWT or old-task retention. It only unblocks the next sequential GPU step.

Evidence labels:

- **Confirmed:** the prior all-ignored-label control failure was fixed for the bounded `reasoning3` control retry.
- **Confirmed:** the retried control completed, produced finite control artifacts, matched expected parameters/manifests, and stayed below 7500 MB peak VRAM.
- **Likely:** treatment can now be run under the same fixed data path without immediately hitting the same batch-93 zero-label failure.
- **Plausible:** treatment may still fail due to projection/harness behavior, VRAM, D-MoLE/SVC dynamics, or unrelated nonfinite loss.
- **Contradicted:** any claim that catastrophic forgetting is solved or that null-space treatment works based only on this control retry.

## 6. Confidence level

**High** for accepting the control gate and allowing the next sequential treatment-only retry.

**Medium-high** that the specific data-loader all-ignored-label failure mode is fixed for this reduced-memory `reasoning3` protocol.

**Low/unknown** for treatment behavior, projection activation, and algorithmic mitigation claims.

## 7. Proceed / revise / retest / abandon / blocked

**Proceed** to treatment-only execution in the same output root, with no comparison until the treatment gate passes.

Do **not** run `--arm both`; the control is already valid and should not be overwritten or rerun unnecessarily. Do **not** compare before treatment artifacts exist. Do **not** route to debug/code unless treatment fails a guardrail, lacks projection evidence, exceeds VRAM, or produces nonfinite artifacts.

## 8. Cheapest next action

Run only the treatment arm against the same fixed code and output root:

```cmd
python experiments\cf_cycle_1\run_nullspace_ablation.py --arm treatment --task-suite reasoning3 --seed 42 --epochs 2 --rank 4 --max-length 256 --output-root experiments\cf_cycle_3\nullspace_ablation_retry --vram-threshold-mb 7500 > experiments\cf_cycle_3\nullspace_ablation_retry\treatment.log 2>&1
```

Then validate the treatment gate before comparison. The treatment gate must check completed status, finite metrics/instrumentation, matching `reasoning3` parameters/manifests, peak VRAM at or below 7500 MB, non-empty frozen basis evidence, positive frozen-basis reassignment calls, and positive removed norm. Only after that should comparison run:

```cmd
python experiments\cf_cycle_1\compare_nullspace_ablation.py --root experiments\cf_cycle_3\nullspace_ablation_retry --vram-threshold-mb 7500 > experiments\cf_cycle_3\nullspace_ablation_retry\comparison.json
```

## evidence_quality

Moderate to strong for the control-gate decision; unavailable for treatment and comparison claims.

## hypothesis_verdicts

- H1 fixed data loader removes the all-ignored-label failure in the bounded control retry: **supported / confirmed for this protocol**.
- H2 reduced-memory `reasoning3` control can complete with finite artifacts under 7500 MB: **supported / confirmed for this seed and run**.
- H3 treatment projection behavior is active and comparable: **not addressed**.
- H4 null-space treatment improves forgetting metrics: **not addressed**.
- H5 catastrophic forgetting is solved: **contradicted as an allowed conclusion from current evidence**.

## confounders_and_gaps

- One-run, one-seed, reduced-suite evidence only.
- Source provenance depends on working-tree state as well as `git_revision`.
- Validator is sufficient for control gating but not a general artifact auditor.
- Treatment projection evidence and comparison are absent.
- Nonfatal D-MoLE warnings should be monitored but are not a control-gate blocker.

## recommended_next_cycle_seed

Seed the next executable step as treatment-only GPU execution under the same fixed code and output root, followed by a treatment-gate validator. If treatment passes, run comparison. If treatment fails, route to debug/code with the treatment log, treatment artifacts, and control baseline preserved.

## handoff_target

`llm-report-writer`
