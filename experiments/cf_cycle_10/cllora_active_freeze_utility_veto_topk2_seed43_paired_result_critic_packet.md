# CF-cycle-10 Result Critic Packet — structural-proxy utility-veto active-freeze top-k 2

## 1. Claim reviewed

The reviewed claim is that the structural-proxy utility-veto active-freeze treatment cllora-active-freeze-utility-veto-topk-2 should be promoted after a fresh paired reduced-suite GPU run under reduced reasoning3, seed 43, rank 4, max length 256, two epochs, and a 7500 MB VRAM threshold.

## 2. Evidence quality

**Verdict: moderate for the bounded same-envelope reduced-suite decision; contradictory for promotion.**

The evidence is fresh, paired, same-seed, same-revision, and gate-valid for this exact reduced reasoning3 envelope. It is not broad-suite evidence and it is not a true held-out old-task utility probe.

## 3. Main support

- Data preflight was valid: [`reasoning3_prepare_data_preflight.json`](experiments/cf_cycle_10/cllora_active_freeze_utility_veto_topk2_seed43_paired/reasoning3_prepare_data_preflight.json:6) reports valid true, three tasks, zero zero-label batches, and max sequence length 256.
- Both arms completed in the same envelope and code revision:
  - [`control/run_status.json`](experiments/cf_cycle_10/cllora_active_freeze_utility_veto_topk2_seed43_paired/control/run_status.json:3) reports completed status, seed 43, rank 4, two epochs, git revision 0b1266a, and peak VRAM 6843.90966796875 MB.
  - [`treatment/run_status.json`](experiments/cf_cycle_10/cllora_active_freeze_utility_veto_topk2_seed43_paired/treatment/run_status.json:3) reports completed status, the same run envelope, git revision 0b1266a, and peak VRAM 6843.90966796875 MB.
- Standard gates passed:
  - [`control_gate.json`](experiments/cf_cycle_10/cllora_active_freeze_utility_veto_topk2_seed43_paired/control_gate.json:2) reports valid true.
  - [`treatment_gate.json`](experiments/cf_cycle_10/cllora_active_freeze_utility_veto_topk2_seed43_paired/treatment_gate.json:2) reports valid true, treatment projection required and valid.
- Active/capacity/utility instrumentation passed:
  - [`treatment_active_gate.json`](experiments/cf_cycle_10/cllora_active_freeze_utility_veto_topk2_seed43_paired/treatment_active_gate.json:2) reports valid true.
  - [`treatment_active_gate.json`](experiments/cf_cycle_10/cllora_active_freeze_utility_veto_topk2_seed43_paired/treatment_active_gate.json:27) confirms the intended variant, threshold 0.05, top-k 2, admission policy utility_veto, and utility probe enabled.
  - [`treatment_active_gate.json`](experiments/cf_cycle_10/cllora_active_freeze_utility_veto_topk2_seed43_paired/treatment_active_gate.json:39) reports 3600 calls with frozen basis, max frozen cols 2, removed norm per frozen call 0.0009696987048669649, 6900 active reassignment path calls, and active adjustment norm sum 50.43182728852844.
  - [`treatment_active_gate.json`](experiments/cf_cycle_10/cllora_active_freeze_utility_veto_topk2_seed43_paired/treatment_active_gate.json:66) reports the utility-veto path was exercised: 38 utility-probed freeze calls, 84 candidates considered, 24 admitted, 45 vetoed, 6 zero-admission freeze calls, and per-old-task veto counts on task 0.

## 4. Main confounders or failure modes

- The utility signal is a structural proxy, not a true held-out old-task utility probe. It can veto candidates without measuring real old-task loss/accuracy.
- The run is only one seed and one reduced three-task suite. It is enough for a bounded engineering decision, but not enough for broad generalization.
- The result may depend on the reduced reasoning3 task mix, short two-epoch budget, rank 4, max length 256, and current control variance.
- Baseline Windows GPU memory was high during preflight, but this is not decision-changing because both arm statuses stayed under the 7500 MB threshold.
- Mechanism strength is real but not sufficient: strong projection and active-reassignment evidence does not imply useful old-task retention.
- The proxy appears possibly over-restrictive or misaligned: 45 of 84 candidates were vetoed, six freeze calls admitted nothing, mean utility delta sum was negative, and final task-level gaps were mixed.

## 5. Corrected conclusion

Promotion is contradicted. The treatment executed correctly and materially exercised the intended active-freeze plus structural utility-veto mechanism, but the fresh paired comparison reports continue false, BWT delta -0.11410674934247411 points, average-accuracy delta -0.039064248456699424 points, and mixed old-task gaps [0.38962788760765754, -0.6178413862926058] in [`comparison.json`](experiments/cf_cycle_10/cllora_active_freeze_utility_veto_topk2_seed43_paired/comparison.json:2). This weakens the utility-veto proxy hypothesis rather than supporting promotion.

The most likely interpretation is not "active freezing failed to run"; it did run. The more useful conclusion is that this structural proxy utility-veto admission rule did not translate mechanism activity into improved reduced-suite backward transfer under the tested envelope.

## 6. Confidence level

**Medium-high** for the narrow decision not to promote this exact variant under this exact reduced-suite envelope.

**Medium** that the specific proxy/veto threshold is misaligned or too blunt.

**Low** for claims about full-suite, other seeds, v10/Digital Twin, or true held-out utility behavior because those checks were intentionally not run.

## 7. Proceed / revise / retest / abandon / blocked

**Decision: revise, with a bias toward abandoning this proxy-veto formulation unless the next cycle can replace or calibrate the utility signal cheaply. Do not promote. Do not escalate to broad suites.**

Rationale:

- Continue false and negative BWT delta directly contradict promotion.
- Instrumentation is strong enough to reject the no-op explanation.
- A simple repeat of the same variant is unlikely to be the cheapest uncertainty reducer.
- The next useful step is a bounded hypothesis/design cycle focused on proxy misalignment, not another full run of the unchanged treatment.

## 8. Cheapest next action

Seed the coordinator with a bounded analysis/design objective:

> CF-cycle-11 should diagnose why the structural proxy utility-veto admitted/vetoed directions did not improve BWT despite strong mechanism activity. Prefer CPU/artifact-level analysis of freeze events and proxy alignment first; if a new GPU run is later proposed, test exactly one revised admission policy or threshold schedule under the same reduced reasoning3, seed 43 envelope with explicit user approval. Candidate revisions: replace all-or-nothing zero-drop veto with a tolerance/margin, add task-balanced veto accounting so task 0 does not dominate, compare against salience/top-k-only admission using stored instrumentation, or pivot to a different bottleneck such as projection timing/cadence if proxy diagnostics show no plausible calibration path.

## Explicit non-claims

- This does not show catastrophic forgetting is solved.
- This does not validate broad-suite performance.
- This does not validate full `current4`, v10, Digital Twin, or generative-subset behavior.
- This does not prove all utility-aware freezing is bad.
- This does not prove true held-out old-task utility probing would fail.
- This does not justify promotion or larger GPU escalation for cllora-active-freeze-utility-veto-topk-2 as-is.

## Handoff target

handoff target: llm-report-writer

The report writer should preserve the contradiction cleanly: fresh paired evidence is usable, the mechanism and instrumentation passed, but the final comparison argues against promotion and toward a bounded revision/pivot cycle.
