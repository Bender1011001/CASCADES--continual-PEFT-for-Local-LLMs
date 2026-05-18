# CF-cycle-7 top-k frozen-basis treatment-only result critic packet

Handoff target: `llm-result-critic`

## 1. Hypothesis tested

The `cllora-active-freeze-topk-2` treatment, which preserves CL-LoRA-active flags and admits at most the two strongest frozen-basis directions per freeze after the 0.05 threshold/fallback rule, should increase useful frozen-basis mechanism strength and improve the reduced `reasoning3` seed-43 copied-control comparison without exceeding the 7500 MB VRAM guardrail.

## 2. Experiment performed

Bounded treatment-only GPU experiment, using the previously valid seed-43 control copied from [`experiments/cf_cycle_4/nullspace_ablation_seed43/control`](../cf_cycle_4/nullspace_ablation_seed43/control). No full current4, v10, Digital Twin, generative subset, broad fuzzer, paired fallback, cadence change, D-MoLE timing change, or rank change was launched.

## 3. Files changed or generated

- Generated/copy-refreshed experiment root: [`experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43`](cllora_active_freeze_topk2_seed43)
- Copied control artifacts: [`experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/control`](cllora_active_freeze_topk2_seed43/control)
- Treatment artifacts: [`experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/treatment`](cllora_active_freeze_topk2_seed43/treatment)
- Standard treatment gate: [`experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/treatment_gate.json`](cllora_active_freeze_topk2_seed43/treatment_gate.json)
- Active/capacity gate: [`experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/treatment_active_gate.json`](cllora_active_freeze_topk2_seed43/treatment_active_gate.json)
- Comparison: [`experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43/comparison.json`](cllora_active_freeze_topk2_seed43/comparison.json)
- This packet: [`experiments/cf_cycle_7/cllora_active_freeze_topk2_result_critic_packet.md`](cllora_active_freeze_topk2_result_critic_packet.md)
- Updated project context: [`CONTEXT.md`](../../CONTEXT.md)

No algorithm code was changed during this execution step.

## 4. Commands/checks run

```cmd
if not exist experiments\cf_cycle_7\cllora_active_freeze_topk2_seed43 mkdir experiments\cf_cycle_7\cllora_active_freeze_topk2_seed43 && xcopy /E /I /Y experiments\cf_cycle_4\nullspace_ablation_seed43\control experiments\cf_cycle_7\cllora_active_freeze_topk2_seed43\control
```

Exit code: 0.

```cmd
python experiments\cf_cycle_1\run_nullspace_ablation.py --arm treatment --task-suite reasoning3 --seed 43 --epochs 2 --rank 4 --max-length 256 --output-root experiments\cf_cycle_7\cllora_active_freeze_topk2_seed43 --treatment-variant cllora-active-freeze-topk-2 --vram-threshold-mb 7500 > experiments\cf_cycle_7\cllora_active_freeze_topk2_seed43\treatment.log 2>&1
```

Exit code: 0.

```cmd
python experiments\cf_cycle_4\validate_seed43_arm_gate.py --root experiments\cf_cycle_7\cllora_active_freeze_topk2_seed43 --arm treatment --vram-threshold-mb 7500 > experiments\cf_cycle_7\cllora_active_freeze_topk2_seed43\treatment_gate.json
```

Exit code: 0.

```cmd
python experiments\cf_cycle_5\validate_active_treatment_gate.py --root experiments\cf_cycle_7\cllora_active_freeze_topk2_seed43 --arm treatment --expected-variant cllora-active-freeze-topk-2 --expected-frozen-basis-variance-threshold 0.05 --expected-frozen-basis-top-k-per-freeze 2 --out experiments\cf_cycle_7\cllora_active_freeze_topk2_seed43\treatment_active_gate.json
```

Exit code: 0.

```cmd
xcopy /E /I /Y experiments\cf_cycle_4\nullspace_ablation_seed43\control experiments\cf_cycle_7\cllora_active_freeze_topk2_seed43\control && python experiments\cf_cycle_1\compare_nullspace_ablation.py --root experiments\cf_cycle_7\cllora_active_freeze_topk2_seed43 --vram-threshold-mb 7500 > experiments\cf_cycle_7\cllora_active_freeze_topk2_seed43\comparison.json
```

Exit code: 0; xcopy reported 8 files copied.

Fresh verification before completion:

```cmd
python -c "import json, pathlib, math; root=pathlib.Path(r'experiments\cf_cycle_7\cllora_active_freeze_topk2_seed43'); paths=['control/metrics.json','control/run_status.json','treatment/config.json','treatment/run_status.json','treatment/metrics.json','treatment/instrumentation.json','treatment_gate.json','treatment_active_gate.json','comparison.json','treatment.log']; missing=[p for p in paths if not (root/p).exists()]; tcfg=json.loads((root/'treatment/config.json').read_text()); trun=json.loads((root/'treatment/run_status.json').read_text()); tmet=json.loads((root/'treatment/metrics.json').read_text()); inst=json.loads((root/'treatment/instrumentation.json').read_text()); gate=json.loads((root/'treatment_gate.json').read_text()); agate=json.loads((root/'treatment_active_gate.json').read_text()); comp=json.loads((root/'comparison.json').read_text()); ctrl=json.loads((root/'control/metrics.json').read_text()); checks={'missing':missing,'run_completed':trun.get('status')=='completed','standard_gate_valid':gate.get('valid') is True,'active_gate_valid':agate.get('valid') is True,'comparison_valid':comp.get('valid') is True,'variant':tcfg.get('treatment_variant'),'top_k':tcfg.get('ablation_config',{}).get('frozen_basis_top_k_per_freeze'),'threshold':tcfg.get('ablation_config',{}).get('frozen_basis_variance_threshold'),'peak_vram':trun.get('peak_vram_mb'),'under_vram':trun.get('peak_vram_mb',999999) <= 7500,'removed_norm_per_frozen_call':inst.get('reassign',{}).get('removed_norm_per_frozen_call'),'active_path_calls':inst.get('reassign',{}).get('calls_with_active_reassign_path'),'frozen_basis_calls':inst.get('reassign',{}).get('calls_with_frozen_basis'),'delta_bwt_points':comp.get('delta_bwt_points'),'delta_avg_acc_points':comp.get('delta_avg_acc_points'),'old_task_delta_gaps_points':comp.get('old_task_delta_gaps_points'),'continue':comp.get('continue'),'treatment_avg_acc':tmet.get('avg_acc'),'control_avg_acc':ctrl.get('avg_acc'),'treatment_bwt':tmet.get('bwt'),'control_bwt':ctrl.get('bwt')}; print(json.dumps(checks, indent=2)); assert not missing; assert checks['run_completed']; assert checks['standard_gate_valid']; assert checks['active_gate_valid']; assert checks['comparison_valid']; assert checks['variant']=='cllora-active-freeze-topk-2'; assert checks['top_k']==2; assert abs(checks['threshold']-0.05)<1e-12; assert checks['under_vram']; assert checks['removed_norm_per_frozen_call']>0.00076231115494036; assert checks['active_path_calls']>0; assert checks['frozen_basis_calls']>0"
```

Exit code: 0.

## 5. Raw or summarized results

### Observed facts

- Config: [`treatment/config.json`](cllora_active_freeze_topk2_seed43/treatment/config.json) records `treatment_variant="cllora-active-freeze-topk-2"`, `frozen_basis_variance_threshold=0.05`, and `frozen_basis_top_k_per_freeze=2`.
- Run status: [`treatment/run_status.json`](cllora_active_freeze_topk2_seed43/treatment/run_status.json) reports `status="completed"`, peak VRAM 6843.90966796875 MB, and wall time 1066.974350452423 seconds.
- Treatment metrics: [`treatment/metrics.json`](cllora_active_freeze_topk2_seed43/treatment/metrics.json) reports average accuracy 0.5692412473762564, BWT -0.02357593129097782, final accuracies `[0.39373799242837565, 0.5480484789920578, 0.7659372707083357]`, diagonal accuracies `[0.4319063638056756, 0.5570319701967135, 0.7659372707083357]`, and old-task deltas `[-0.03816837137729995, -0.008983491204655691]`.
- Copied-control metrics: [`control/metrics.json`](cllora_active_freeze_topk2_seed43/control/metrics.json) reports average accuracy 0.5665482637215719, BWT -0.026559348750060024, and old-task deltas `[-0.04616356831829843, -0.006955129181821618]`.
- Standard gate: [`treatment_gate.json`](cllora_active_freeze_topk2_seed43/treatment_gate.json) reports `valid=true`, required artifacts present, finite metrics/instrumentation, finite 3x3 accuracy matrix, peak VRAM below 7500 MB, projection active, frozen basis nonempty, 3600 calls with frozen basis, removed norm sum 3.530579238988139, and 38 freeze events.
- Active/capacity gate: [`treatment_active_gate.json`](cllora_active_freeze_topk2_seed43/treatment_active_gate.json) reports `valid=true`, all config checks true, all instrumentation checks true, active adjustment positive, 3600 frozen-basis calls, max frozen columns 2, removed norm sum 3.530579238988139, removed norm per frozen call 0.000980716455274483, 6900 active reassignment path calls, active adjustment norm sum 50.596070232306374, and active adjustment norm max 0.12196692824363708.
- Mechanism thresholds: removed norm per frozen call 0.000980716455274483 is above CF-cycle-6 baseline 0.0005073194128438748, CF-cycle-5 baseline 0.000609848923952288, and material-strength reference 0.00076231115494036.
- Comparison: [`comparison.json`](cllora_active_freeze_topk2_seed43/comparison.json) reports `valid=true`, BWT delta +0.2983417459082205 points, average-accuracy delta +0.2692983654684511 points, old-task delta gaps `[0.7995196940998484, -0.20283620228340737]`, projection active true, frozen basis nonempty true, and `continue=false`.

### Interpretations

- Mechanism activation is confirmed for this bounded treatment: active path calls are positive, frozen-basis calls are positive, removed norm sum is positive, and normalized removed norm exceeds all specified reference thresholds.
- Utility signal is positive but still bounded: the copied-control comparison improves BWT and average accuracy on the reduced `reasoning3` proxy, but `continue=false` and one old-task gap is negative.
- This result is not a catastrophic-forgetting solution claim and not a promotion to full-suite escalation by itself.

## 6. Result label

Likely useful, not success.

Rationale: valid run, valid gates, strengthened mechanism, and positive BWT/average-accuracy deltas make the top-k treatment more useful than the CF-cycle-6 threshold-only variant, but the comparison gate still sets `continue=false`, the absolute BWT remains negative, old-task gaps are mixed, and this is one bounded reduced-suite treatment-only smoke/proxy run.

## 7. Confidence level

Medium-high for execution/gate validity and mechanism-strength facts.

Medium for usefulness, because it is one seed, a copied-control comparison, and a reduced three-task proxy.

High that catastrophic forgetting is not solved by this evidence alone.

## 8. Recommended next action

Send this packet to `llm-result-critic` to assess whether the strengthened mechanism plus positive copied-control deltas justify the next coordinator cycle. Do not launch paired fallback, full current4, v10, Digital Twin, generative subset, broad fuzzer, cadence changes, D-MoLE timing changes, or rank changes from this mode without coordination.

## Output contract fields

### experiments_run

- One bounded treatment-only GPU run: `cllora-active-freeze-topk-2`, `reasoning3`, seed 43, epochs 2, rank 4, max length 256, VRAM threshold 7500 MB.

### commands_or_actions

- Copied seed-43 control from CF-cycle-4.
- Ran treatment only.
- Ran standard treatment gate.
- Ran active/capacity gate.
- Re-copied control and ran comparison.
- Ran fresh verification assertion over required artifacts and mechanism thresholds.

### files_changed_or_inspected

- Generated/copied artifacts under [`experiments/cf_cycle_7/cllora_active_freeze_topk2_seed43`](cllora_active_freeze_topk2_seed43).
- Inspected treatment config, run status, metrics, instrumentation, gates, comparison, and copied-control metrics.
- Updated [`CONTEXT.md`](../../CONTEXT.md) and wrote this packet.

### artifacts_and_outputs

- [`treatment.log`](cllora_active_freeze_topk2_seed43/treatment.log)
- [`treatment/config.json`](cllora_active_freeze_topk2_seed43/treatment/config.json)
- [`treatment/run_status.json`](cllora_active_freeze_topk2_seed43/treatment/run_status.json)
- [`treatment/metrics.json`](cllora_active_freeze_topk2_seed43/treatment/metrics.json)
- [`treatment/instrumentation.json`](cllora_active_freeze_topk2_seed43/treatment/instrumentation.json)
- [`treatment_gate.json`](cllora_active_freeze_topk2_seed43/treatment_gate.json)
- [`treatment_active_gate.json`](cllora_active_freeze_topk2_seed43/treatment_active_gate.json)
- [`comparison.json`](cllora_active_freeze_topk2_seed43/comparison.json)

### hypothesis_status_map

- Top-k config serialized correctly: confirmed.
- Treatment completed under VRAM guardrail: confirmed.
- Frozen-basis path active: confirmed.
- Active reassignment path active: confirmed.
- Mechanism strength above CF-cycle-6, CF-cycle-5, and material thresholds: confirmed.
- Copied-control BWT and average-accuracy deltas positive: confirmed.
- Success/promote condition: contradicted by `continue=false` and mixed old-task gaps.
- Catastrophic forgetting solved: contradicted/unsupported.

### handoff_target

`llm-result-critic`
