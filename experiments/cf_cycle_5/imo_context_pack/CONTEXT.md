---
project: CASCADES--continual-PEFT-for-Local-LLMs
status: cf-cycle-4 report closed; seed-43 reduced-memory reasoning3 replication confirmed feasibility and active projection but contradicted algorithmic success; pivot CF-cycle-5 to treatment-strength hypothesis redesign before escalation
updated: 2026-05-17
---

# CASCADES Digital Twin Pipeline

## Resume
- **Pick up at**: LLM Loop Coordinator should start CF-cycle-5 from the closeout report [`experiments/cf_cycle_4/REPORT.md`](experiments/cf_cycle_4/REPORT.md:1).
- **Last session**: Report Writer closed CF-cycle-4 after the result-critic subtask was interrupted, using the verdict-ready seed-43 packet [`experiments/cf_cycle_4/seed43_result_critic_packet.md`](experiments/cf_cycle_4/seed43_result_critic_packet.md:1).
- **Decision seed**: [`experiments/cf_cycle_4/nullspace_ablation_seed43/comparison.json`](experiments/cf_cycle_4/nullspace_ablation_seed43/comparison.json:1) is valid with active projection and under-threshold VRAM, but delta BWT is -0.008929480489536235 points and continue is false, so seed 43 misses the plus 1.5 BWT-point threshold after seed 42 also missed it.
- **Blocked on**: Full current4, v10, and Digital Twin escalation remain blocked; CF-cycle-5 should pivot to revised treatment-strength hypotheses before larger GPU runs. Catastrophic forgetting is not solved.

## Recent Work

### 2026-05-17 — CF-cycle-4 report writer closeout
- Wrote durable closeout report [`experiments/cf_cycle_4/REPORT.md`](experiments/cf_cycle_4/REPORT.md:1) because the result-critic tool call was interrupted but the seed-43 packet already contained verdict-ready evidence.
- Preserved the confirmed feasibility evidence: [`experiments/cf_cycle_4/nullspace_ablation_seed43/reasoning3_prepare_data_preflight.json`](experiments/cf_cycle_4/nullspace_ablation_seed43/reasoning3_prepare_data_preflight.json:1) reports valid true with zero zero-label batches; [`experiments/cf_cycle_4/nullspace_ablation_seed43/control/run_status.json`](experiments/cf_cycle_4/nullspace_ablation_seed43/control/run_status.json:1) and [`experiments/cf_cycle_4/nullspace_ablation_seed43/treatment/run_status.json`](experiments/cf_cycle_4/nullspace_ablation_seed43/treatment/run_status.json:1) both report completed status and peak VRAM 6843.90966796875 MB.
- Preserved active-projection evidence: [`experiments/cf_cycle_4/nullspace_ablation_seed43/treatment_gate.json`](experiments/cf_cycle_4/nullspace_ablation_seed43/treatment_gate.json:1) reports valid true, active projection, non-empty frozen basis, 3600 calls with frozen basis, and removed norm sum 3.100088352795865.
- Recorded the contradicted algorithmic result: [`experiments/cf_cycle_4/nullspace_ablation_seed43/comparison.json`](experiments/cf_cycle_4/nullspace_ablation_seed43/comparison.json:1) reports valid true, delta BWT -0.008929480489536235 points, delta average accuracy -0.008046083930968173 points, and continue false.
- Decision: revise. Combined with seed 42 in [`experiments/cf_cycle_3/REPORT.md`](experiments/cf_cycle_3/REPORT.md:1), treatment effect is weak or mixed and below threshold across two reduced-memory seeds; do not run current4, v10, or Digital Twin yet.
- Handoff to LLM Loop Coordinator: CF-cycle-5 should go to LLM Research Planner or LLM Hypothesis Generator for treatment-strength redesign. Candidate areas include CL-LoRA reassignment, earlier or stricter freeze, disabling D-MoLE migration as a confound, adjusting frozen-basis capacity or threshold, and only then considering a default v10 or full null-space stack comparison.
- Non-negotiable carry-forward: catastrophic forgetting is not solved.

### 2026-05-17 — CF-cycle-4 experiment-design seed-43 replication
- Wrote the concise seed-43 protocol to [`experiments/cf_cycle_4/EXPERIMENT_PROTOCOL.md`](experiments/cf_cycle_4/EXPERIMENT_PROTOCOL.md:1) because CF-cycle-3 requested one more reduced-memory `reasoning3` replication before escalation.
- Added [`experiments/cf_cycle_4/validate_seed43_arm_gate.py`](experiments/cf_cycle_4/validate_seed43_arm_gate.py:1) because seed 43 needed explicit control/treatment gate artifacts under the same finite, manifest, VRAM, and projection checks as CF-cycle-3.
- Ran CPU/data/GPU preflight: bytecode compile plus [`tests/test_data.py`](tests/test_data.py:1) and [`tests/test_cf_cycle_2_guardrails.py`](tests/test_cf_cycle_2_guardrails.py:1) passed 19 tests; [`experiments/cf_cycle_4/nullspace_ablation_seed43/reasoning3_prepare_data_preflight.json`](experiments/cf_cycle_4/nullspace_ablation_seed43/reasoning3_prepare_data_preflight.json:1) reports seed 43 valid with zero zero-label batches; [`experiments/cf_cycle_4/nullspace_ablation_seed43/gpu_preflight.csv`](experiments/cf_cycle_4/nullspace_ablation_seed43/gpu_preflight.csv:1) showed the RTX 4060 Ti idle enough to proceed.
- Completed sequential control first, no `--arm both`: [`experiments/cf_cycle_4/nullspace_ablation_seed43/control/run_status.json`](experiments/cf_cycle_4/nullspace_ablation_seed43/control/run_status.json:1) reports completed status and peak VRAM 6843.90966796875 MB; [`experiments/cf_cycle_4/nullspace_ablation_seed43/control_gate.json`](experiments/cf_cycle_4/nullspace_ablation_seed43/control_gate.json:1) reports valid true.
- Completed treatment only after valid control: [`experiments/cf_cycle_4/nullspace_ablation_seed43/treatment/run_status.json`](experiments/cf_cycle_4/nullspace_ablation_seed43/treatment/run_status.json:1) reports completed status and peak VRAM 6843.90966796875 MB; [`experiments/cf_cycle_4/nullspace_ablation_seed43/treatment_gate.json`](experiments/cf_cycle_4/nullspace_ablation_seed43/treatment_gate.json:1) reports valid true, active projection, non-empty frozen basis, 3600 calls with frozen basis, and removed norm sum 3.100088352795865.
- Ran comparison: [`experiments/cf_cycle_4/nullspace_ablation_seed43/comparison.json`](experiments/cf_cycle_4/nullspace_ablation_seed43/comparison.json:1) reports valid true, projection active true, frozen basis non-empty true, both peaks under 7500 MB, `delta_bwt_points=-0.008929480489536235`, `delta_avg_acc_points=-0.008046083930968173`, old-task gaps `[-0.1564058367372001, 0.13854687575812763]`, and `continue=false`.
- Wrote result-critic handoff [`experiments/cf_cycle_4/seed43_result_critic_packet.md`](experiments/cf_cycle_4/seed43_result_critic_packet.md:1). Recommended next action: result critic should accept/reject the evidence quality, then the loop should pivot to revised treatment-strength hypotheses before any current4/v10/Digital Twin escalation. Do not claim catastrophic forgetting is solved.

### 2026-05-17 — CF-cycle-3 report writer closeout
- Wrote the durable closeout report to [`experiments/cf_cycle_3/REPORT.md`](experiments/cf_cycle_3/REPORT.md:1) because the loop needed a stable objective/evidence/decision/next-seed packet after the corrected seed-42 control and treatment comparison.
- Preserved the confirmed root cause from [`experiments/cf_cycle_3/root_cause_packet.md`](experiments/cf_cycle_3/root_cause_packet.md:1): Task 0 Batch 93 at max length 256 produced all-ignored labels because prompt-first truncation left no supervised response labels.
- Recorded the implemented fix in [`_build_supervised_sequence()`](cascades/data.py:80) and [`prepare_data()`](cascades/data.py:107), plus regression evidence from [`tests/test_data.py`](tests/test_data.py:147) and [`experiments/cf_cycle_3/batch93_static_probe.verify.json`](experiments/cf_cycle_3/batch93_static_probe.verify.json:1): 14 data tests passed, batch 93 valid labels equal 1, and zero zero-label batches remained.
- Recorded the valid seed-42 feasibility comparison: [`experiments/cf_cycle_3/nullspace_ablation_retry/comparison.json`](experiments/cf_cycle_3/nullspace_ablation_retry/comparison.json:1) reports valid true, projection active true, frozen basis non-empty true, control peak 6848.810546875 MB, treatment peak 6850.75390625 MB, delta BWT +0.36167033073388477 points, delta average accuracy +0.24463211986087696 points, and continue false.
- Decision: retest once before escalation. The seed-42 treatment was tested and did not harm reduced proxy accuracy, but it missed the +1.5 BWT threshold in [`experiments/cf_cycle_2/EXPERIMENT_PROTOCOL.md`](experiments/cf_cycle_2/EXPERIMENT_PROTOCOL.md:114), so no catastrophic-forgetting solution claim is supported.
- Handoff to llm-loop-coordinator: CF-cycle-4 should run seed 43 under the same reduced-memory reasoning3 sequential control-then-treatment protocol; if seed 43 also misses the threshold, pivot to revised treatment-strength hypotheses before current4, v10, or Digital Twin runs.

### 2026-05-17 — CF-cycle-3 result-critic treatment comparison review
- Reviewed the first valid reduced-memory reasoning3 control/treatment comparison after the [`prepare_data()`](cascades/data.py:107) response-preserving truncation fix because the loop needed a bounded escalation decision, not a catastrophic-forgetting claim.
- Accepted feasibility at high confidence: [`comparison.json`](experiments/cf_cycle_3/nullspace_ablation_retry/comparison.json:1) reports valid true, both [`control/run_status.json`](experiments/cf_cycle_3/nullspace_ablation_retry/control/run_status.json:1) and [`treatment/run_status.json`](experiments/cf_cycle_3/nullspace_ablation_retry/treatment/run_status.json:1) completed under the 7500 MB VRAM threshold, and treatment projection was active in [`instrumentation.json`](experiments/cf_cycle_3/nullspace_ablation_retry/treatment/instrumentation.json:694).
- Rejected algorithmic escalation: [`comparison.json`](experiments/cf_cycle_3/nullspace_ablation_retry/comparison.json:4) shows delta BWT only 0.36167033073388477 points, below the 1.5 point success threshold in [`EXPERIMENT_PROTOCOL.md`](experiments/cf_cycle_2/EXPERIMENT_PROTOCOL.md:114), and [`treatment/metrics.json`](experiments/cf_cycle_3/nullspace_ablation_retry/treatment/metrics.json:9) still has negative absolute BWT.
- Wrote the durable criticism handoff to [`experiments/cf_cycle_3/treatment_comparison_result_critic_packet.md`](experiments/cf_cycle_3/treatment_comparison_result_critic_packet.md:1) so [`llm-report-writer`](experiments/cf_cycle_3/treatment_comparison_result_critic_packet.md:5) can produce the CF-cycle-3 report without overstating the result.
- Next executable loop seed: CF-cycle-4 should run one additional reduced-memory reasoning3 seed replication, sequential control then treatment, before either revising treatment strength or considering broader current4, v10, or Digital Twin escalation.

### 2026-05-17 — CF-cycle-3 result-critic control-gate review
- Reviewed the CF-cycle-3 reduced-memory [`reasoning3`](experiments/cf_cycle_3/control_retry_handoff_packet.md:5) control retry because treatment should not launch until the new control artifact passes the protocol gate.
- Accepted the control gate at high confidence for the bounded decision: [`experiments/cf_cycle_3/control_gate_validation.json`](experiments/cf_cycle_3/control_gate_validation.json:1) reports `valid=true`, [`experiments/cf_cycle_3/nullspace_ablation_retry/control/run_status.json`](experiments/cf_cycle_3/nullspace_ablation_retry/control/run_status.json:1) reports completed status, and [`experiments/cf_cycle_3/nullspace_ablation_retry/control/metrics.json`](experiments/cf_cycle_3/nullspace_ablation_retry/control/metrics.json:1) contains finite metrics.
- Wrote the durable criticism packet to [`experiments/cf_cycle_3/control_retry_result_critic_packet.md`](experiments/cf_cycle_3/control_retry_result_critic_packet.md:1) so the report writer can preserve the corrected conclusion: control evidence is acceptable, treatment remains untested, and catastrophic forgetting is not solved.
- Next executable step is treatment-only in [`experiments/cf_cycle_3/nullspace_ablation_retry`](experiments/cf_cycle_3/nullspace_ablation_retry), not `both`; comparison should run only after the treatment gate confirms completed status, finite artifacts, under-threshold VRAM, and active projection evidence.

### 2026-05-17 — CF-cycle-3 experiment-design control retry
- Added [`experiments/cf_cycle_3/reasoning3_prepare_data_preflight.py`](experiments/cf_cycle_3/reasoning3_prepare_data_preflight.py:1) and ran it because the prior fixed-loader evidence covered Task 0/batch 93, while the retry needed all three `reasoning3` tasks scanned at max length 256.
- CPU preflight output [`experiments/cf_cycle_3/reasoning3_prepare_data_preflight.json`](experiments/cf_cycle_3/reasoning3_prepare_data_preflight.json:1) shows all three `reasoning3` tasks valid, with zero zero-label batches and max sequence length 256.
- Launched the reduced-memory control only: `python experiments\cf_cycle_1\run_nullspace_ablation.py --arm control --task-suite reasoning3 --seed 42 --epochs 2 --rank 4 --max-length 256 --output-root experiments\cf_cycle_3\nullspace_ablation_retry --vram-threshold-mb 7500 > experiments\cf_cycle_3\nullspace_ablation_retry\control.log 2>&1`.
- The new control status in [`experiments/cf_cycle_3/nullspace_ablation_retry/control/run_status.json`](experiments/cf_cycle_3/nullspace_ablation_retry/control/run_status.json:1) is `completed`, with `peak_vram_mb=6848.810546875`, under the 7500 MB threshold; the old failed CF-cycle-2 artifacts were preserved.
- Added [`experiments/cf_cycle_3/validate_control_gate.py`](experiments/cf_cycle_3/validate_control_gate.py:1) and ran `python -m py_compile experiments\cf_cycle_3\reasoning3_prepare_data_preflight.py experiments\cf_cycle_3\validate_control_gate.py && python experiments\cf_cycle_3\validate_control_gate.py > experiments\cf_cycle_3\control_gate_validation.json && type experiments\cf_cycle_3\control_gate_validation.json`; exit 0 and [`experiments/cf_cycle_3/control_gate_validation.json`](experiments/cf_cycle_3/control_gate_validation.json:1) reports `valid=true`.
- Wrote [`experiments/cf_cycle_3/control_retry_handoff_packet.md`](experiments/cf_cycle_3/control_retry_handoff_packet.md:1) because result critic needs raw artifact references, observed facts, skipped checks, and a treatment-blocking decision boundary. Treatment and comparison were intentionally not run in this step; no catastrophic-forgetting claim is supported.

### 2026-05-16 — CF-cycle-3 data-loader TDD bugfix
- Added CPU regression tests in [`tests/test_data.py`](tests/test_data.py:143) before production changes because the reduced-memory `reasoning3` control failed when [`prepare_data()`](cascades/data.py:80) emitted all-`-100` labels at max length 256.
- Verified the red state first: `python -m pytest tests/test_data.py -q` failed for the expected reason, with all-ignored label batches including batch 93 and a synthetic long-prompt sample retaining zero supervised labels.
- Patched [`prepare_data()`](cascades/data.py:80) through helper [`_build_supervised_sequence()`](cascades/data.py:80) so truncation reserves at least one response token, masks prompt tokens, preserves response supervision, and raises if a response tokenizes to no supervised tokens.
- Verified green state: `python -m pytest tests/test_data.py -q` passed 14 tests; `python -m py_compile cascades\data.py experiments\cf_cycle_3\batch93_probe.py` exited 0.
- Re-ran a real-tokenizer CPU scan for [`data/task0_gsm8k_cot.jsonl`](data/task0_gsm8k_cot.jsonl:1) at seed 42 and max length 256, writing [`experiments/cf_cycle_3/batch93_static_probe.verify.json`](experiments/cf_cycle_3/batch93_static_probe.verify.json:1): batch 93 now has 1 valid label, `batch_93_all_labels_masked=false`, zero-label batches are empty, and min valid labels is 1.
- Decision: the reduced-memory control can be retried by Experiment Designer. Treatment remains blocked until that control is finite and under threshold; do not claim catastrophic forgetting is solved.

### 2026-05-16 — CF-cycle-3 debug root-cause diagnosis
- Investigated the `reasoning3` Task 0 Epoch 1 Batch 93 non-finite control failure without launching treatment or changing training behavior.
- Created [`experiments/cf_cycle_3/batch93_probe.py`](experiments/cf_cycle_3/batch93_probe.py:1) and output [`experiments/cf_cycle_3/batch93_static_probe.json`](experiments/cf_cycle_3/batch93_static_probe.json:1) to reproduce DataLoader shuffle order and token/label stats under seed 42 and max length 256.
- Root cause found: corrected DataLoader order maps batch 93 to zero-based dataset index 9 / JSONL line 10 in [`data/task0_gsm8k_cot.jsonl`](data/task0_gsm8k_cot.jsonl:10); the prompt tokenizes to 304 tokens, so max length 256 truncation leaves zero supervised labels and PyTorch mean cross-entropy returns `NaN` for all-`-100` labels.
- Wrote evidence packet [`experiments/cf_cycle_3/root_cause_packet.md`](experiments/cf_cycle_3/root_cause_packet.md:1). Recommended next step: confirm diagnosis, then implement a minimal `prepare_data()` guard/fix plus CPU test before any bounded GPU control retry. Treatment remains blocked.

### 2026-05-16 — CF-cycle-2 report writer closeout
- Wrote durable closeout report to [`experiments/cf_cycle_2/REPORT.md`](experiments/cf_cycle_2/REPORT.md:1) because the loop needed a durable objective/evidence/verdict/next-seed artifact after the approved control failed its guardrail.
- Recorded the key decision: treatment is blocked, the failed control is valid evidence that the new guardrails reject unsafe runs, and no null-space mitigation or catastrophic-forgetting claim is supported.
- Next-cycle seed: CF-cycle-3 should debug-first reproduce and instrument the reasoning3 Task 0 non-finite loss near batch 93 before trying lower learning rates, disabled SVC/principal expansion, or any treatment launch.

### 2026-05-16 — CF-cycle-2 research planning
- Read the durable CF-cycle-1 report, result critic packet, direct null-space ablation wrapper, comparison script, harness audit, and project context to scope the recovery cycle.
- Planning conclusion: CF-cycle-2 must go to code before hypothesis or GPU work because the current wrapper and comparison path still lack hard non-finite loss aborts, hard/failed-run peak-VRAM enforcement, and invalid-arm rejection for both control and treatment.
- Saved the implementation and handoff plan to [`docs/superpowers/plans/2026-05-16-cf-cycle-2-reasoning3-guardrails.md`](docs/superpowers/plans/2026-05-16-cf-cycle-2-reasoning3-guardrails.md:1) so the next worker can patch guardrails before any reduced-memory [`reasoning3`](experiments/cf_cycle_1/run_nullspace_ablation.py:32) GPU run.

### 2026-05-16 — CF-cycle-1 research planning
- Read targeted catastrophic-forgetting context in [README.md](README.md), [experiments/v10_hypotheses.md](experiments/v10_hypotheses.md), [research_loop_design.md](research_loop_design.md), [experiment_matrix.py](experiment_matrix.py), [train.py](train.py), [cascades/adapters.py](cascades/adapters.py), [cascades/injection.py](cascades/injection.py), [cascades/data.py](cascades/data.py), saved result CSVs, and [reports/parametric_memory_issue/REPORT.md](reports/parametric_memory_issue/REPORT.md).
- Purpose: prepare CF-cycle-1 handoff focused on reducing BWT variance and validating `enable_coso_nullspace` / frozen null-space anti-forgetting behavior under RTX 4060 Ti 8GB constraints.
- Planning conclusion: prioritize a small, controlled reproducibility/ablation plan before new architecture work because saved matrices show high BWT variance across runs and the current runner/config layer has known gaps around task overrides, rank sweep pass-through, and generative metric capture.

### 2026-05-16 — CF-cycle-1 hypothesis generation
- Converted the planner packet into ranked, falsifiable hypotheses for BWT variance and old-task retention failure modes.
- Recommended the first GPU-consuming experiment be a two-arm controlled ablation: no frozen null-space/no reassignment control versus frozen null-space treatment, using identical task order, seed, rank 8, max length 384, and 2 epochs.
- Added preconditions for the experiment designer to verify harness comparability, task count/order consistency, returned metrics, VRAM headroom, and projection activation before spending RTX 4060 Ti 8GB time.

### 2026-05-16 — CF-cycle-1 experiment design
- Ran CPU-light harness checks before GPU time. Earlier evidence said `python -m pytest tests/test_data.py -q` failed because [`tests/test_data.py`](tests/test_data.py:1) expected 3 sequential tasks while [`cascades/data.py`](cascades/data.py:35) defined 4 tasks ordered CSQA → ARC → GSM8K → Digital Twin; this was later superseded by the report-writer verification note below.
- Verified `research_runner.ExperimentRunner._run_cascades()` is not safe for the first controlled ablation because it pops/drops `task_files` and does not forward `rank` or `max_length` to `train_cascades()`.
- Verified `num_samples` in `train_cascades()` is signature/docstring-only for the current path and should not be part of the experiment contract until wired through.
- Designed the official CF-cycle-1 GPU run as a direct-wrapper, one-seed two-arm ablation using current4 first: control `enable_coso_nullspace=False`, `enable_cllora_reassign=False`; treatment `enable_coso_nullspace=True`, `enable_cllora_reassign=False`; both use seed 42, rank 8, max length 384, 2 epochs, same model, same task order, same sleep setting.
- Saved the detailed experiment plan and evidence packet to `docs/superpowers/plans/2026-05-16-cf-cycle-1-nullspace-ablation.md`. Code-mode implementation is required before GPU execution to add durable audit/runner scripts, projection/basis instrumentation, and task-suite documentation/test repair.

### 2026-05-16 — CF-cycle-1 result criticism
- Reviewed [`experiments/cf_cycle_1/README.md`](experiments/cf_cycle_1/README.md:37), [`experiments/cf_cycle_1/harness_audit.log`](experiments/cf_cycle_1/harness_audit.log:92), [`experiments/cf_cycle_1/run_nullspace_ablation.py`](experiments/cf_cycle_1/run_nullspace_ablation.py:205), [`train.py`](train.py:171), [`cascades/data.py`](cascades/data.py:35), and the partial control artifacts.
- Verdict: the running current4 control arm is diagnostic only, not confirmatory algorithm evidence, because GPU execution violated the audit precondition, no treatment arm exists, no final metrics or instrumentation exist, VRAM exceeds the 7500 MB threshold, and Task 3 shows non-finite loss in [`experiments/cf_cycle_1/nullspace_ablation/control.log`](experiments/cf_cycle_1/nullspace_ablation/control.log:181).
- Recommendation: block treatment, do not kill the running process without approval, and seed the next cycle with a reduced-memory reasoning3 ablation plus hard finite-loss and VRAM guards. Durable packet written to [`experiments/cf_cycle_1/result_critic_packet.md`](experiments/cf_cycle_1/result_critic_packet.md:1).

## Status
- **Working**: 
  - Google Takeout extraction (6,577 chunks @ E:\digital-twin\takeout_chunks)
  - Neo4j Knowledge Graph: **50,179 nodes / 85,720 relationships** (bolt://localhost:7687, pw: cascades2024)
  - Training data JSONL generation (159 unique pairs, 27.7MB @ E:\digital-twin\training_data\digital_twin_cascades.jsonl)
  - KG builder with self-contained Cypher MERGE statements (zero-error execution)
  - V10 twin training completed (4 tasks: T0-T3)
- **Pending**: Larger training dataset (current 159 pairs could be expanded with more diverse query templates)

## Tech Stack
- Python 3.x, PyTorch, Transformers
- Neo4j Community 5.26.2 (E:\neo4j\neo4j-community-5.26.2)
- Qwen3-4B (abliterated) - target model
- RTX 4060 Ti 8GB - training GPU

## Key Files
- `extract_takeout.py` — Extracts Google Takeout ZIPs → text chunks (12k chars each)
- `build_knowledge_graph.py` — Rule-based entity extraction → Cypher → Neo4j
- `build_training_data.py` — Chunks → CASCADES ChatML JSONL training pairs
- `local_extract_cascades.py` — LLM-assisted entity extraction (uses CASCADES server)
- `train.py` — Main CASCADES training pipeline

## Architecture Quirks
- Chunks in Drive are from a PREVIOUS session (32k chars each, pre-existing)
- The current extraction script produces 12k char chunks, but we're using Drive's 32k chunks
- Each Cypher statement is self-contained with inline MATCH/MERGE (no cross-stmt variable deps)
- Statements separated by semicolons, executed individually for reliability

## Trap Diary
| Issue | Cause | Fix |
|-------|-------|-----|
| Neo4j had 5326 nodes but only 7 relationships | Cypher used variable refs (user, page) across separate statements | Made every MERGE relationship self-contained with inline MATCH |
| TransactionError on batched commits | Some statements in batch failed, rolling back entire batch | Switched to individual auto-commit per statement |
| Pyre2 lint errors everywhere | Type checker false positives on slice indexing, string formatting | Ignored — runtime-valid Python |
| extract_takeout.py writing to old path | Process had old code loaded when started | Copied chunks from Drive\takeout_chunks_32k to takeout_chunks |

## Anti-Patterns (DO NOT)
- DO NOT use cross-statement variable refs in Cypher without a single transaction
- DO NOT use batch transactions with mixed MATCH/MERGE (some may fail)
- DO NOT filter entities — user wants ALL data, good and bad

## Data Paths
- Raw ZIP files: E:\code.projects\CASCADES--continual-PEFT-for-Local-LLMs\data\
- Extracted raw: E:\digital-twin\takeout_raw\
- Text chunks: E:\digital-twin\takeout_chunks\ (6,577 files)
- Cypher output: E:\digital-twin\cypher_output\
- Training data: E:\digital-twin\training_data\digital_twin_cascades.jsonl

## Build / Verify
```powershell
# Check Neo4j
python -c "from neo4j import GraphDatabase; d = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j','cascades2024')); s = d.session(); print(s.run('MATCH (n) RETURN count(n)').single()[0]); s.close(); d.close()"

# Run KG builder
python build_knowledge_graph.py

# Build training data
python build_training_data.py
```

## Task Log

### 2026-05-16 — CF-cycle-1 report writer closeout
- Wrote durable cycle report to [`experiments/cf_cycle_1/REPORT.md`](experiments/cf_cycle_1/REPORT.md:1) because the loop needed an objective/evidence/verdict/next-seed packet that survives mode handoff.
- Corrected the stale CF-cycle-1 test status: the later CPU-light verification command ran Python bytecode compilation over [`experiments/cf_cycle_1/harness_audit.py`](experiments/cf_cycle_1/harness_audit.py:1), [`experiments/cf_cycle_1/run_nullspace_ablation.py`](experiments/cf_cycle_1/run_nullspace_ablation.py:1), and [`experiments/cf_cycle_1/compare_nullspace_ablation.py`](experiments/cf_cycle_1/compare_nullspace_ablation.py:1), then ran quiet pytest over [`tests/test_data.py`](tests/test_data.py:1); it exited 0, with 11 tests in [`tests/test_data.py`](tests/test_data.py:1) passing.
- Final CF-cycle-1 decision: do not claim catastrophic forgetting is solved; this cycle produced harness diagnostics, identified a protocol-deviating partial [`current4`](experiments/cf_cycle_1/run_nullspace_ablation.py:26) control run, and blocked invalid treatment execution.
- Next coordinator handoff: continue with a reduced-memory [`reasoning3`](experiments/cf_cycle_1/run_nullspace_ablation.py:32) null-space ablation using hard finite-loss and peak-VRAM guardrails before returning to [`current4`](experiments/cf_cycle_1/run_nullspace_ablation.py:26) or Digital Twin-specific retention.

### 2026-05-16 — CF-cycle-2 planner closeout
- Wrote the CF-cycle-2 plan because the loop needs guardrail repair before any new algorithmic evidence can be trusted.
- Decision: hand to code first, not hypothesis generation, with exact target files [`train.py`](train.py:171), [`experiments/cf_cycle_1/run_nullspace_ablation.py`](experiments/cf_cycle_1/run_nullspace_ablation.py:205), and [`experiments/cf_cycle_1/compare_nullspace_ablation.py`](experiments/cf_cycle_1/compare_nullspace_ablation.py:12).
- GPU remains blocked until CPU-light verification proves hard non-finite-loss aborts, peak-VRAM failure marking above 7500 MB, invalid-arm comparison rejection, and mandatory treatment projection evidence.

### 2026-05-16 — CF-cycle-2 guardrail repair implementation
- Added CPU-only guardrail tests in [`tests/test_cf_cycle_2_guardrails.py`](tests/test_cf_cycle_2_guardrails.py:1) because synthetic invalid artifacts must be rejected before any reduced-memory GPU attempt.
- Patched [`train_cascades()`](train.py:199) with default-off `abort_on_nonfinite` and `vram_threshold_mb` guardrails, plus `TrainingGuardrailViolation`, so normal training calls stay backward compatible while the direct wrapper can fail fast on non-finite losses or peak-VRAM breaches.
- Patched [`experiments/cf_cycle_1/run_nullspace_ablation.py`](experiments/cf_cycle_1/run_nullspace_ablation.py:205) to pass guardrails into training, track observed VRAM peaks from instrumentation/final CUDA state, and write durable `run_status.json` for both completed and failed-guardrail arms.
- Patched [`experiments/cf_cycle_1/compare_nullspace_ablation.py`](experiments/cf_cycle_1/compare_nullspace_ablation.py:102) so comparisons require metrics, instrumentation, and run status for both arms, reject failed/non-finite/over-threshold evidence, and require active treatment projection evidence.
- Created the CF-cycle-2 runbook at [`experiments/cf_cycle_2/README.md`](experiments/cf_cycle_2/README.md:1) documenting CPU-light verification and explicit-approval-only reduced-memory `reasoning3` commands; invalid CF-cycle-1 [`current4`](experiments/cf_cycle_1/run_nullspace_ablation.py:26) artifacts remain diagnostic only and must not be used as algorithmic evidence.

### 2026-05-16 — CF-cycle-2 experiment-design review
- Reviewed the patched guardrails and runbook because the loop needed an explicit ready/not-ready decision before any reduced-memory GPU launch.
- Re-ran CPU-light verification from the repository root: `python -m py_compile train.py experiments\cf_cycle_1\run_nullspace_ablation.py experiments\cf_cycle_1\compare_nullspace_ablation.py && python -m pytest tests/test_data.py tests/test_cf_cycle_2_guardrails.py -q`; it exited 0 with 16 tests passing in 0.90 seconds.
- Re-ran CLI help checks for [`experiments/cf_cycle_1/run_nullspace_ablation.py`](experiments/cf_cycle_1/run_nullspace_ablation.py:364) and [`experiments/cf_cycle_1/compare_nullspace_ablation.py`](experiments/cf_cycle_1/compare_nullspace_ablation.py:176); both exited 0 and showed the expected guardrail/comparison options.
- Wrote the explicit launch protocol at [`experiments/cf_cycle_2/EXPERIMENT_PROTOCOL.md`](experiments/cf_cycle_2/EXPERIMENT_PROTOCOL.md:1), recommending control-only first and treatment only after a completed, finite, under-threshold `reasoning3` control.
- Did not start any GPU run; frozen null-space mitigation remains untested until the approved sequential GPU protocol completes and [`comparison.json`](experiments/cf_cycle_2/EXPERIMENT_PROTOCOL.md:47) is valid.

### 2026-05-16 — CF-cycle-2 result-critic review
- Critiqued the reduced-memory `reasoning3` launch protocol because the loop needed a skeptical evidence-quality gate before spending GPU time.
- Verdict: approve a sequential GPU launch starting with control only, not both arms and not any catastrophic-forgetting claim; treatment remains conditional on a valid control and comparison remains conditional on valid treatment.
- Evidence quality is sufficient for launch-readiness guardrails but insufficient for algorithmic mitigation claims because only CPU-light syntax, test, and CLI-help checks were run; no GPU control, treatment, or comparison evidence exists yet.
- Key residual risks to preserve for the report writer: synthetic tests do not prove real GPU guardrail behavior, task-manifest matching is mainly protocol-level, and one seed/one short reduced suite can only support feasibility, not broad generalization.

### 2026-05-16 — CF-cycle-2 failed-control result criticism
- Reviewed the approved reduced-memory [`reasoning3`](experiments/cf_cycle_2/nullspace_ablation/control/run_status.json:4) control evidence after launch because the protocol required a control gate before treatment.
- Verdict: block treatment. The control persisted [`status=failed_guardrail`](experiments/cf_cycle_2/nullspace_ablation/control/run_status.json:3) with reason [`non-finite training loss task=0 epoch=1 batch=93`](experiments/cf_cycle_2/nullspace_ablation/control/run_status.json:13), so the stop condition in [`experiments/cf_cycle_2/EXPERIMENT_PROTOCOL.md`](experiments/cf_cycle_2/EXPERIMENT_PROTOCOL.md:91) is met.
- Interpreted the result as moderate evidence that guardrails fail fast and preserve rejected evidence under the VRAM threshold, not as evidence for or against frozen null-space mitigation; [`peak_vram_mb=6630.5986328125`](experiments/cf_cycle_2/nullspace_ablation/control/run_status.json:14) stayed below the 7500 MB threshold.
- Wrote the durable criticism packet to [`experiments/cf_cycle_2/result_critic_packet.md`](experiments/cf_cycle_2/result_critic_packet.md:1) and recommended handing to report writer to close CF-cycle-2, with CF-cycle-3 seeded as debug-first work on Task 0 batch ~93 non-finite loss before any further treatment launch.

### 2026-05-16 — CF-cycle-2 report writer closeout
- Wrote [`experiments/cf_cycle_2/REPORT.md`](experiments/cf_cycle_2/REPORT.md:1) to preserve the cycle objective, implementation summary, verification evidence, failed-control evidence, decision, confidence labels, and coordinator handoff.
- Updated this project memory because the next worker should not relitigate the guardrail decision: CF-cycle-2 successfully blocked invalid evidence, treatment remains blocked, and catastrophic forgetting is not solved.
- Handoff to [`llm-loop-coordinator`](experiments/cf_cycle_2/REPORT.md:92): continue with CF-cycle-3 debug-first reproduction of reasoning3 Task 0 batch ~93, instrumenting sample IDs, token lengths, losses, gradients, adapter norms, and SVC/principal-expansion state before testing lower learning rates or disabled SVC/principal expansion.
