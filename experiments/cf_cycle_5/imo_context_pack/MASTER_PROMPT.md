# Master Prompt for Advanced Math/Science LLM Reviewer

You are an advanced mathematical and scientific reasoning model. You are reviewing a compact context pack from the CASCADES project, a continual-PEFT system for local LLMs. The project goal is to reduce catastrophic forgetting during sequential fine-tuning.

Your task is not to summarize superficially. Your task is to analyze the mathematics, optimization dynamics, and experimental evidence, then propose the most likely next intervention that can actually improve backward transfer under the project constraints.

## Context

CASCADES uses low-rank adapter-style training with several mechanisms intended to preserve old-task capability while learning new tasks:

- frozen null-space projection / CoSO-style subspace protection;
- CL-LoRA-style gradient reassignment;
- energy-accounted reassignment / soft EAR;
- Riemannian/Stiefel adapter updates;
- D-MoLE dynamic layer selection;
- FunLoRA rank-1 adapters for non-critical layers;
- SVC / breathing-manifold calibration;
- PaCA causal attribution;
- DEAL quantization-aware filtering;
- sleep consolidation / ambient deduplication;
- proxy-loss continual-learning metrics and optional generative exact-match evaluation.

The latest controlled evidence is from a reduced `reasoning3` continual-learning suite. The isolated frozen-nullspace treatment activated correctly but did not meet the success threshold:

- Seed 42 comparison: valid evidence, active projection, delta BWT about +0.36 points, below the +1.5 point threshold.
- Seed 43 comparison: valid evidence, active projection, delta BWT about -0.009 points, below threshold and directionally flat/slightly negative.

Therefore, the current conclusion is: the harness works, data bug is fixed, projection activates, but isolated frozen-nullspace protection is too weak or mismatched. Catastrophic forgetting is not solved.

## Files to read first

Read these in order:

1. [`CONTEXT.md`](CONTEXT.md)
2. [`experiments/cf_cycle_4/REPORT.md`](experiments/cf_cycle_4/REPORT.md)
3. [`experiments/cf_cycle_3/REPORT.md`](experiments/cf_cycle_3/REPORT.md)
4. [`cascades/config.py`](cascades/config.py)
5. [`train.py`](train.py)
6. [`cascades/adapters.py`](cascades/adapters.py)
7. [`cascades/math_ops.py`](cascades/math_ops.py)
8. [`cascades/injection.py`](cascades/injection.py)
9. [`experiments/cf_cycle_1/run_nullspace_ablation.py`](experiments/cf_cycle_1/run_nullspace_ablation.py)
10. [`experiments/cf_cycle_1/compare_nullspace_ablation.py`](experiments/cf_cycle_1/compare_nullspace_ablation.py)
11. [`experiments/cf_cycle_3/nullspace_ablation_retry/comparison.json`](experiments/cf_cycle_3/nullspace_ablation_retry/comparison.json)
12. [`experiments/cf_cycle_4/nullspace_ablation_seed43/comparison.json`](experiments/cf_cycle_4/nullspace_ablation_seed43/comparison.json)

Use the remaining files as supporting evidence.

## Hard constraints

- Hardware target: RTX 4060 Ti 8GB.
- Do not require full exhaustive search over all knobs.
- Do not require the full `current4` suite or Digital Twin task as the first validation step.
- The next experiment should stay on the reduced `reasoning3` suite unless you justify otherwise.
- Keep evaluation guardrails: finite losses, under-threshold VRAM, valid `run_status.json`, strict comparison, active-projection evidence where relevant.
- Avoid conclusions that depend on changing many variables at once unless you explicitly describe why a bundle is required and how to isolate it afterward.
- Do not claim catastrophic forgetting is solved without validated multi-seed evidence.

## Your analysis goals

Answer these questions precisely:

1. **Failure-mode diagnosis**
   - Why might frozen null-space projection activate yet fail to improve BWT?
   - Is the likely issue insufficient subspace capture, wrong projection timing, insufficient rank/capacity, optimizer interaction, D-MoLE migration confound, loss/evaluation mismatch, quantization noise, or another mechanism?

2. **Mathematical critique**
   - Inspect the update equations and implementation paths in [`cascades/adapters.py`](cascades/adapters.py) and [`cascades/math_ops.py`](cascades/math_ops.py).
   - Identify whether the projection/reassignment math is internally coherent.
   - Look for basis-orientation errors, rank/capacity bottlenecks, projection onto the wrong subspace, destructive SVD/SVC behavior, bad damping scale, or mismatched gradient geometry.

3. **Treatment redesign**
   - Propose the next best treatment configuration.
   - Prefer one of these classes unless you find a stronger alternative:
     - frozen null-space + CL-LoRA reassignment;
     - soft EAR gamma sweep;
     - altered freeze timing or frozen-basis capacity;
     - disabling D-MoLE migration as a confound;
     - full v10 bundle followed by ablation;
     - rank/capacity increase with strict VRAM guard.

4. **Experiment design**
   - Give a concrete control/treatment protocol for the next run.
   - Specify exactly which knobs change and which must remain fixed.
   - Specify success/failure thresholds.
   - Specify what evidence artifacts must be produced.
   - Specify whether to run one seed first or multiple seeds immediately.

5. **Settings-fuzzer design, if useful**
   - If you recommend a fuzzer/search system, propose a staged search rather than exhaustive brute force.
   - Define the search space, candidate-pruning logic, promotion criteria, and stopping rules.
   - Explain how to avoid false positives caused by noisy one-seed proxy metrics.

## Required output format

Return a report with these sections:

1. **One-paragraph verdict**
2. **Most likely failure mode**, with evidence from files
3. **Mathematical/implementation issues to inspect or fix**, ranked by likelihood
4. **Recommended next treatment**, with exact configuration values
5. **Next experiment protocol**, including control, treatment, seed, task suite, rank, length, epochs, guardrails, and success criteria
6. **If building a settings fuzzer: staged design and search-space proposal**
7. **Top 5 code edits or probes**, each with file/function target and rationale
8. **Risks and possible false positives**

Be direct. If you think the current method is mathematically flawed, say so and explain. If you think the evidence is too weak to decide, propose the cheapest experiment that would disambiguate the leading hypotheses.

