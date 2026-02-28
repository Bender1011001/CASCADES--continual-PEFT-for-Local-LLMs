---
description: Autonomous research and experimentation loop for advancing CASCADES continual PEFT on consumer-size local models. Runs indefinitely — each cycle picks the highest-impact research direction, designs a hypothesis, implements and tests it, logs findings, and feeds wins into the next cycle.
---

# CASCADES Research Loop

> **Goal**: Continuously advance the project or make AI breakthroughs on consumer-size local models (RTX 4060 Ti 8GB).

This workflow is an **infinite loop**. Each iteration is one complete research cycle (~30–60 minutes). The loop only terminates when the user manually stops it or the GPU becomes unavailable.

---

## Preparation (once, at loop start)

1. **Read CONTEXT.md files** to synchronize with project state:
   - `e:\CASCADES--continual-PEFT-for-Local-LLMs\CONTEXT.md`
   - `e:\CASCADES--continual-PEFT-for-Local-LLMs\cascades_exp\CONTEXT.md`

2. **Read the research log** at `e:\CASCADES--continual-PEFT-for-Local-LLMs\research_log.md` (create it if missing). This is the cumulative record of all past cycles.

3. **Check GPU availability**:
   ```powershell
   python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}, VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f}GB' if torch.cuda.is_available() else 'NO GPU')"
   ```
   // turbo

---

## ♻ THE LOOP — repeat indefinitely

### Phase 1: SELECT — Pick the highest-impact research direction

Review the **Active Research Directions** list below and the research log. Select the direction with the highest expected impact that hasn't been recently attempted (or where prior attempts revealed promising follow-ups).

**Active Research Directions** (ordered by estimated impact):

| #   | Direction                       | Current State                                                                                                                  | Impact                                         |
| --- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------- |
| 1   | **Close the EM Gap**            | 0% EM despite 46.82% proxy ACC. Constrained decoding, structured prompting, and answer extraction improvements needed.         | 🔴 Critical — proves the system actually works |
| 2   | **GQA Scaling Paradox**         | 8B plateau at ~33%. Dimension-calibrated Riemannian step sizes, GQA-aware counter-rotation.                                    | 🟠 High — unlocks larger models                |
| 3   | **Novel Adapter Architectures** | Current: GainLoRA + FunLoRA + ResonantCore. Explore: mixture of subspaces, spectral adapters, orthogonal gradient projections. | 🟠 High — potential breakthroughs              |
| 4   | **Training Improvements**       | Curriculum learning, dynamic LR scheduling, better CoT data, synthetic data generation.                                        | 🟡 Medium — incremental gains                  |
| 5   | **Standard CL Benchmarks**      | No comparison with Split-CIFAR, Split-ImageNet, or standard NLP CL benchmarks (TRACE, etc.).                                   | 🟡 Medium — publishability                     |
| 6   | **Hyperparameter Search**       | Manual tuning only. Rank, alpha, learning rate, EAR strength, SVC thresholds.                                                  | 🟡 Medium — low-hanging fruit                  |
| 7   | **Memory Efficiency**           | Current peak 5.2GB. Gradient checkpointing, activation compression, mixed-precision tricks.                                    | 🟢 Low — already within budget                 |
| 8   | **Code Quality & Tests**        | Core math tested (98 tests), but v9 pipeline lacks tests, needs modular extraction.                                            | 🟢 Low — maintenance                           |

You may also **discover new directions** from recent papers. If you find a promising idea from arxiv or the papers/ directory, add it to the list.

### Phase 2: HYPOTHESIZE — Design a concrete, testable hypothesis

Write a clear hypothesis in the research log:

```
## Cycle N: [Direction Name]
**Date**: [timestamp]
**Hypothesis**: [One sentence: "If we [change X], then [metric Y] will [improve/change] because [reason Z]"]
**Method**: [2-3 sentences describing the experiment]
**Success Criteria**: [Concrete: "EM > 5%" or "BWT remains positive" or "VRAM < 7GB"]
```

### Phase 3: IMPLEMENT — Make the changes

1. **Create a branch or experiment variant** — never modify the main experiment script destructively. Create new files like `cascades_exp/hf_cascades_v10_[experiment_name].py` or modify in-place with feature flags.

2. **Implementation rules**:
   - All adapter outputs MUST cast to `input_dtype` (never return float32)
   - Never use `requires_grad` on 4-bit quantized weights
   - Use QR retraction, never Cayley (O(dr²) vs O(d³))
   - U_shared/V_shared are Riemannian-only — NEVER in Adam optimizer
   - Scale init ×0.01, alpha=0.1, grad clip 1.0
   - D-MoLE uses activation variance hooks (not gradient norms)
   - Tangent projection FIRST, then EAR (non-commutative order)
   - After QR retraction, counter-rotate historical buffers by R

3. **Write or update tests** for any new functionality in `tests/`.

### Phase 4: TEST — Run the experiment

// turbo

```powershell
cd e:\CASCADES--continual-PEFT-for-Local-LLMs
python -c "import torch; torch.cuda.empty_cache()" 2>$null
```

Run the experiment. Standard invocations:

```powershell
# For v9 reasoning pipeline (main experiment)
cd e:\CASCADES--continual-PEFT-for-Local-LLMs
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
python cascades_exp/hf_cascades_reasoning.py 2>&1 | Tee-Object -FilePath "logs/research_cycle_N.log"
```

```powershell
# For ablation via unified runner
cd e:\CASCADES--continual-PEFT-for-Local-LLMs
python experiments/run_experiment.py --method cascades_v5 --seeds 0 --output-dir results/cycle_N
```

```powershell
# For unit tests
cd e:\CASCADES--continual-PEFT-for-Local-LLMs
pytest tests/ -v --tb=short
```

**Capture ALL metrics**: Proxy ACC per task, BWT, training time, VRAM peak, EM rate (if applicable).

### Phase 5: ANALYZE — Interpret results and log findings

1. **Parse the log output** for key metrics.

2. **Compare against baselines**:
   | Baseline | Avg ACC | BWT | Time |
   |----------|---------|-----|------|
   | LoRA (Qwen3-4B) | 31.9% | +11.2%\* | 70s |
   | CASCADES v8 (Qwen3-4B) | 14.87% | +0.66% | 27s |
   | **CASCADES v9 Pro (Qwen3-4B + CoT)** | **46.82%** | **+0.82%** | ~800s |

3. **Update the research log** with results:
   ```
   **Results**:
   - Metric 1: [value] (vs baseline [value], delta [±%])
   - Metric 2: [value]
   **Analysis**: [What worked, what didn't, why]
   **Next Steps**: [What this result suggests for the next cycle]
   **Status**: ✅ BREAKTHROUGH / 🔄 PROMISING / ❌ DEAD END / 🔬 NEEDS MORE DATA
   ```

### Phase 6: INTEGRATE — If successful, merge improvements

1. If the experiment shows improvement (any metric up without others regressing):
   - Integrate the change into the main codebase
   - Update `CONTEXT.md` files with new findings
   - Update `cascades_exp/CONTEXT.md` experiment results table
   - Add to the trap diary if you discovered a new gotcha

2. If the experiment failed:
   - Log the failure mode clearly (this prevents repeating the mistake)
   - Add to Anti-Patterns if applicable
   - Identify the **next most promising direction** from Phase 1

### Phase 7: SEARCH — Find new ideas (every 3rd cycle)

Every 3 cycles, spend time on **literature research**:

1. Search arxiv for recent papers on:
   - Continual learning + PEFT/LoRA
   - Riemannian optimization for neural networks
   - Low-rank adaptation efficiency
   - Consumer GPU training techniques
   - Constrained decoding / structured generation

2. Check the `papers/` directory for unread reference papers.

3. Search GitHub for competing implementations or novel techniques.

4. Add any promising leads to the Active Research Directions table.

### Phase 8: LOOP — Go back to Phase 1

Before restarting:

1. Update `research_log.md` with cycle summary
2. Update `CONTEXT.md` if project state changed
3. Clear GPU cache:
   ```powershell
   python -c "import torch; torch.cuda.empty_cache()"
   ```
   // turbo

**→ Go to Phase 1**

---

## Research Log Format

The research log (`e:\CASCADES--continual-PEFT-for-Local-LLMs\research_log.md`) tracks all cycles:

```markdown
# CASCADES Research Log

## Summary Dashboard

| Cycle | Date       | Direction   | Hypothesis           | Result | Key Metric Delta |
| ----- | ---------- | ----------- | -------------------- | ------ | ---------------- |
| 1     | 2026-02-27 | EM Gap      | Constrained decoding | 🔄     | EM: 0%→3%        |
| 2     | 2026-02-27 | GQA Scaling | Dim-calibrated LR    | ❌     | No change        |

## Cycle 1: [Direction]

...
```

---

## Emergency Stops

- **OOM**: Reduce batch size or rank, add `torch.cuda.empty_cache()` between tasks
- **NaN loss**: Check dtype casting, reduce init scale, increase grad clip
- **Proxy accuracy drops below 10%**: Roll back changes, check for regression
- **VRAM exceeds 7.5GB**: Trigger breathing manifold contraction or reduce adapted layers

---

## Sub-Agent Parallelism (Optional)

For research phases that can be parallelized, use the sub-agents skill:

```powershell
# Parallel literature search (3 agents)
$env:OPENAI_HOST = "http://127.0.0.1:3000"; $env:OPENAI_API_KEY = "dummy"
C:\Users\admin\.bun\bin\goose.exe run --provider openai --model gemini-3-flash --with-builtin developer --no-profile --no-session --quiet --max-turns 5 --text "Search arxiv for papers on [topic] published after 2025. Summarize the top 3 most relevant to continual learning with LoRA adapters. Write findings to e:\CASCADES--continual-PEFT-for-Local-LLMs\research_notes\[topic].md"
```

For code experiments that don't need the GPU, dispatch implementation to a sub-agent while the main loop continues analysis.
