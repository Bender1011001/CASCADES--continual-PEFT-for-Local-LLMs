# CASCADES: v3.1 → v5 Tuning Walkthrough (Qwen3-4B, strict 8GB VRAM)

## Objective
Increase **Average Accuracy (proxy)** under a strict **<8GB VRAM** constraint using a **Qwen3-4B** base model, while keeping continual-learning stability (BWT) meaningfully better than earlier CASCADES variants.

Baseline at start of this pass:
- **v3.1 proxy Avg Accuracy:** ~12.15% (exp(-loss) proxy)
- Primary constraint: **must not exceed 8GB VRAM**
- Continual setting: **3-task sequential stream**, no replay buffer

---

## v3.1 Tuning Pass (restore plasticity without breaking memory limits)

### Change 1 — D-MoLE critical threshold
- **Threshold:** 0.30 → 0.15  
- Effect: classifies more layers as “critical,” so more layers receive **full CASCADES adapters** instead of FunLoRA rank-1 fallback.
- Tradeoff: slightly higher memory pressure, but still within 8GB.

### Change 2 — FunLoRA init variance (rank-1 fallback)
- **Variance:** 0.01 → 0.05  
- Effect: gives rank-1 functional adapters enough initial amplitude to matter during short continual sequences.

### Result (v3.1 tuned)
- **Proxy Avg Accuracy:** ~13.79% (up from ~12.15%)
- **BWT:** ~-3.41% (tradeoff: accuracy improved, BWT worsened slightly vs more-protected variants)

---

## v4: Reviewer Corrections + Mathematical Overhaul (coherence > raw accuracy)

After reviewer-level sanity checks (internal), v4 was rewritten to remove theoretical mismatches (e.g., “SVD-free” claims while doing explicit SVD filters; causal counterfactual language without replay).

### Key v4 changes
1) **Energy-preserving gradient reassignment**
- Fixed the CL-style redirection so the update does not inject scalar projection energy into a single arbitrary direction.
- Reinjection is now distributed in the feasible subspace (orthogonal gradient coordinates), preserving update scale *within the constraint set*.

2) **Coordinate-space filtering (no explicit O(d^3) SVD)**
- Removed explicit SVD from the DEAL/heat-kernel style filter.
- Implemented adaptive smoothing in the learned coordinate space.
- Noise floor made quantization-aware using a step-based proxy: ~k·(Δ²/12) per quant group.

### Result (v4)
- **Proxy Avg Accuracy:** ~8.28%
- **BWT:** ~-1.48% (significant stabilization vs v3.1)
- Interpretation: the corrections increased protection/stability but reduced plasticity under strict quant + VRAM bounds.

---

## v5: TAG + ARR (attempt to recover accuracy while keeping v4 stability)

### Added mechanism 1 — Task-Aware Subspace Gating (TAG)
- Injected a task embedding into the GainLoRA-style gate so the system can suppress irrelevant subspace components during evaluation.

### Added mechanism 2 — Adaptive Rank Routing (ARR)
- After singular-channel calibration, hard-thresholded low-energy channels.
- Attempted to recycle dead rank dimensions (r=8) by injecting noise + re-orthogonalizing.

### Result (v5)
- **Proxy Avg Accuracy:** ~5.41%
- **BWT:** ~-0.93% (extremely stable)
- Interpretation: hard recycling + strict orthogonality disrupted momentum and over-isolated task paths. Forward transfer collapsed.

---

## Conclusions (from this tuning window)
1) **Plasticity vs stability frontier is real under 8GB + 4-bit constraints**
- v3.1 tuned moved toward plasticity (higher Avg Accuracy) at the cost of worse BWT.
- v4/v5 moved toward stability (better BWT) at the cost of accuracy.

2) **Hard rank recycling is too violent under strict orthogonality**
- Noise injection + re-orthogonalization breaks optimizer momentum.
- Task-ID gates can over-partition subspaces and eliminate shared benefit.

3) **Near-term direction**
- Replace hard ARR with **soft recycling** (gradual drift, EMA-based reinit, or soft routing).
- Relax strict orthogonality during recycling (soft constraint penalty instead of hard resets).
- Use gating as a *regularizer* not a hard partitioner (avoid “task silo” collapse).

---

## Summary Table (this pass)
| Version         | Key change                        | Proxy Avg Accuracy |      BWT | VRAM |
| --------------- | --------------------------------- | -----------------: | -------: | ---: |
| v3.1 (baseline) | starting point                    |            ~12.15% | (varies) | <8GB |
| v3.1 tuned      | D-MoLE 0.15, FunLoRA var 0.05     |            ~13.79% |  ~-3.41% | <8GB |
| v4              | math + “no SVD” coordinate filter |             ~8.28% |  ~-1.48% | <8GB |
| v5              | TAG + ARR                         |             ~5.41% |  ~-0.93% | <8GB |

Notes:
- “Avg Accuracy” here is an exp(-loss) proxy unless otherwise stated.
- All claims assume budget-matched evaluation under the same VRAM bound.
