# Parametric Memory Issue Report
## Why the CASCADES Model Doesn't Remember After Training

**Date:** 2026-03-11
**Status:** In Progress — Multi-epoch training running now
**Hardware:** RTX 4060 Ti 8GB, Windows

---

## Problem Statement

After training the CASCADES model on personal data (Google Takeout → Neo4j Knowledge Graph → Q&A pairs), the model fails to recall any learned facts during inference. When asked "Who is Bender1011001?", it responds with generic answers like "there is no widely known public figure by that name."

---

## Root Cause Analysis

### Attempt 1: learn_and_extract.py (LLM processes chunks while learning)
- Model read Google Takeout chunks, generated Cypher, and trained on its own output
- **Problem:** 6,577 chunks × ~3 sub-chunks × 5 min/sub-chunk = **57 days** on 4060 Ti
- Only processed 1 chunk before we pivoted

### Attempt 2: chat_twin.py (Load previous trained weights + test)
- Loaded `cascades_v10_twin_weights.pt` (trained on 159 Q&A benchmark pairs)
- **Problem:** Weight mismatch
  - Checkpoint: rank=8, 336 keys
  - Model: rank=4, different D-MoLE layer assignments
  - Only **80/336 keys matched**, 256 FunLoRA keys (`adapter.a`, `adapter.b`) not found
  - The training data was generic Q&A, not personal identity data
- **Result:** Model didn't know anything about the user

### Attempt 3: agent_daemon.py --learn-graph (Graph → Q&A → CASCADES)
- Mined Neo4j KG (50K nodes, 85K relationships) → 193 Q&A pairs
- Trained with CASCADES Riemannian descent

**The most obvious root cause: Only 1 epoch (1 pass through 193 pairs).**

| Metric | Value | Meaning |
|--------|-------|---------|
| Final avg_loss | 3.3 | ~3.7% probability on correct token |
| Required for recall | < 1.0 | ~37% probability on correct token |
| Required for rote memory | < 0.5 | ~60% probability on correct token |

**A single pass through 193 pairs is not enough for a 4B parameter model to memorize anything.** The adapter perturbation is tiny relative to the base model's strong prior. Multiple epochs are required to overfit on the facts.

### Secondary Issue: VRAM at Rank 32
- Rank 32 caused 10GB usage (2GB spilling to shared memory via PCIe)
- Training slowed from ~15 sec/step to ~72 sec/step
- **Fix:** Reduced to rank 16 — fits in 8GB dedicated VRAM

### Secondary Issue: Low-quality training data
The graph is dominated by VISITED relationships (42,613 out of 42,899):

| Relationship | Count | Quality |
|---|---|---|
| VISITED | 42,613 | Low — mostly npm registry URLs |
| KNOWS | 203 | Medium — noisy (includes "The Sun", "Password") |
| USES | 38 | High — real tools and tech |
| RESEARCHES | 26 | High — actual research topics |
| SEARCHED_FOR | 13 | High — search queries |
| CONTRIBUTES_TO | 4 | High — real projects |
| INTERESTED_IN | 2 | High — interests |

The VISITED relationship generates Q&A pairs like:
```
Q: What websites does Bender1011001 visit?
A: Bender1011001 visits: registry.npmjs.org - rollup-win32-ia32-msvc-4.59.0.tgz...
```
This is noise, not useful identity information.

---

## Fix Applied

### Multi-epoch training (agent_daemon.py v2)
- Train for up to **20 epochs** over the same 193 Q&A pairs
- Stop early when loss drops below **1.0**
- Only freeze subspace (Titanium Padlock) **after** convergence, not mid-training
- Save checkpoint every epoch for crash recovery
- Shuffle data each epoch to prevent ordering bias

### Estimated timeline
- ~193 pairs × 15 sec/step × 20 epochs = ~16 hours worst case
- Expected early stop around epoch 5-10 if loss converges

---

## Architecture Summary

The current system implements an **Asynchronous Actor-Learner Architecture**:

```
┌─────────────────────────────────────────────┐
│           agent_daemon.py                    │
│                                              │
│  ┌──────────────────┐  ┌──────────────────┐ │
│  │ Hemisphere A      │  │ Hemisphere B      │ │
│  │ (Awake Actor)     │  │ (Dream Learner)  │ │
│  │                   │  │                   │ │
│  │ • Chat Interface  │  │ • Self-Synthesize │ │
│  │ • VM Idle Loop    │  │ • Train CASCADES  │ │
│  │ • Generate()      │  │ • Freeze Subspace │ │
│  │                   │  │ • Save Brain      │ │
│  └───────┬──────────┘  └───────┬──────────┘ │
│          │    brain_lock        │             │
│          └──────────────────────┘             │
│                    │                          │
│          ┌────────┴────────┐                 │
│          │  agent_brain.pt  │                 │
│          │  (432 MB)        │                 │
│          └─────────────────┘                 │
└─────────────────────────────────────────────┘
         ▲
         │ graph_synthesizer.py
         │
┌────────┴────────────────────────────────────┐
│          Neo4j Knowledge Graph               │
│  50,179 nodes │ 85,720 relationships         │
│  Mined from 6,577 Google Takeout chunks      │
└─────────────────────────────────────────────┘
```

---

## Key Files

| File | Purpose |
|------|---------|
| `agent_daemon.py` | Main lifelong agent — dual hemisphere, chat, VM, learning |
| `graph_synthesizer.py` | Mines Neo4j KG → dense Q&A training pairs |
| `chat_twin.py` | Simple chat interface for testing model recall |
| `weight_diag.txt` | Diagnostic output showing weight key mismatches |
| `graph_synth_output.txt` | Sample Q&A pairs generated from KG |
| `chunk_analysis.txt` | Breakdown of 6,577 takeout chunk categories |

---

## Next Steps

1. **Wait for multi-epoch training to converge** (loss < 1.0)
2. **Run identity test** to verify factual recall
3. **Clean graph data** — filter out npm registry noise from VISITED
4. **Expand training data** — extract more relationship types, add identity-specific hardcoded facts
5. **Consider rank expansion** — if 16 dimensions fill up, upgrade GPU or rent H100 for rank 32+
