# CASCADES Research Log

> Autonomous research tracking for continual PEFT breakthroughs on consumer GPUs.

## Summary Dashboard

| Cycle | Date | Direction | Hypothesis | Result | Key Metric Delta |
| ----- | ---- | --------- | ---------- | ------ | ---------------- |
| —     | —    | —         | —          | —      | —                |

## Baselines (reference)

| Method                               | Avg ACC    | BWT        | Time  | VRAM   |
| ------------------------------------ | ---------- | ---------- | ----- | ------ |
| LoRA (Qwen3-4B)                      | 31.9%      | +11.2%\*   | 70s   | 4.5 GB |
| CASCADES v8 (Qwen3-4B)               | 14.87%     | +0.66%     | 27s   | ~5 GB  |
| **CASCADES v9 Pro (Qwen3-4B + CoT)** | **46.82%** | **+0.82%** | ~800s | 5.2 GB |
| LoRA (Qwen3-8B Heretic)              | 25.84%     | -12.18%    | —     | 7.8 GB |
| CASCADES v9 Pro (Standard 8B)        | 32.97%     | +2.01%     | —     | 5.8 GB |

\* LoRA "positive BWT" on 4B is a proxy artifact — model collapses to low-entropy attractor.

---

<!-- Cycle entries will be appended below -->

## Cycle 1: Generative Evaluation Pipeline (2026-02-27)

**Direction**: Solve the Exact Match (EM) gap — 0% EM despite 46.82% proxy accuracy.

**Root Cause**: Training pipeline only computed `exp(-loss)` proxy, never generated text. The standalone `eval_exact_match.py` used raw `model.generate()` with no structured prompting and strict `==` string matching.

**Implementation**:

- Created `cascades/eval.py` with `<think>` tag answer extraction, text normalization (Unicode, LaTeX, whitespace), 3-level matching (exact → normalized → containment + numeric equivalence), and structured system prompting
- Integrated as Phase 13 in training pipeline (`--eval_em` flag)
- 35 unit tests, all passing

**Status**: ✅ Complete — awaiting GPU run for actual EM numbers.

---

## Cycle 2: Modular Library Refactoring (2026-02-27)

**Direction**: Extract 1145-line monolithic `hf_cascades_reasoning.py` into testable library.

**Implementation**:

- `cascades/config.py` — `AblationConfig` frozen dataclass replacing 8 global `ENABLE_*` flags
- `cascades/adapters.py` — `FunLoRA_Adapter`, `ResonantCore`, `CASCADESAdapter`, `CASCADESLinear` with config injection
- `cascades/injection.py` — `inject_cascades()`, D-MoLE scoring, batched null-space/SVC ops
- `tests/test_adapters_v9.py` — 31 tests covering shapes, dtypes, descent step, promote/demote, config

**Result**: 129 total tests pass (98 prior + 31 new). Library imports cleanly:

```python
from cascades import CASCADESAdapter, CASCADESLinear, AblationConfig, inject_cascades
```

**Status**: ✅ Complete.
