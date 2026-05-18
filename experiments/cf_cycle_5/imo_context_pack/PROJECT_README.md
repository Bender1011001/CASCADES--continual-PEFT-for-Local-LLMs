# CASCADES (v9 Pro): The Cognitive Ecosystem

**Mathematically Secure Continual Learning for Abliterated LLMs under 8GB Constraints.**

CASCADES is an _Autopoietic Cognitive Ecosystem_ designed to adapt fragile, abliterated ("Heretic") models continuously with significantly reduced forgetting compared to standard fine-tuning methods.

## 🏆 The 4B Heretic Breakthrough

Standard fine-tuning (LoRA) destroys the reasoning architecture of abliterated models. CASCADES solves this by constraining updates to Autopoietic Stiefel Manifolds and training on pure zero-shot reasoning tasks.

| Method              | Model                | Avg ACC    | BWT        | VRAM      |
| :------------------ | :------------------- | :--------- | :--------- | :-------- |
| Budget-Matched LoRA | Qwen3-4B             | ~11.5%     | -28.4%     | ~7.8GB    |
| Standard LoRA       | Qwen3-8B Heretic     | 25.84%     | -12.18%    | 7.8GB     |
| **CASCADES v9 Pro** | **Qwen3-4B Heretic** | **35.91%** | **-1.46%** | **5.2GB** |

### Key Results

- **Avg Reasoning Accuracy**: 35.91% across 3 zero-shot reasoning task streams (495 examples).
- **Reduced Forgetting**: -1.46% BWT in best observed run (v9 Pro, seed 42). BWT varies across configurations — see Limitations.
- **Hardware**: Benchmarked on a single RTX 4060 Ti (8GB VRAM, 5.2GB used).
- **Speed**: 2.4x throughput increase via D-MoLE dynamic expert routing.

## 🚀 Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/your-org/CASCADES--continual-PEFT-for-Local-LLMs.git
cd CASCADES--continual-PEFT-for-Local-LLMs
pip install -e ".[dev]"
```

### Reproduce the Breakthrough

```bash
# Lightning reproduction (approx. 5 mins, 8GB GPU)
python reproduce_the_breakthrough.py
```

### Run Training with Generative Evaluation

```bash
# Full training + exact match evaluation
python train.py --eval_em

# Custom model / seed
python train.py \
    --model_id "p-e-w/Qwen3-4B-Instruct-2507-heretic" \
    --eval_em --seed 42
```

### Run Tests

```bash
pytest                    # 129 tests, all CPU-only
pytest tests/ -v --tb=short
```

## 📦 Library API

CASCADES ships as an installable Python package. Core components can be imported directly:

```python
from cascades import (
    # Configuration
    AblationConfig, DEFAULT_CONFIG,
    # Core adapters
    CASCADESAdapter,      # Full 5-pillar adapter for critical layers
    CASCADESLinear,       # Wrapper with promote/demote support
    FunLoRA_Adapter,      # Rank-1 adapter for non-critical layers
    ResonantCore,         # Hebbian multi-core routing
    # Injection
    inject_cascades,      # Inject adapters into HuggingFace models
    # Evaluation
    evaluate_generative,  # Structured prompting + multi-level matching
    answers_match,        # 3-level answer comparison
)
```

### Ablation Configuration

Replace global flags with a frozen dataclass:

```python
from cascades import AblationConfig

# Ablate specific components
config = AblationConfig(enable_paca=False, enable_svc=False)

adapter = CASCADESAdapter(d_in, d_out, rank=8, config=config)
```

### Evaluation Pipeline

The generative eval module solves the exact match gap with structured `<think>` prompting:

```python
from cascades import evaluate_generative, answers_match

# Run structured generative evaluation
results = evaluate_generative(model, tokenizer, task_id=0, max_samples=50)
print(f"Exact: {results['exact_match_rate']:.1%}")
print(f"Normalized: {results['normalized_match_rate']:.1%}")
print(f"Containment: {results['containment_match_rate']:.1%}")

# Or compare individual answers
match = answers_match("42", "$42$", level="normalized")  # True
```

## 📁 Project Structure

```text
train.py                   # v9 Pro training pipeline
evaluate.py                # EM diagnostic evaluator
reproduce_the_breakthrough.py

cascades/                  # Installable library
├── adapters.py            # CASCADESAdapter, CASCADESLinear, FunLoRA, ResonantCore
├── config.py              # AblationConfig frozen dataclass (8 flags)
├── injection.py           # inject_cascades(), D-MoLE, batched ops
├── eval.py                # Generative evaluation + answer matching
├── math_ops.py            # Riemannian gradient, QR retraction, EAR, DEAL, SVC
├── data.py                # Task prompts + DataLoader creation
├── metrics.py             # Average accuracy, BWT, forward transfer
└── sleep.py               # Bio-inspired sleep consolidation

data/                      # Training data suites
├── task2_csqa_cot.jsonl    # current4 task 0 — CommonsenseQA
├── task1_arc_cot.jsonl     # current4 task 1 — ARC Science
├── task0_gsm8k_cot.jsonl   # current4 task 2 — GSM8K Math
└── task3_digital_twin.jsonl # current4 task 3 — Digital Twin

tests/                     # Unit tests
papers/                    # Research papers + reference PDFs
```

## 🧠 Core Architecture

## 🧪 Task Suites

Current CF-cycle-1 task suite (`current4`, used by `cascades/data.py`) is:
1. `data/task2_csqa_cot.jsonl` — CommonsenseQA
2. `data/task1_arc_cot.jsonl` — ARC Science
3. `data/task0_gsm8k_cot.jsonl` — GSM8K Math
4. `data/task3_digital_twin.jsonl` — Digital Twin

Historical v9/v10 BWT values in saved CSVs and earlier README text may refer to the original 3-task reasoning suite. Do not compare BWT across suites without recording the task manifest.

CASCADES v9 Pro implements a 5-pillar architecture:

1. **Stiefel Manifold Learning** — Shared U/V bases constrained to orthonormal manifolds via QR retraction
2. **Gated Integration** — GainLoRA interference gates + FunLoRA rank-1 for non-critical layers
3. **Energy-Accounted Reassignment (EAR)** — Gradient redirection preserving ‖g‖₂ in the free subspace
4. **D-MoLE Dynamic Selection** — Activation-variance layer importance scoring for adapter allocation
5. **Autopoietic Regulation** — Dynamic rank expansion/contraction (Breathing Manifolds: r → r ± 1)

Additional v9 innovations:

- **Dormant Core Distillation**: Polar factor extraction to preserve memories during layer demotion
- **Riemannian Freeze**: Hibernation locks for manifold stability during expert dormancy
- **The Ripple Fix**: Basis counter-rotation using R^T (not R^{-1}) for covariant buffer mixing

## 📄 Research Paper

Details on the **Transpose Parity Bug** fix (R^⊤ mixing) and the **GQA Scaling Paradox** can be found in [papers/CASCADES_v9_Final_Paper.md](papers/CASCADES_v9_Final_Paper.md).

## ⚠️ Limitations

- **Forgetting Variance**: The reported -1.46% BWT is from the best v9 Pro run. Across 7 saved runs, BWT ranges from +2.42% to -32.96%. Several v10 configurations show BWT worse than -10%, indicating that anti-forgetting performance is sensitive to model variant, data distribution, and hyperparameters. All BWT values are computed on proxy accuracy (`exp(-loss)`), not generative exact match.
- **Exact Match**: While proxy accuracy reaches 35.91%, generative exact match rates require the structured evaluation pipeline (`--eval_em`)
- **GQA Scaling Paradox**: Performance plateaus at 8B (32.97%) compared to 4B breakthrough — see Section 7 of the paper
- **Training Data**: Current CF-cycle-1 experiments use the `current4` suite (CommonsenseQA, ARC Science, GSM8K Math, Digital Twin). Historical saved runs may use the original 3-task reasoning suite.
- **Benchmark Scale**: Most results are from single-seed experiments. Multi-seed and cross-domain evaluations are needed to confirm generalizability, and BWT should not be compared across task suites without a recorded manifest.

---

_CASCADES is open-source research for the unconstrained reasoning community._
