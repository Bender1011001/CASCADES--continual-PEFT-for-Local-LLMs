# CASCADES — Root Project Context

## Status

- **Working**: Core library (`cascades/`), thin orchestrator (`train.py`), EM evaluator (`evaluate.py`), 152 unit tests, Colab notebook
- **Proven**: v9 Data Overhaul (35.91% avg ACC, -1.46% BWT on Qwen3-4B Heretic, 5.2GB VRAM)
- **Broken/Incomplete**: EM generative scores (0% — truncation issue, not knowledge gap), 8B GQA scaling paradox
- **Fixed**: Optimizer state corruption on rank contraction, ENABLE\_\* globals replaced with AblationConfig

## Tech Stack

- Python 3.11+, PyTorch ≥2.3, transformers ≥4.44, bitsandbytes ≥0.43
- Target models: `p-e-w/Qwen3-4B-Instruct-2507-heretic` (current), omni models (planned)
- Hardware: RTX 4060 Ti 8GB, NF4 quantization, bfloat16 compute dtype

## Key Files

- `train.py` — Thin training orchestrator (~350 lines, imports from cascades library)
- `evaluate.py` — EM diagnostic evaluator
- `cascades/adapters.py` — Core v9 adapter classes (AblationConfig-driven)
- `cascades/config.py` — AblationConfig frozen dataclass
- `cascades/data.py` — CoT JSONL data loader + per-example loss diagnostic
- `cascades/injection.py` — D-MoLE, adapter injection, batched ops, quant noise estimation
- `cascades/math_ops.py` — Core Riemannian math (Stiefel, EAR, DEAL, PaCA, SVC)
- `cascades/eval.py` — Generative evaluation with answer extraction
- `cascades/sleep.py` — Bio-inspired sleep consolidation
- `reproduce_the_breakthrough.py` — Lightning reproduction (10 steps, 2 tasks)
- `colab_cascades_v9.ipynb` — Google Colab notebook for GPU training

## Architecture Quirks

- `train.py` is a THIN orchestrator — all logic lives in `cascades/` library
- 4-bit quantized model uses bfloat16 compute → adapters must init small (×0.01), alpha-mix at 0.1
- GainLoRA gate computes in float32 → output MUST cast to input dtype
- D-MoLE uses activation variance hooks (not gradient norms) due to 4-bit no-grad restriction
- EAR and tangent projection are non-commutative — tangent FIRST, then EAR
- QR retraction generates R matrix that must counter-rotate all historical buffers (Ripple Fix)
- U_shared/V_shared are Riemannian-only — NEVER in Adam optimizer
- Rank contraction stores `_last_dead_idx` for surgical optimizer state cleanup

## Trap Diary

| Issue                         | Cause                                  | Fix                                             |
| ----------------------------- | -------------------------------------- | ----------------------------------------------- |
| Double-Optimizer annihilation | U/V in Adam AND manual Riemannian step | Exclude U/V from Adam                           |
| EAR-StelLA non-commutativity  | EAR applied before tangent projection  | Tangent first, EAR in tangent space             |
| Basis-Destruction bug         | QR R-matrix rotates basis columns      | Counter-rotate historical cores by R            |
| NaN loss in 4-bit             | fp32/fp16 mixing + large init          | Scale ×0.01, alpha=0.1, clip 1.0                |
| UnicodeEncodeError on Windows | Emoji in print() with cp1252           | UTF-8 reconfigure + ASCII markers               |
| EM gap (0% exact match)       | Model runs out of tokens mid-reasoning | Increase max_new_tokens or shorten think chains |
| Optimizer state corruption    | Full state flush on rank contraction   | Surgical zeroing via `_last_dead_idx`           |
| Frankenstein train.py         | 600 lines of duplicate classes/funcs   | Thin orchestrator importing from library        |

## Anti-Patterns (DO NOT)

- Return adapter forward output in float32 — always cast to input dtype
- Use requires_grad on 4-bit quantized weights
- Use fixed spectral thresholds in DEAL on quantized models
- Apply Cayley retraction (O(d³)) — use QR (O(dr²))
- Put U_shared/V_shared in any standard optimizer
- Use ENABLE\_\* globals — use AblationConfig from cascades.config
- Duplicate library code in train.py — import from cascades/

## Mental Map

```text
cascades/          → Core library (standalone, 152 unit tests)
data/              → Training data (495 zero-shot reasoning examples)
tests/             → Unit tests (mirror cascades/ structure)
papers/            → Research papers + reference PDFs
train.py           → Thin orchestrator (imports from cascades/)
evaluate.py        → EM diagnostic evaluator
```

## Training Data (v9 Data Overhaul)

- `data/task0_logic_cot.jsonl` — 159 examples (first-principles logic, math, Fermi estimation)
- `data/task1_decomp_cot.jsonl` — 184 examples (anti-sycophancy, critical analysis)
- `data/task2_action_cot.jsonl` — 152 examples (Python algorithmic synthesis)

## Build / Verify

```bash
pytest tests/ -v --tb=short
python train.py --eval_em
python evaluate.py --weights cascades_v9_weights.pt --fast --max_samples 10
```
