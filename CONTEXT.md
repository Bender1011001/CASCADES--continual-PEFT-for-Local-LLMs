# CASCADES — Root Project Context

## Status

- **Working**: Core library (`cascades/`), thin orchestrator (`train.py`), EM evaluator (`evaluate.py`), 152 unit tests, Colab notebook
- **Proven**: v9 refactored run: 35.60% avg ACC, **+2.42% BWT** on Qwen3-4B Heretic, 5.2GB VRAM
- **v10 Patches Applied**: GQA preconditioning, ambient trace dedup, soft-EAR, principal tangent expansion, CFG decoding
- **Broken/Incomplete**: EM generative scores (improving — containment >60%), 8B GQA scaling paradox (fix now in code, untested)

## Tech Stack

- Python 3.11+, PyTorch ≥2.3, transformers ≥4.44, bitsandbytes ≥0.43
- Target models: `p-e-w/Qwen3-4B-Instruct-2507-heretic` (current), omni models (planned)
- Hardware: RTX 4060 Ti 8GB, NF4 quantization, bfloat16 compute dtype

## Key Files

- `train.py` — Thin training orchestrator (~350 lines, imports from cascades library)
- `evaluate.py` — EM diagnostic evaluator
- `cascades/adapters.py` — Core **v10** adapter classes (GQA preconditioning, soft-EAR, principal expansion, CFG boost)
- `cascades/config.py` — AblationConfig frozen dataclass (v10 flags: gqa_ratio, ear_gamma, enable_soft_ear, enable_principal_expansion, cfg_lambda)
- `cascades/data.py` — CoT JSONL data loader + per-example loss diagnostic
- `cascades/injection.py` — D-MoLE, adapter injection + **v10 GQA auto-detection**, batched ops, quant noise
- `cascades/math_ops.py` — Core Riemannian math + **v10 gqa_precondition_gradient** + **soft_ear**
- `cascades/eval.py` — Generative evaluation with answer extraction
- `cascades/sleep.py` — Bio-inspired sleep consolidation + **v10 ambient trace dedup**
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
- **v10**: GQA ratio auto-detected from model.config; K/V adapters get gqa_ratio attribute at injection
- **v10**: Cross-adapter dedup uses ambient trace Tr(Λ_A^T M_U Λ_B M_V), NOT flattened cosine
- **v10 BWT**: `frozen_null_basis` accumulates occupied subspace across task boundaries — gradients projected out of frozen + streaming null-space
- **v10 BWT**: `beta_ear=0.999` (~1000 step half-life), `Q_null_U` full-rank, sleep SVD threshold 0.98

## Trap Diary

| Issue                         | Cause                                                 | Fix                                                           |
| ----------------------------- | ----------------------------------------------------- | ------------------------------------------------------------- |
| Double-Optimizer annihilation | U/V in Adam AND manual Riemannian step                | Exclude U/V from Adam                                         |
| EAR-StelLA non-commutativity  | EAR applied before tangent projection                 | Tangent first, EAR in tangent space                           |
| Basis-Destruction bug         | QR R-matrix rotates basis columns                     | Counter-rotate historical cores by R                          |
| NaN loss in 4-bit             | fp32/fp16 mixing + large init                         | Scale ×0.01, alpha=0.1, clip 1.0                              |
| EM gap (0% exact match)       | Model runs out of tokens mid-reasoning                | Increase max_new_tokens, suppress tool_call                   |
| Optimizer state corruption    | Full state flush on rank contraction                  | Surgical zeroing via `_last_dead_idx`                         |
| Frankenstein train.py         | 600 lines of duplicate classes/funcs                  | Thin orchestrator importing from library                      |
| Cross-adapter dedup invalid   | Flattened core cosine in different coordinate frames  | Ambient trace projection via cyclic trace                     |
| EAR noise amplification       | Hard 1% cutoff → discontinuous gradient               | Tikhonov smooth regularization (soft-EAR)                     |
| 8B GQA scaling paradox        | K/V gradient inflation from fan-out                   | Scale by 1/√(H_q/H_kv) before Stiefel                         |
| Expansion shock               | Stochastic mini-batch init on revival                 | Power iteration on EAR sketch (noise-free)                    |
| Negative BWT on real data     | EAR sketch decays old-task info, no cross-task accum  | Frozen null-space snapshots at task boundary                  |
| Null-Space geometry bleed     | Prior-task projection applied to Stiefel tangent      | v10.1: Project ambient Euclidean EMA _before_ tangent mapping |
| Numerical projection leakage  | Sequential Gram-Schmidt drifts over 1000s of steps    | v10.2: Unified strict-orthogonal QR projection base           |
| Un-normalized covariance      | Eigenvalues reflect gradient magnitude instead of var | v10.2: Normalize sketch columns before covariance             |

## Anti-Patterns (DO NOT)

- Return adapter forward output in float32 — always cast to input dtype
- Use requires_grad on 4-bit quantized weights
- Use fixed spectral thresholds in DEAL on quantized models
- Apply Cayley retraction (O(d³)) — use QR (O(dr²))
- Put U_shared/V_shared in any standard optimizer
- Use ENABLE\_\* globals — use AblationConfig from cascades.config
- Duplicate library code in train.py — import from cascades/
- Compare flattened cores across adapters (geometrically invalid in different bases)

## Mental Map

```text
cascades/          → Core library (standalone, 152 unit tests, v10 architecture)
data/              → Real benchmark data (GSM8K, ARC, CSQA — downloaded via script)
tests/             → Unit tests (mirror cascades/ structure)
papers/            → Research papers + reference PDFs
train.py           → Thin orchestrator (imports from cascades/)
evaluate.py        → EM diagnostic evaluator
```

## Training Data (v10 Real Benchmarks)

- `data/task0_gsm8k_cot.jsonl` — 150 examples (GSM8K grade-school math, natural CoT from dataset)
- `data/task1_arc_cot.jsonl` — 150 examples (ARC-Challenge science reasoning, generated CoT)
- `data/task2_csqa_cot.jsonl` — 150 examples (CommonsenseQA, generated CoT)
- `download_real_data.py` — Script to download/convert from HuggingFace

## Build / Verify

```bash
pytest tests/ -v --tb=short
python download_real_data.py --num_samples 150
python train.py --eval_em
python evaluate.py --weights cascades_v10_colab_weights.pt --fast --max_samples 10
```
