# CASCADES — Root Project Context

## Status

- **Working**: Core math ops library (`cascades/`), v1-v5 experiment scripts, v9 Pro reasoning pipeline, reproduction script, unit tests, Colab notebook, Reddit post
- **Proven**: 4B Heretic Breakthrough (+0.82% BWT, 46.82% proxy accuracy on Qwen3-4B)
- **Addressed**: Exact Match inference gap (generative eval pipeline built, `--eval_em` flag), library refactoring complete
- **Broken/Incomplete**: 8B GQA scaling paradox

## Tech Stack

- Python 3.11+ (3.13 in dev), PyTorch ≥2.3, transformers ≥4.44, bitsandbytes ≥0.43
- Target models: `p-e-w/Qwen3-4B-Instruct-2507-heretic` (primary), `meta-llama/Meta-Llama-3-8B-Instruct` (secondary)
- Hardware: RTX 4060 Ti 8GB, NF4 quantization, bfloat16 compute dtype

## Key Files

- `cascades_exp/hf_cascades_reasoning.py` — v9 Pro main implementation (1145 lines)
- `cascades/adapters.py` — Core v9 adapter classes (library-grade, config-injected)
- `cascades/config.py` — AblationConfig frozen dataclass (replaces global flags)
- `cascades/injection.py` — D-MoLE, adapter injection, batched ops
- `cascades/math_ops.py` — Core Riemannian math (standalone, testable)
- `cascades/eval.py` — Generative evaluation with answer extraction & multi-level matching
- `cascades/sleep.py` — Bio-inspired sleep consolidation (SVD, dedup, re-ortho, SHY)
- `CASCADES_v9_Training.ipynb` — Google Colab notebook for GPU training
- `reproduce_the_breakthrough.py` — Lightning BWT verification
- `papers/CASCADES_v9_Final_Paper.md` — Research paper

## Architecture Quirks

- 4-bit quantized model uses float16/bfloat16 compute → adapters must init small (×0.01), alpha-mix at 0.1
- GainLoRA gate computes in float32 → output MUST cast to input dtype
- D-MoLE uses activation variance hooks (not gradient norms) due to 4-bit no-grad restriction
- EAR and tangent projection are non-commutative — tangent FIRST, then EAR
- QR retraction generates R matrix that must counter-rotate all historical buffers (Ripple Fix)
- U_shared/V_shared are Riemannian-only — NEVER in Adam optimizer

## Trap Diary

| Issue                         | Cause                                  | Fix                                  |
| ----------------------------- | -------------------------------------- | ------------------------------------ |
| Double-Optimizer annihilation | U/V in Adam AND manual Riemannian step | Exclude U/V from Adam                |
| EAR-StelLA non-commutativity  | EAR applied before tangent projection  | Tangent first, EAR in tangent space  |
| Basis-Destruction bug         | QR R-matrix rotates basis columns      | Counter-rotate historical cores by R |
| NaN loss in 4-bit             | fp32/fp16 mixing + large init          | Scale ×0.01, alpha=0.1, clip 1.0     |
| SDPA dtype mismatch           | GainLoRA float32 upcast leaked         | Cast adapt_out to input_dtype        |
| D-MoLE probe crash            | requires_grad on 4-bit weights         | Activation variance hooks            |

## Anti-Patterns (DO NOT)

- Return adapter forward output in float32 — always cast to input dtype
- Use requires_grad on 4-bit quantized weights
- Use fixed spectral thresholds in DEAL on quantized models
- Apply Cayley retraction (O(d³)) — use QR (O(dr²))
- Put U_shared/V_shared in any standard optimizer

## Mental Map

```
cascades/          → Core math ops library (standalone, tested)
cascades_exp/      → All experiment scripts v1-v9 + training data + logs
experiments/       → Unified runner + ablation scripts
tests/             → Unit tests for math_ops, adapters, data, metrics
papers/            → Research papers + reference PDFs
results/           → CSV metrics across versions
docs/              → Additional documentation
scripts/           → Utility scripts
```

## Build / Verify

```
pytest tests/ -v --tb=short  # 166 tests
python reproduce_the_breakthrough.py  # Needs 8GB GPU
python cascades_exp/hf_cascades_reasoning.py --eval_em  # Training + generative EM eval
```

## Training Data

- `task0_logic_cot.jsonl` — 52 examples (math, proofs, algorithms, combinatorics)
- `task1_decomp_cot.jsonl` — 25 examples (architecture, system design, planning)
- `task2_action_cot.jsonl` — 22 examples (bash, python, terraform, docker)
- Total: **99 examples** across 3 continual learning tasks
- Workflow: `.agents/workflows/create-training-data.md`

## Active Research Directions

1. ~~Exact Match Gap~~: Addressed with `cascades/eval.py` + `--eval_em` flag (needs GPU run)
2. **GQA Scaling Paradox**: 8B plateau at ~33% — dimension-calibrated Riemannian step sizes
3. **Standard Benchmarks**: No comparison with published CL benchmarks yet
4. ~~Code Refactoring~~: Complete — `cascades/adapters.py`, `config.py`, `injection.py`
5. ~~Test Coverage~~: 166 tests passing including v9 adapter + sleep tests
6. ~~Sleep Consolidation~~: Complete — 4-phase bio-inspired system in `cascades/sleep.py`
