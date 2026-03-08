# CASCADES — Math Review Package

## What this is
CASCADES is a continual PEFT (Parameter-Efficient Fine-Tuning) framework for LLMs that uses Riemannian geometry on Stiefel manifolds to prevent catastrophic forgetting during sequential task adaptation.

**Target**: Qwen3-4B (4-bit NF4 quantized), 8 GB VRAM, no replay buffer.

## Key mathematical claims to review

1. **Stiefel manifold retraction** (`math_ops.py` — `stiefel_retract`): QR-based retraction after Riemannian gradient step. Is the tangent-space projection and retraction mathematically correct?

2. **Ripple Fix** (`math_ops.py` — `ripple_fix`): After QR retraction Q = qr(U − η·ξ), all covariant buffers are rotated by R^T and contravariant cores by R. Is this the correct transformation to maintain orthonormal-basis alignment?

3. **EAR (Energy-Accounted Reassignment)** (`math_ops.py` — `cllora_gradient_reassign`): Projects gradients onto the null-space of the occupied subspace, then rescales by ||g||/||g_free||. Is this a valid energy-preserving null-space projection?

4. **DEAL shrinkage** (`math_ops.py` — `deal_shrinkage`): Quantization-aware soft-thresholding with ε_quant = Δ²/12. Is the quantization noise floor formula correct and is the shrinkage operator well-defined?

5. **Dormant Core Distillation** (`adapters.py` — `_distill_to_rank1`): On layer demotion, lift SVD: a = U_h · u₁ · σ₁^(1/2), b^T = σ₁^(1/2) · v₁^T · V_h^T. Is this a valid manifold-lifting rank-1 approximation?

6. **FunLoRA demote scale** (`adapters.py`): scale = sqrt(σ₁ / 2.25). The 2.25 corrects for SiLU activation f'(0) = 1 · sigmoid(0) + 0 · ... ≈ 0.5, so gain ≈ 2.25 empirically. Is this scaling heuristic well-motivated?

7. **Breathing Manifolds** (`adapters.py` — `contract_rank`): Dead singular channels (D_weak < 1e-5) are zeroed rather than sliced to preserve autograd graph. Is this equivalent to true rank reduction for the purposes of gradient flow?

8. **Backward Transfer (BWT)** (`metrics.py`): BWT = (1/(T-1)) Σᵢ [A[T-1, i] − A[i, i]]. Standard continual learning definition — verify the implementation matches.

9. **GQA preconditioning** (`math_ops.py` — `gqa_precondition`): Scales gradients by 1/√γ where γ = n_heads / n_kv_heads (GQA ratio). Is this the right correction for the asymmetric gradient distribution in grouped-query attention?

## File map

| File | Contents |
|---|---|
| `CASCADES_v9_Final_Paper.md` | Full paper with theorems, algorithm, results |
| `math_ops.py` | All Riemannian math: retraction, EAR, DEAL, GORP, GQA preconditioning |
| `adapters.py` | CASCADESAdapter, FunLoRA, ResonantCore, Dormant Core Distillation |
| `injection.py` | D-MoLE layer selection, adapter injection, null-space accumulation |
| `sleep.py` | Sleep consolidation (SVD pruning, synaptic homeostasis) |
| `config.py` | AblationConfig — all hyperparameters |
| `train.py` | Full sequential training pipeline |
| `eval.py` | Proxy accuracy = exp(−loss), generative EM evaluation |
| `data.py` | Task data loading, chat-template tokenization |
| `metrics.py` | BWT, forward transfer, accuracy matrix |
| `AUDIT_REPORT.md` | Independent verification of all empirical claims |

## Empirical results (v9 Pro, Qwen3-4B Heretic, RTX 4060 Ti 8GB)

| Method | Avg Proxy ACC | BWT | VRAM |
|---|---|---|---|
| LoRA baseline (Qwen3-8B Heretic) | 25.84% | −12.18% | 7.8 GB |
| CASCADES v9 Pro (8B) | 32.97% | +2.01% | 5.8 GB |
| **CASCADES v9 Pro (4B Heretic)** | **46.82%** | **+0.82%** | **5.2 GB** |

Proxy Accuracy = exp(−cross_entropy_loss) on held-out response tokens.
BWT = backward transfer (positive = no forgetting, negative = catastrophic forgetting).
