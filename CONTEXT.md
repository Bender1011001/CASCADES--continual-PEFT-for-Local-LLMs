# CASCADES — Continual PEFT for Local LLMs

## Status
- **Working**: Training pipeline, 3-task evaluation, A100 scaling (rank/max_length/lr CLI args), v10.4 BWT fixes
- **Broken**: None known (BWT -14.5% → expected positive after v10.4 Colab run)

## Tech Stack
- Python 3.11+, PyTorch 2.x, bitsandbytes (4-bit quant)
- Targets: Qwen3-4B abliterated, A100 training, 8GB VRAM inference

## Key Files
- `cascades/adapters.py` — CASCADESAdapter (Stiefel manifold), FunLoRA, ResonantCore, full_descent_step
- `cascades/sleep.py` — SleepConsolidation cycle (reorth only; SVD/dedup/SHY disabled v10.4)
- `cascades/config.py` — AblationConfig (frozen dataclass)
- `train.py` — Main training loop with CLI args (--rank, --max_length, --lr_riemannian)
- `colab_cascades_qwen3_abliterated.ipynb` — Colab notebook for A100 training

## Architecture Quirks
- W = U @ C @ V^T where U,V are Stiefel manifold bases, C is liquid core pool
- full_descent_step is ORDER-DEPENDENT: pop→PaCA→DEAL→EMA→phantom repulsion→tangent map→EAR→core grad proj→QR retract→ripple fix
- FunLoRA activation is `x + σ(x) + tanh(x) - 0.5` (zero-centered at origin since v10.4)
- D-MoLE promotes FunLoRA→CASCADESAdapter at task boundaries

## Trap Diary
| Issue | Cause | Fix | Version |
|-------|-------|-----|---------|
| BWT -32.96% | 6 bugs: tangent paradox, promote amnesia, EAR leakage, Q_null rank, SVD skew, sleep parity | v10.3 fixes (see KI) | v10.3 |
| Task 0: 92%→75% | FunLoRA σ(0)=0.5 bias vanishes at promotion | Subtract 0.5 in activation | v10.4 |
| V/C unprotected | Only U had EAR/frozen basis; V freely rotated, C freely overwritten | V subspace tracking + core grad projection | v10.4 |
| EMA leaks frozen dirs | EMA accumulates energy in frozen directions over 100s of steps | Phantom repulsion: project EMA before tangent map | v10.4 |
| Sleep lobotomy | SVD consolid, cross-dedup, SHY rescale past-task magnitudes | Disabled all 3; keep only reorth | v10.4 |

## Anti-Patterns (DO NOT)
- Do NOT project EAR before tangent mapping (re-introduces frozen dirs via sym_U subtraction)
- Do NOT enable SVD consolidation/SHY in sleep (destroys past-task routing magnitudes)
- Do NOT update core_pool via Adam without projecting out frozen subspaces
- Do NOT use full-rank Q_null_U (rank//2 preserves plasticity)

## Build / Verify
```
python -m pytest tests/ -v --tb=short  # 151/152 pass (1 pre-existing data naming failure)
```
