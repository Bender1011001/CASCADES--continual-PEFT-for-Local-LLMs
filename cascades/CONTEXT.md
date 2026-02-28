# cascades/ — CASCADES library module

## Status

- **Working**: All library modules import cleanly, 129 tests pass
- **Working**: `config.py`, `adapters.py`, `injection.py`, `eval.py`, `math_ops.py`, `data.py`, `metrics.py`
- **Pending**: Experiment file still has inline class definitions (uses both library + inline)

## Tech Stack

- Python 3.11+ (3.13 tested)
- PyTorch 2.x
- No HuggingFace dependency for library code (only injection.py needs it)

## Key Files

- `adapters.py` — Core v9 classes: FunLoRA, ResonantCore, CASCADESAdapter, CASCADESLinear
- `config.py` — Frozen AblationConfig dataclass (8 flags, replaces global ENABLE\_\*)
- `injection.py` — inject_cascades(), D-MoLE, batched null-space/SVC ops
- `eval.py` — Generative evaluation: answer extraction, normalization, 3-level matching
- `math_ops.py` — Riemannian gradient, QR retraction, EAR, DEAL filter, SVC, PaCA
- `data.py` — Task prompts, DataLoader creation
- `metrics.py` — Average accuracy, BWT, forward transfer, proxy accuracy

## Architecture Quirks

- `adapters.py` imports `deal_heat_kernel_filter` from `math_ops.py` but wraps it with config-aware `_deal_filter()`
- `CASCADESAdapter.full_descent_step()` modifies parameters in-place inside `torch.no_grad()` — this is intentional
- `ResonantCore` uses detached Hebbian key updates during training (not in autograd graph)
- `CASCADESLinear.promote/demote` replace the adapter in-place — optimizer param groups must be rebuilt after

## Anti-Patterns (DO NOT)

- Do NOT use global `ENABLE_*` flags — use `AblationConfig` dataclass
- Do NOT rearrange operations in `full_descent_step()` — order is non-commutative
- Do NOT use R^{-1} for covariant buffers — use R^T (the Transpose Parity Fix)
- Do NOT add optimizer params inside `torch.no_grad()` autopoiesis block

## Build / Verify

```bash
pytest tests/ -v --tb=short  # 129 tests
python -c "from cascades import CASCADESAdapter, CASCADESLinear, FunLoRA_Adapter; print('OK')"
```
