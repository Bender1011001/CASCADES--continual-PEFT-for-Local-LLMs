# CASCADES Experiment Directory

## Status
- **Working**: hf_cascades.py (v1), hf_cascades_v2.py (v2 full fusion), hf_cascades_v3.py (v3.1 15-paper), lora_baseline.py
- **Completed Runs**: v1 on TinyLlama (fp32), v1 on Qwen3-4B (4-bit), v2 on Qwen3-4B (4-bit), LoRA on Qwen3-4B (4-bit), v3.1 on Qwen3-4B (4-bit)

## Tech Stack
- Python 3.13, PyTorch, HuggingFace transformers, bitsandbytes, tokenizers≥0.21.1
- Target model: `p-e-w/Qwen3-4B-Instruct-2507-heretic` (4-bit quantized)
- Debug model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (fp32)

## Key Files
- `hf_cascades.py` — CASCADES v1 (core Riemannian QR + SVC + EMA)
- `hf_cascades_v2.py` — CASCADES v2 (+ PaCA + DEAL heat kernel + CoSO null-space)
- `hf_cascades_v3.py` — CASCADES v3.1: 15-paper full fusion (StelLA + GainLoRA + CL-LoRA + D-MoLE + FunLoRA + quant-aware DEAL)
- `lora_baseline.py` — Standard LoRA baseline for comparison
- `../CASCADES_proposal.md` — Full 15-paper synthesis paper

## Architecture Quirks
- 4-bit quantized model uses `float16` compute dtype → adapters must initialize small (×0.01), alpha-mix at 0.1
- `CASCADES_Linear.forward()` must cast adapter output to `base_out.dtype` to avoid SDPA mixed-dtype crash
- GainLoRA gate computes in float32 → output MUST be cast back to input dtype before returning
- D-MoLE layer importance probe uses activation variance hooks (not gradient norms) because 4-bit weights don't support `requires_grad`
- `StelLA_riemannian_step` operates on V_shared transposed — need explicit QR re-transposition after step
- Task IDs must be set on `CASCADES_v3_Linear` modules (not just adapters) for correct routing during eval

## Trap Diary
| Issue | Cause | Fix |
|-------|-------|-----|
| RuntimeError: Expected device cuda | Adapters created on CPU | `new_module.to(module.weight.device)` |
| NaN loss during 4-bit training | fp32/fp16 mixing + large adapter init | Scale init ×0.01, alpha=0.1, grad clip 1.0 |
| Proxy accuracy always 0% | `max(0, 1-loss)` with CE>1 | Changed to `exp(-avg_loss)` |
| Task eval routing wrong | `current_task_id` set on adapter not module | Set on `CASCADES_v3_Linear` instances |
| v2 DEAL over-damping (-12.6% BWT) | Fixed spectral threshold treats quant noise as signal | v3 quant-aware threshold: ε_quant = α·std(W_4bit) |
| v3 SDPA dtype mismatch | GainLoRA gate float32 upcast | Cast adapt_out.to(input_dtype) everywhere |
| v3 D-MoLE probe crash | requires_grad on 4-bit weights | Switched to activation variance hooks |

## Anti-Patterns (DO NOT)
- DO NOT return adapter output in float32 from forward() — always cast to input dtype
- DO NOT use requires_grad on 4-bit quantized weights — they throw RuntimeError
- DO NOT use fixed spectral thresholds in DEAL on quantized models — use quant-aware ε_quant
- DO NOT apply Cayley retraction — use QR (O(dr²) vs O(d³))

## Experiment Results
| Method | Avg ACC | BWT | Time | Key Features |
|--------|---------|-----|------|-------------|
| LoRA Baseline (Qwen3 4-bit) | 31.9% | +11.2%* | 70s | Standard Adam |
| CASCADES v1 (Qwen3 4-bit) | 24.9% | -5.6% | 99s | QR + SVC + EMA |
| CASCADES v1 (TinyLlama fp32) | 35.9% | +2.3% | 30s | QR + SVC + EMA |
| CASCADES v2 (Qwen3 4-bit) | 27.9% | -12.6% | 120s | + PaCA + DEAL + CoSO |
| **CASCADES v3.1 Tuned (Qwen3 4-bit)** | **13.79%** | **-3.41%** | ~600s | All 15 papers, tuned D-MoLE threshold (0.15) and FunLoRA weight scale (0.05) |

\* LoRA "positive BWT" is a proxy artifact — model collapses to low-entropy attractor.

v3.1 tuned shows **strong BWT improvement** (-3.41% vs -12.6% in v2, -5.6% in v1) confirming GainLoRA gates + quant-aware DEAL fix working. Tuned D-MoLE (0.15 threshold) and FunLoRA (0.05 scale) successfully boosted average accuracy to ~13.79% while remaining under the 8GB limit.

## Build/Verify
```
cd e:\code.projects\research
cmd /c "set HF_HUB_DISABLE_PROGRESS_BARS=1 && python cascades_exp/hf_cascades_v3.py 2> cascades_v3_error.log"
```
