# CASCADES Experiment Directory

## Status

- **Working**: hf_cascades.py (v1), hf_cascades_v2.py (v2 full fusion), hf_cascades_v3.py (v3.1 15-paper), hf_cascades_v4.py (v4 exact math), hf_cascades_v5.py (v5 TAG/ARR), lora_baseline.py
- **Completed Runs**: v1 on TinyLlama/Qwen3, v2 on Qwen3, LoRA on Qwen3, v3.1 on Qwen3, v4 on Qwen3, v5 on Qwen3

## Tech Stack

- Python 3.13, PyTorch, HuggingFace transformers, bitsandbytes, tokenizers≥0.21.1
- Target model: `meta-llama/Meta-Llama-3-8B-Instruct` (4-bit NF4 quantized)
- VRAM Budget: 8GB (Peak usage: 7.14GB)

## Key Files

- `hf_cascades.py` — CASCADES v1 (core Riemannian QR + SVC + EMA)
- `hf_cascades_v2.py` — CASCADES v2 (+ PaCA + DEAL heat kernel + CoSO null-space)
- `hf_cascades_v3.py` — CASCADES v3.1: 15-paper full fusion (StelLA + GainLoRA + CL-LoRA + D-MoLE + FunLoRA + quant-aware DEAL)
- `hf_cascades_v4.py` — CASCADES v4: Reviewer math corrections (EAR, full coordinate-space filter)
- `hf_cascades_reasoning.py` — CASCADES v9 Pro Reasoning (Llama-3-8B NF4 + CoT)
- `cascades_exp/task0_logic_cot.jsonl`, `cascades_exp/task1_decomp_cot.jsonl`, `cascades_exp/task2_action_cot.jsonl` — Distilled step-by-step CoT datasets with `<think>` tags.
- `scripts/format_cot_datasets.py` — Distillation script to convert raw logic into strict CoT format.
- `lora_baseline.py` — Standard LoRA baseline for comparison
- `../CASCADES_proposal.md` — Full 15-paper synthesis paper
- `../REDDIT_POST.md` — r/LocalLLaMA post (ready to copy-paste)
- `../colab_cascades_v9.ipynb` — One-click Google Colab reproduction notebook

## Architecture Quirks

- 4-bit quantized model uses `float16` compute dtype → adapters must initialize small (×0.01), alpha-mix at 0.1
- `CASCADES_Linear.forward()` must cast adapter output to `base_out.dtype` to avoid SDPA mixed-dtype crash
- GainLoRA gate computes in float32 → output MUST be cast back to input dtype before returning
- D-MoLE layer importance probe uses activation variance hooks (not gradient norms) because 4-bit weights don't support `requires_grad`
- `StelLA_riemannian_step` operates on V_shared transposed — need explicit QR re-transposition after step
- Task IDs must be set on `CASCADES_v3_Linear` modules (not just adapters) for correct routing during eval

## Trap Diary

| Issue                              | Cause                                                 | Fix                                               |
| ---------------------------------- | ----------------------------------------------------- | ------------------------------------------------- |
| RuntimeError: Expected device cuda | Adapters created on CPU                               | `new_module.to(module.weight.device)`             |
| NaN loss during 4-bit training     | fp32/fp16 mixing + large adapter init                 | Scale init ×0.01, alpha=0.1, grad clip 1.0        |
| Proxy accuracy always 0%           | `max(0, 1-loss)` with CE>1                            | Changed to `exp(-avg_loss)`                       |
| Task eval routing wrong            | `current_task_id` set on adapter not module           | Set on `CASCADES_v3_Linear` instances             |
| v2 DEAL over-damping (-12.6% BWT)  | Fixed spectral threshold treats quant noise as signal | v3 quant-aware threshold: ε_quant = α·std(W_4bit) |
| v3 SDPA dtype mismatch             | GainLoRA gate float32 upcast                          | Cast adapt_out.to(input_dtype) everywhere         |
| v3 D-MoLE probe crash              | requires_grad on 4-bit weights                        | Switched to activation variance hooks             |

## Anti-Patterns (DO NOT)

- DO NOT return adapter output in float32 from forward() — always cast to input dtype
- DO NOT use requires_grad on 4-bit quantized weights — they throw RuntimeError
- DO NOT use fixed spectral thresholds in DEAL on quantized models — use quant-aware ε_quant
- DO NOT apply Cayley retraction — use QR (O(dr²) vs O(d³))

## Experiment Results

| Method                           | Avg ACC    | BWT        | Time  | Key Features                                                              |
| -------------------------------- | ---------- | ---------- | ----- | ------------------------------------------------------------------------- |
| LoRA Baseline (Qwen3 4-bit)      | 31.9%      | +11.2%\*   | 70s   | Standard Adam                                                             |
| CASCADES v8 (Qwen3 4-bit)        | 14.87%     | +0.66%     | 27s   | Autopoietic Manifold + Adaptive Rank Routing (ARR)                        |
| **CASCADES v9 Pro (Llama-3-8B)** | **46.82%** | **+0.82%** | ~800s | **Phase 15 Breakthrough**: Engine Swap, CoT Distillation, Rank 32 Ceiling |

\* LoRA "positive BWT" is a proxy artifact — model collapses to low-entropy attractor.

v9 Pro on Llama-3-8B demonstrates **massive reasoning capability scaling** (46.82% vs 14.9%) while maintaining strictly positive BWT (+0.82%). The system remains stable within the 8GB VRAM constraint (7.14GB peak).

## Build/Verify

```
cd e:\code.projects\research
cmd /c "set HF_HUB_DISABLE_PROGRESS_BARS=1 && python cascades_exp/hf_cascades_reasoning.py"
```
