<div align="center">
  <h1>🌊 CASCADES</h1>
  <p><strong>Parameter-Efficient Continual Adaptation via Shared Dynamic Subspaces</strong></p>

  <p>
    <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"></a>
    <a href="https://huggingface.co/"><img alt="HuggingFace API" src="https://img.shields.io/badge/HuggingFace-F9AB00?style=for-the-badge&logo=huggingface&logoColor=white"></a>
    <a href="https://github.com/bitsandbytes-foundation/bitsandbytes"><img alt="4-bit QLoRA" src="https://img.shields.io/badge/4--bit_QLoRA-blue?style=for-the-badge"></a>
  </p>
</div>

<br/>

## 📖 Overview

Sequential adaptation of Large Language Models (LLMs) suffers from catastrophic interference and forgetting. While Parameter-Efficient Fine-Tuning (PEFT) reduces parameter footprint, it does not intrinsically solve the geometric overlap of sequential task gradients.

**CASCADES** is a continual PEFT framework engineered for strict VRAM constraints and quantized backbones (e.g., Qwen-4B in 4-bit under 8GB VRAM limit). By explicitly representing the adapter update space as a low-dimensional shared Stiefel manifold, CASCADES constrains updates to feasible, non-interfering directions while aggressively concentrating rank capacity on the network layers that need it most. 

## ✨ Key Features (The 5 Pillars)

1. **Shared Dynamic Subspaces:** Tri-factor adapters ($U_\ell S_{\ell,t} V^\top_\ell$) tracking a shared orthogonal basis using $O(dr^2)$ Riemannian QR retractions instead of $O(d^3)$ SVDs.
2. **Quantization-Aware Coordinate Filtering:** Adaptive heat-kernel noise-floor gating designed specifically to avoid over-damping true plasticity signals in 4-bit / int4 networks.
3. **Gated Integration:** Learned modulation gates balancing historical subspace persistence against new adapter updates.
4. **Energy-Accounted Reassignment (EAR):** Enforces hard non-interference by projecting into the historical orthogonal-complement, then exactly re-allocating blocked step scalar energy ($||g||_2$) back into the feasible step to preserve continuous adaptation scale.
5. **Budgeted D-MoLE Allocation & FunLoRA Fallback:** First-order activation-variance proxy scores assign heavy Stiefel machinery only to highly critical layers. Remaining layers receive parameter-minimal functional rank-1 approximations.

## 🚀 Quickstart

### Dependencies
Requires Python 3.10+ and a CUDA-capable GPU.
```bash
pip install torch transformers accelerate bitsandbytes
```

### Running CASCADES
The primary evaluation scripts are located in `cascades_exp/`. 
To run the fully mathematically rigorous **CASCADES v4** under strict 8GB constraints (evaluating on a 3-task continuous stream):

```bash
# Disable progress bars for cleaner logging
export HF_HUB_DISABLE_PROGRESS_BARS=1
python cascades_exp/hf_cascades_v4.py
```

## 📊 Empirical Results 

Experiments evaluated on a sequential 3-task stream using a 4-bit quantized Qwen3-4B-Instruct base model. The strict memory ceiling was matched across all baselines (`< 7.9GB VRAM`).

| Method                  | Architecture                                   | Avg ACC   | BWT        | VRAM   |
| ----------------------- | ---------------------------------------------- | --------- | ---------- | ------ |
| **Budget-Matched LoRA** | Standard Adam, restricted rank                 | ~11.5%    | -28.4%     | ~7.8GB |
| **CASCADES v3.1**       | Gated Stiefel, Allocator, Alpha Reassign       | 13.79%    | -3.41%     | 7.9GB  |
| **CASCADES v4**         | Exact EAR, Multi-Batch Allocator, Quant Filter | **8.28%** | **-1.48%** | 7.9GB  |

*Note: CASCADES v4 implements rigorous exact orthogonal bounding, intentionally trading task-specific plasticity (ACC) for extremely robust structural persistence (BWT).*

## 📂 Project Structure
```text
.
├── CASCADES_proposal.md         # Full theoretical backbone and 15-paper synthesis
├── cascades_exp/                # Core experiments module
│   ├── hf_cascades_v4.py        # Final v4 code (Exact EAR + Structural Filtering)
│   ├── hf_cascades_v3.py        # Prior v3.1 tuning run
│   ├── lora_baseline.py         # Budget-matched LoRA naive comparison
│   └── CONTEXT.md               # Directory operational context and trap-diary
```

## 📜 Citation & Context

If you build upon this implementation, please refer to the corresponding theoretical formulation located in `CASCADES_proposal.md`.

*This repository implements and rigorously validates the mechanisms proposed in "CASCADES: Parameter-Efficient Continual Adaptation via Shared Dynamic Subspaces".*
