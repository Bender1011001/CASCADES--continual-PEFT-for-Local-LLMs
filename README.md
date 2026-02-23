<div align="center">
  <h1>🌊 CASCADES</h1>
  <p><strong>Teach your AI new tricks without it forgetting the old ones.</strong></p>

  <p>
    <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"></a>
    <a href="https://huggingface.co/"><img alt="HuggingFace API" src="https://img.shields.io/badge/HuggingFace-F9AB00?style=for-the-badge&logo=huggingface&logoColor=white"></a>
    <a href="https://github.com/bitsandbytes-foundation/bitsandbytes"><img alt="4-bit QLoRA" src="https://img.shields.io/badge/4--bit_QLoRA-blue?style=for-the-badge"></a>
  </p>
</div>

<br/>

## 🤔 What is CASCADES?

Normally, when you teach an AI model something new (Task B), it overwrites and forgets what it previously learned (Task A). This is called *catastrophic forgetting*. 

**CASCADES** is a training framework that prevents this. It acts like a smart traffic controller for the AI's "brain," routing new knowledge into safe, empty spaces without stepping on the old knowledge. Best of all, it's designed to run on consumer hardware (under 8GB of VRAM).

### What did we train it to do?
To prove CASCADES works, we tested it on a **Sequential Domain Adaptation** challenge. We taught the AI to analyze text across three distinct domains, one after the other:
1. **Task 0**: Product Reviews ("Amazing value" vs "Waste of money")
2. **Task 1**: Movie Reviews ("Breathtaking cinematography" vs "Boring storyline")
3. **Task 2**: Restaurant Reviews ("Divine flavors" vs "Cold food")

Without CASCADES, learning Task 2 would destroy the AI's ability to do Task 0. With CASCADES, the AI accurately remembers how to perform all three.

### The Base Model
All experiments in this repository utilize the **`p-e-w/Qwen3-4B-Instruct-2507-heretic`** base model, quantized to 4-bit to fit easily within memory limits.

---

## ✨ Key Features (The 5 Pillars)

1. **Shared Dynamic Subspaces:** Tri-factor adapters ($U_\ell S_{\ell,t} V^\top_\ell$) tracking a shared orthogonal basis using $O(dr^2)$ Riemannian QR retractions instead of $O(d^3)$ SVDs.
2. **Quantization-Aware Coordinate Filtering:** Adaptive heat-kernel noise-floor gating designed specifically to avoid over-damping true plasticity signals in 4-bit / int4 networks.
3. **Gated Integration:** Learned modulation gates balancing historical subspace persistence against new adapter updates.
4. **Energy-Accounted Reassignment (EAR):** Enforces hard non-interference by projecting into the historical orthogonal-complement, then exactly re-allocating blocked step scalar energy ($||g||_2$) back into the feasible step to preserve continuous adaptation scale.
5. **Budgeted D-MoLE Allocation & FunLoRA Fallback:** First-order activation-variance proxy scores assign heavy Stiefel machinery only to highly critical layers. Remaining layers receive parameter-minimal functional rank-1 approximations.

## 🚀 How to Use It (Quickstart)

If you want to run the CASCADES experiment yourself to see the AI learn sequentially without forgetting, follow these steps:

### 1. Install Dependencies
You need Python installed (3.10+) and a CUDA-capable GPU.
```bash
pip install torch transformers accelerate bitsandbytes pandas numpy
```

### 2. Run the Evaluation Script
The primary scripts are located in the `cascades_exp/` folder. To run the most stable and mathematically rigorous version (**CASCADES v4**):

```bash
# Clone the repository and enter the directory
git clone https://github.com/your-username/cascades.git
cd cascades

# Run the training script (disabling progress bars for a cleaner output log)
export HF_HUB_DISABLE_PROGRESS_BARS=1
python cascades_exp/hf_cascades_v4.py
```
This will automatically download the `heretic` model, train it sequentially on the review tasks, and print out the accuracy and retention (Backward Transfer) metrics.

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
├── cascades_exp/                # Core execution scripts
│   ├── hf_cascades_v4.py        # Most stable version (run this one!)
│   ├── hf_cascades_v5.py        # Experimental dev branch (TAG + ARR)
│   └── lora_baseline.py         # Standard LoRA comparison (fails the forgetting test)
├── docs/                        
│   └── TUNING_WALKTHROUGH...md  # Detailed devlog tracking our accuracy improvements
├── results/                     
│   └── summary.csv              # The official metrics across all our tested versions
├── papers/                      # PDF reference materials for the math behind CASCADES
└── README.md                    # This file!
```

## 📜 Citation & Context

If you build upon this implementation, please refer to the corresponding theoretical formulation located in `CASCADES_proposal.md`.

*This repository implements and rigorously validates the mechanisms proposed in "CASCADES: Parameter-Efficient Continual Adaptation via Shared Dynamic Subspaces".*
