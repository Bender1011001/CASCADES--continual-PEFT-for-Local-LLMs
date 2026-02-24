# CASCADES (v9 Pro): The Cognitive Ecosystem

**Mathematically Secure Continual Learning for Abliterated LLMs under 8GB Constraints.**

CASCADES is an *Autopoietic Cognitive Ecosystem* designed to adapt fragile, abliterated ("Heretic") models continuously without catastrophic forgetting or representational collapse.

## 🏆 The 4B Heretic Breakthrough

Standard fine-tuning (LoRA) destroys the reasoning architecture of abliterated models. CASCADES solves this by constraining updates to Autopoietic Stiefel Manifolds.

| Method              | Model                | Backward Transfer (BWT) | Status                      |
| :------------------ | :------------------- | :---------------------- | :-------------------------- |
| **Standard LoRA**   | Qwen3-8B Heretic     | **-12.18%**             | **Catastrophic Forgetting** |
| **CASCADES v9 Pro** | **Qwen3-4B Heretic** | **+0.82%**              | **REASONING PRESERVED**     |

### Key Results
- **Proxy Accuracy**: 46.82% on reasoning task streams.
- **Hardware**: Benchmarked on a single RTX 4060 Ti (8GB VRAM).
- **Speed**: 2.4x throughput increase via D-MoLE dynamic expert routing.

## 🚀 Quick Start: Reproduce the Breakthrough

You can verify the zero-forgetting architecture locally on an 8GB GPU using the lightning reproduction script.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the lightning reproduction (approx. 5 mins)
python reproduce_the_breakthrough.py
```

## 🧠 Core Architecture

CASCADES v9 Pro introduces three "Biological" regulation mechanisms:
- **Dormant Core Distillation**: Polar factor extraction to preserve memories during layer demotion.
- **Riemannian Freeze**: Hibernation locks for manifold stability during expert dormancy.
- **Breathing Manifolds**: Native VRAM rank expansion/contraction ($r \to r \pm 1$).

## 📄 Research Paper
Details on the **Transpose Parity Bug** fix ($R^\top$ mixing) and the **GQA Scaling Paradox** can be found in [papers/CASCADES_v9_Final_Paper.md](papers/CASCADES_v9_Final_Paper.md).

## ⚠️ Limitations
While CASCADES successfully achieves positive BWT at scale, we observe a performance plateau at 8B (v1 Patch @ 32.97%) compared to the 4B breakthrough. See Section 7 of the paper for the mathematical diagnosis of the GQA Scaling Paradox.

---
*CASCADES is open-source research for the unconstrained reasoning community.*
