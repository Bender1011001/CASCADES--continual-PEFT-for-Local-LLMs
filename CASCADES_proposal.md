# CASCADES: Parameter-Efficient Continual Adaptation via Shared Dynamic Subspaces and Zero-Shot Reasoning

## 1. Introduction & The Meta-Architecture

Parameter-Efficient Fine-Tuning (PEFT) has revolutionized continual learning (CL) in large foundation models; however, significant challenges prevent universal adoption: catastrophic forgetting due to spectral over-accumulation, the computational overhead of periodic subspace projection, and the failure of highly quantized models (e.g., 4-bit) to retain complex task mappings.

CASCADES (Continual Adaptation via Shared Dynamic Subspaces) synthesizes these challenges into a single cohesive meta-architecture. In Version 9 (v9 Pro), CASCADES is built on five core mechanisms optimized for consumer-grade hardware (8GB VRAM).

### A. Shared Subspace Learning on the Stiefel Manifold (Anchor)

CASCADES replaces frozen subspaces with a "live" foundational basis via subspace learning on the Stiefel manifold. We utilize a $USV^\top$ three-factor decomposition, avoiding explicit $O(d^3)$ periodic SVDs in favor of $r$-scale factorizations and QR/retraction updates. This ensures continuous adaptation with a guaranteed effective rank without computational overhead.

### B. Gated Integration Between Adapters

To control backward transfer, we learn integration gates to modulate how new adapters influence old tasks. CASCADES extends naive gating by making it operate directly at the shared-subspace level, coupled with a stability regularizer that minimizes the drift of historical task representations.

### C. Orthogonal Complement & Energy-Accounted Reassignment

CASCADES utilizes both task-shared and task-specific adapters. Gradients for new tasks are projected into the orthogonal complement of the learned Stiefel basis. To maximize plasticity without interference, we compute an **energy-accounted gradient reassignment**—reallocating blocked gradient magnitude mathematically back into the free subspace so no plasticity signal is lost.

### D. Budgeted, Layer-Wise Allocation (D-MoLE)

Not all transformer layers possess the same critical importance for plasticity. CASCADES incorporates Dynamic Mixture-of-Low-Rank-Experts (D-MoLE), using gradient and activation importance computed cheaply from first-order statistics. It dynamically assigns high-rank Stiefel adapters (the heavy machinery) only to the most critical layers, strictly respecting VRAM memory constraints.

### E. FunLoRA: Cheap Expressivity for Non-Critical Layers

For layers deemed non-critical by the D-MoLE allocator, CASCADES utilizes functional rank-1 adapters (FunLoRA). This acts as a highly efficient fallback that still provides non-linear expressiveness while massively reducing the memory footprint.

## 2. Methodology & Formulations

### 2.1 Forward Pass

For input $X$, hidden representations are processed using the Stiefel-optimized Universal Shared Basis, merged with task-specific coefficients $\Lambda_t$ and modulated by learned integrated gates $g_t$:
$$H_t = W_{base}X + g_t \cdot (U_{shared}\Lambda_t V_{shared}X)$$

Injection follows the D-MoLE allocation budget. Critical layers use the full CASCADES adapter topology; non-critical layers fall back to FunLoRA rank-1 adapters.

### 2.2 Energy-Accounted Gradient Reassignment

Let $P$ be the projection matrix into the historically occupied subspace. We project task-specific gradients $\nabla$ into the orthogonal subspace:
$$\nabla_{\perp} = (I - P)\nabla$$
To preserve gradient energy (maintaining $\|\nabla_{reassigned}\|_F = \|\nabla\|_F$) without injecting structurally disjoint noise, we re-allocate the blocked magnitude $\|P \nabla\|_F$ into a valid basis of the free subspace, aligned proportionally with $\nabla_{\perp}$:
$$\nabla_{reassigned} = \nabla_{\perp} + \left( \frac{\|P \nabla\|_F}{\|\nabla_{\perp}\|_F} \right) \nabla_{\perp}$$

### 2.3 Task-Utility Proxy and Quantization Filter

Traditional heat-kernels over-damp 4-bit networks. We introduce a quantization noise floor parameterized per-group (e.g., in `nf4`) as $\epsilon \approx k \cdot (\Delta^2/12)$. Smoothing is evaluated strictly within the coordinates of the learned Stiefel basis, isolating legitimate low-frequency plasticity signals from bit-width noise.

## 3. The Zero-Shot Reasoning Data Overhaul

A key discovery in CASCADES v9 is that **model capacity on highly quantized continual sequences is bottlenecked by rote memorization targets**.

To unlock true continual generalization, the training sequence was overhauled from standard causal generation tasks to rigorous **Zero-Shot Algorithmic & Logical Reasoning** via detailed `<think>` tags:

1. **Task 0 (Logic & Math):** 159 high-quality examples covering Fermi estimation, paradoxes, and complex derivations (replacing rote trivia).
2. **Task 1 (Critical Analysis):** 184 examples enforcing extreme anti-sycophancy, concise scientific rigor, and debunking of empirical myths.
3. **Task 2 (Algorithmic Synthesis):** 152 examples replacing standard scripts with pure Python algorithmic mechanics (dynamic programming, Kahn's algorithm, sliding windows).

By restructuring the targets to emphasize algorithmic steps (~150 words) rather than long multi-paragraph memorization, the loss landscape smoothed dramatically, enabling the 4B quantized model to learn efficiently without catastrophically forgetting prior distributions.

## 4. Empirical Results & Evaluation

### 4.1 Constraints

Experiments were executed under strict constraints: **8GB VRAM cap**, _4-bit Qwen3-4B-Instruct_, and **zero replay buffering**. CASCADES is evaluated against a budget-matched LoRA baseline to ensure parameter equivalency.

### 4.2 Results Matrix

| Method                          | Architecture                         | Avg ACC Proxy | BWT        | VRAM      |
| :------------------------------ | :----------------------------------- | :------------ | :--------- | :-------- |
| Budget-Matched LoRA             | Standard Adam, restricted            | ~11.5%        | -28.4%     | ~7.8GB    |
| CASCADES v3.1                   | Stiefel + Gates                      | 13.79%        | -3.41%     | 7.9GB     |
| **CASCADES v9 (Data Overhaul)** | **Stiefel + D-MoLE + Zero-Shot CoT** | **35.91%**    | **-1.46%** | **5.2GB** |

_Note: CASCADES unquantized (fp32) has demonstrated positive BWT (+2.01% on an 8B class model)._

### 4.3 Key Findings

1. **Recovery from Catastrophic Forgetting:** CASCADES v9 successfully reduced backward transfer regression to **-1.46%**, beating standard LoRA by a massive margin and surpassing our >-2% stability target on a severely memory-constrained model.
2. **Algorithmic Transfer:** Task 2 (Algorithmic Synthesis) maintained nearly 50% accuracy on highly complex problems, proving the model can isolate and continuously adapt deep internal algorithms without overwriting fundamental reasoning pathways.
3. **VRAM Efficiency:** The D-MoLE allocation and FunLoRA strategies successfully compacted the architecture footprint to 5.2GB, allowing for aggressive training sequences on 8GB consumer GPUs.

## 5. Conclusion & Formal Abstract

**Title:** CASCADES: Parameter-Efficient Continual Adaptation via Shared Dynamic Subspaces and Zero-Shot Reasoning

**Abstract:**
Parameter-Efficient Fine-Tuning (PEFT) holds immense promise for continual learning in foundation models, yet significant challenges prevent universal adoption—namely catastrophic forgetting, computational overhead of subspace projections, and the failure of highly quantized models to retain complex representations. CASCADES resolves this by maintaining a shared rank-$r$ subspace across tasks, allocating task-specific capacity under strict memory budgets. The architecture learns the shared basis on the Stiefel manifold via QR/retraction updates, coupled with gated integration modules, orthogonal-complement updates via energy-accounted reassignment, and a Dynamic Mixture-of-Low-Rank-Experts (D-MoLE) allocator. Furthermore, we demonstrate that restructuring continual sequences into strict zero-shot `<think>` reasoning tasks significantly aids plasticity in extreme quantization regimes (e.g., 4-bit). Without replay buffering, CASCADES achieves 35.91% average reasoning accuracy and near-zero forgetting (-1.46% BWT) on consumer-grade hardware (5.2GB VRAM profile).
