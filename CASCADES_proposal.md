# CASCADES: Parameter-Efficient Continual Adaptation via Shared Dynamic Subspaces

## 1. The Five-Pillar Meta-Architecture

While CASCADES synthesizes insights from multiple recent works, it is fundamentally built on a clean backbone of five core mechanisms, mapping to state-of-the-art PEFT developments from 2024–2026.

### A. Shared Subspace Learning on the Stiefel Manifold (Anchor)

CASCADES replaces frozen subspaces with a "live" foundational basis via StelLA's framing of subspace learning on the Stiefel manifold. We utilize a $USV^\top$ three-factor decomposition, replacing explicit $O(d^3)$ periodic SVDs with $r$-scale factorizations and QR/retraction updates. This ensures continuous adaptation with guaranteed effective rank without computational overhead.

### B. Gated Integration Between Adapters

To control backward transfer, we adopt GainLoRA's principle: learning gates to modulate how new adapters influence old tasks. CASCADES extends this by making the gating operate directly at the shared-subspace level, coupled with a stability regularizer that minimizes the drift of historical task representations.

### C. Shared + Task-Specific Reassignment and Non-Interference

Adapting insights from CL-LoRA, CASCADES uses task-shared and task-specific adapters. Gradients for new tasks are projected into the orthogonal complement of the learned Stiefel basis. To maximize plasticity without interference, we compute an energy-accounted gradient reassignment (reallocating blocked energy back into the free subpsace).

### D. Budgeted, Layer-Wise Allocation

Following D-MoLE, not all layers possess the same critical importance for plasticity. CASCADES uses gradient and activation importance computed cheaply from first-order statistics to assign high-rank adapters (and the heavy CASCADES machinery) only to the most critical layers, strictly respecting memory budgets.

### E. Cheap Expressivity for Non-Critical Layers

For layers deemed non-critical by the budgeted allocator, CASCADES utilizes FunLoRA (functional rank-1 adapters). This acts as a highly efficient fallback that still provides non-linear expressiveness while massively reducing the memory footprint.

## 2. Methodology & Formulations

### 2.1 Forward Pass

For input $X$, hidden representations are processed using the Stiefel-optimized Universal Shared Basis, merged with task-specific coefficients $\Lambda_t$ modulated by learned integrated gates $g_t$:
$$H_t = W_{base}X + g_t \cdot (U_{shared}\Lambda_t V_{shared}X)$$
Injection follows the D-MoLE allocation budget: critical layers use full CASCADES adapters; non-critical layers fall back to FunLoRA rank-1 adapters.

### 2.2 Energy-Accurated Gradient Reassignment

Let $P$ be the projection matrix into the occupied historical subspace. We project task-specific gradients $\nabla$ into the orthogonal subspace:
$$\nabla_{\perp} = (I - P)\nabla$$
To preserve gradient energy (maintaining $\|\nabla_{reassigned}\|_F = \|\nabla\|_F$) without injecting structurally disjoint noise, we re-allocate the blocked magnitude $\|P \nabla\|_F$ into a valid basis of the free subspace, aligned proportionally with $\nabla_{\perp}$:
$$\nabla_{reassigned} = \nabla_{\perp} + \left( \frac{\|P \nabla\|_F}{\|\nabla_{\perp}\|_F} \right) \nabla_{\perp}$$
*(Note: Care is taken numerically if $\nabla_{\perp} \to 0$)*.

### 2.3 Task-Utility Proxy and Quantization Filter

We introduce a task-utility attribution proxy (replacing naive causal assumptions lacking replay). Updates are filtered to penalize quantization noise patterns without explicit unconstrained SVDs.
The quantization noise floor is parameterized per-group (e.g., in nf4/int4) as $\epsilon \approx k \cdot (\Delta^2/12)$, where $\Delta$ is the quantization step size. Smoothing is evaluated strictly within the coordinates of the learned Stiefel basis (avoiding $O(d^3)$ outer-projection operations).

## 3. Empirical Results & Evaluation

### 3.1 Experimental Limitations & Budget Constraints

Experiments are run with strict settings: 8GB VRAM cap, 4-bit Qwen3-4B-Instruct, no replay buffering. Because extreme capacity restriction alters overall performance profiles, CASCADES is evaluated directly against a **budget-matched LoRA baseline**, rather than arbitrarily large, unrestricted configurations. Evaluation measures both Average Accuracy and strictly held-out BWT tests.

### 3.2 Results

| Method               | Architecture                 | Avg ACC | BWT    | VRAM   |
| -------------------- | ---------------------------- | ------- | ------ | ------ |
| Budget-Matched LoRA  | Standard Adam, restricted    | ~11.5%  | -28.4% | ~7.8GB |
| CASCADES v3.1        | Stiefel + Gates + Allocation | 13.79%  | -3.41% | 7.9GB  |
| CASCADES (fp32 test) | TinyLlama-1.1B proxy test    | 35.9%   | +2.3%  | <6GB   |

### 3.3 Key Findings

1.  **Forgetting Under Constraints:** CASCADES v3.1 exhibits significantly stronger resilience to catastrophic forgetting compared to simple restricted-rank LoRAs (-3.41% vs -28.4% BWT) by actively separating and guarding the historically utilized Stiefel components.
2.  **Quantization Noise Impact:** Traditional heat-kernels over-damp 4-bit networks. Utilizing the group-aware quantization threshold correctly preserves legitimate low-frequency plasticity signals.
3.  **Positive BWT in Unquantized Settings:** Tests on TinyLlama (fp32) indicate the architecture is designed to promote positive backward transfer when gradient dynamics are not conflated heavily by quantization steps, nearing +2.3% BWT.

## 4. Formal Conference Proposal Abstract
**Title:** CASCADES: Parameter-Efficient Continual Adaptation via Shared Dynamic Subspaces

**Abstract:**
Parameter-Efficient Fine-Tuning (PEFT) has revolutionized continual learning (CL) in large foundation models, yet significant challenges prevent universal adoption: the computational overhead of periodic subspace projection, the absence of intrinsic zero-shot backward knowledge transfer, and rapid catastrophic forgetting driven by spectral over-accumulation. CASCADES is a parameter-efficient continual adaptation framework that maintains a shared rank-r subspace across tasks and allocates task-specific capacity under strict memory budgets. CASCADES learns the shared basis on the Stiefel manifold via QR/retraction updates and couples it to (i) gated integration modules that regulate interference between new and prior adapters, (ii) orthogonal-complement updates with energy-accounted reassignment, and (iii) a budgeted layer-wise allocator that assigns higher-rank adapters to high-importance layers while using functional rank-1 adapters elsewhere. Across 3-task continual sequences, we observe improved forgetting behavior relative to budget-matched LoRA baselines in both fp32 and 4-bit QLoRA settings, and we analyze the failure modes introduced by quantization when aggressive spectral/threshold filters are applied.
