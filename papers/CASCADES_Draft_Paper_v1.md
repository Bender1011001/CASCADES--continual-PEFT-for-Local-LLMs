# CASCADES: Parameter-Efficient Continual Adaptation via Shared Dynamic Subspaces

## Abstract
Parameter-efficient fine-tuning (PEFT) reduces the cost of adapting large language models, but sequential adaptation still suffers from interference and catastrophic forgetting, especially under strict VRAM budgets and 4-bit quantized training. We propose CASCADES, a continual PEFT framework that (i) maintains a shared rank-$r$ adapter subspace per layer, (ii) allocates task-specific capacity without replay, and (iii) explicitly constrains updates to reduce interference while preserving step “energy” in feasible directions. CASCADES parameterizes each adapted linear map as a tri-factor update $\Delta W_{\ell,t} = U_\ell S_{\ell,t} V_\ell^\top$ where $U_\ell$ and $V_\ell$ are shared across tasks and constrained to the Stiefel manifold, and $S_{\ell,t}$ is a small task-specific core. $U_\ell$ and $V_\ell$ evolve via Riemannian gradient steps with QR retraction, avoiding $O(d^3)$ full decompositions and using only $O(dr^2)$ operations per layer. Interference is controlled through gated integration between new and prior task adapters, an orthogonal-complement constraint with energy-accounted reassignment, and a budgeted layer-wise allocator that concentrates rank on high-importance layers while using functional rank-1 adapters elsewhere. In 4-bit QLoRA regimes we additionally apply quantization-aware gradient shrinkage with a locally tracked noise floor, preventing filters from mistaking quantization structure for task signal. We evaluate CASCADES on sequential task streams under an 8GB VRAM cap and report average final-task accuracy and backward transfer (BWT) under budget-matched baselines.

## 1. Introduction
Continual adaptation in LLMs is fundamentally a constrained optimization problem: new-task gradients are informative but frequently overlap with directions that support prior tasks, so naïve sequential fine-tuning produces negative backward transfer. PEFT reduces parameter movement but does not solve the geometry of interference. CASCADES is built around a single thesis: if the adapter update space is explicitly represented as a low-dimensional subspace with a stable orthonormal basis, then (a) interference can be expressed as overlap in that basis, (b) updates can be constrained to feasible directions with minimal compute overhead, and (c) limited VRAM can be spent where it matters most.

CASCADES is designed for the realistic regime your prototype targets: 4-bit quantized base model, frozen backbone weights, adapters trained under tight memory constraints, and sequential tasks with no replay buffer. The code-path you already implemented follows this structure: Stiefel-style Riemannian steps with QR retraction, shared adapter factors plus per-task cores, gating between branches, quantization-aware shrinkage, and selective layer allocation using an activation-variance heuristic in 4-bit mode.

### Contributions
*(1) Shared dynamic subspace adapters*: A tri-factor adapter parameterization with shared Stiefel bases $U_\ell,V_\ell$ and a task core $S_{\ell,t}$, enabling continual adaptation with low overhead and stable geometry.
*(2) Interference-constrained updates with energy accounting*: A projection-and-reassignment operator that enforces zero update in occupied directions while preserving step scale in the feasible subspace.
*(3) Budgeted allocation under VRAM limits*: A cheap, first-order importance score to allocate rank to critical layers and fall back to functional rank-1 adapters in non-critical layers.
*(4) Quantization-aware filtering*: A noise-floor–aware shrinkage operator designed specifically for 4-bit regimes to prevent over-damping of true gradients.

## 2. Problem setup and notation
We observe a sequence of tasks $t = 1 \dots T$. Each task provides a stream of samples $(x,y)$ from $\mathcal{D}_t$. We assume no replay buffer (no stored samples from $\mathcal{D}_i$ for $i < t$). Let a transformer have linear layers $W_\ell \in \mathbb{R}^{d_{out} \times d_{in}}$ for layer $\ell$.

We measure continual performance using a task-by-task accuracy matrix $A$ where $A_{t,i}$ is accuracy on task $i$ after finishing training up to task $t$. Standard metrics:
* Final average accuracy: $\text{ACC} = \frac{1}{T} \sum_i A_{T,i}$
* Backward transfer: $\text{BWT} = \frac{1}{T-1} \sum_{i<T} (A_{T,i} - A_{i,i})$

*(Note: We formally measure proxy accuracy as $\exp(-\text{avg\_loss})$ in our prototype).*

## 3. CASCADES method

### 3.1 Shared tri-factor adapters with Stiefel constraints
For each adapted linear layer $\ell$, CASCADES adds a low-rank update $\Delta W_{\ell,t}$:
$$ \Delta W_{\ell,t} = U_\ell S_{\ell,t} V_\ell^\top $$
where $U_\ell \in \text{St}(d_{out}, r)$, $V_\ell \in \text{St}(d_{in}, r)$, and $S_{\ell,t} \in \mathbb{R}^{r \times r}$.
$U_\ell$ and $V_\ell$ are shared across tasks; $S_{\ell,t}$ is task-specific (one core per task, per layer). In forward pass:
$$ h_{out} = W_\ell h_{in} + \alpha \cdot (U_\ell S_{\ell,t} V_\ell^\top h_{in}) $$

**Stiefel updates via Riemannian gradient and QR retraction**
Given a Euclidean gradient $G$ on $U$, the Stiefel Riemannian gradient is:
$$ G_R = G - U \cdot \text{sym}(U^\top G), \quad \text{sym}(A) = \frac{A + A^\top}{2} $$
Then retract with QR:
$$ U \leftarrow \text{qf}(U - \eta G_R) $$
The cost per retraction is QR on a $d_{out} \times r$ matrix: $\mathcal{O}(d_{out} r^2)$, avoiding full $O(d^3)$ SVD.

### 3.2 Gated integration to regulate interference
Even with a shared basis, new tasks can push $S_{\ell,t}$ and subspace evolution in directions that degrade prior tasks. CASCADES includes a learned gate $g_{\ell,t} \in (0,1)$ that scales the adapter contribution:
$$ \Delta h_{\ell,t} = g_{\ell,t} \cdot (U_\ell S_{\ell,t} V_\ell^\top h_{in}) $$

The gate allows controlling the magnitude of new updates. Regularizing $g_{\ell,t}$ relies on interference estimation.

### 3.3 Orthogonal-complement constraint and energy-accounted reassignment
Hard non-interference is enforced by projecting updates away from occupied (historically sensitive) directions.

**Occupied subspace and projector**
Maintain an orthonormal basis $B_\ell \in \mathbb{R}^{d \times k}$ for “occupied” directions (a sketch of historically important gradients in U-space and V-space). Define $P_\ell = B_\ell B_\ell^\top$. For any gradient $g$:
$$ g_{free} = (I - P_\ell) g $$
$$ g_{occ} = P_\ell g $$

**Energy-Accounted Reassignment (EAR)**
The key issue with hard projection is step shrinkage when $g$ overlaps with occupied space. EAR restores step scale while staying in the feasible subspace:
$$ g_{EAR} = \frac{\|g\|_2}{\|g_{free}\|_2 + \epsilon} \cdot g_{free} $$
Because $g = g_{free} + g_{occ}$ and $g_{free} \perp g_{occ}$, Euclidean norm is exactly preserved: $\|g\|_2^2 = \|g_{free}\|_2^2 + \|g_{occ}\|_2^2$.

### 3.4 Quantization-aware shrinkage (4-bit regime)
In 4-bit QLoRA settings, aggressive filtering can annihilate task signal. We define $\epsilon_{quant}$ as a function of quantization step $\Delta$ ($\text{variance} \approx \Delta^2/12$).
* Stage 1 (noise-floor gating): if $\|g\| < \epsilon_{quant} \rightarrow$ treat as noise and zero.
* Stage 2 (smooth shrinkage): otherwise apply structural soft-decay in subspace coordinates.

### 3.5 Budgeted layer-wise allocation and rank fallback
**Importance scoring**
For 4-bit frozen base weights, we use an activation variance heuristic via forward hooks (averaged over multiple batches for stability). Layers exceeding a variance threshold $\tau$ are marked as critical.

**Selective injection and Fallback**
Adapters are only injected into critical projection modules. Non-critical modules use functional rank-1 adapters which are $\mathcal{O}(d)$ parameters but possess non-linear expressivity.

## 4. Training algorithm

We summarize the entire training loop over a task sequence.

**Algorithm 1: CASCADES sequential training (per task)**
```text
Inputs: model with frozen backbone weights; target layer set L*; rank r; VRAM budget; tasks t=1..T
State per layer ℓ∈L*: shared Uℓ,Vℓ; occupied basis Bℓ; per-task core Sℓ,t; gate params ϕℓ

For t in 1..T:
  1. Allocate critical layers: compute activation variance on probe batches; mark ℓ critical if score ≥ τ.
  2. Inject adapters: deploy full tri-factor CASCADES in critical layers, functional rank-1 elsewhere.
  3. Initialize Sℓ,t for critical layers (e.g., identity) and set task ID in wrappers.
  4. For each minibatch (x,y) from Dt:
       a. Forward/backward: compute grads for Uℓ,Vℓ,Sℓ,t, and gates.
       b. Apply quantization-aware shrinkage (Stage 1 & 2) to U/V gradients.
       c. Update EMA gradients buffer for stability.
       d. Orthogonal projection: apply EAR to grads using Bℓ projector.
       e. Update Uℓ,Vℓ with Riemannian step + QR retraction.
       f. Update Sℓ,t and gate params with Euclidean optimizer (Adam).
  5. End-of-task: update occupied basis sketch Bℓ from EMA gradients.
  6. Evaluate A_{t,i} for i≤t.
```

## 5. Complexity and memory accounting
Per adapted layer $\ell$:
* **Parameters**: Shared $U_\ell$ ($d_{out} \times r$), $V_\ell$ ($d_{in} \times r$); Per-task core $S_{\ell,t}$ ($r \times r$ per task); Fallback rank-1 ($\mathcal{O}(d)$).
* **Compute**: Riemannian gradient uses $\mathcal{O}(dr)$ matrix multiplies and $\mathcal{O}(dr^2)$ QR factorization.

## 6. Experimental protocol & Results
**Setup**: 4-bit Qwen-class 4B, $< 8GB$ VRAM strict budget.
**Metrics**: Proxy ACC $\exp(-\text{loss})$ and BWT.

### Experimental Results

| Method              | Architecture                             | Avg ACC | BWT    | VRAM   |
| ------------------- | ---------------------------------------- | ------- | ------ | ------ |
| Budget-Matched LoRA | Standard Adam, restricted rank           | ~11.5%  | -28.4% | ~7.8GB |
| CASCADES v3.1       | Gated Stiefel, Allocator, Alpha Reassign | 13.79%  | -3.41% | 7.9GB  |
| CASCADES v4         | EAR, Multi-Batch Allocator, Quant Filter | 8.28%   | -1.48% | 7.9GB  |

## 7. Practical notes and failure modes
* **Quantization interaction**: The noise-floor threshold must be conservative. Aggressive limits destroy fine-grained gradients in QLoRA.
* **Gate collapse**: Start gates positively biased so adaptation isn't starved.
* **Allocator brittleness**: Use small moving averages over batches (e.g. 3) to stabilize the activation variance heuristic.

---

## Appendix A: EAR Code Patch
In `hf_cascades_v4.py`, the exact mathematical representation of Energy-Accounted Reassignment (EAR) has been integrated as follows:

```python
def cllora_gradient_reassign(grad, null_sketch, alpha=1.0):
    if not ENABLE_CLLORA_REASSIGN or null_sketch is None:
        return grad
    
    Q, _ = torch.linalg.qr(null_sketch)
    occupied_component = Q @ (Q.T @ grad)
    null_component = grad - occupied_component
    
    grad_energy = grad.norm()
    null_energy = null_component.norm()
    
    # EXACT EAR SCALING: g_EAR = ( ||g||_2 / (||g_free||_2 + eps) ) * g_free
    if grad_energy > 1e-8 and null_energy > 1e-8:
        return (grad_energy / (null_energy + 1e-8)) * null_component
        
    return null_component
```

---

## Appendix B: CASCADES Architecture Ablation Template

| Method             | Shared Subspace | DEAL Filter | GainLoRA Gate | D-MoLE Allocator | EAR Projection | Avg ACC | BWT    |
| ------------------ | --------------- | ----------- | ------------- | ---------------- | -------------- | ------- | ------ |
| Baseline LoRA      | -               | -           | -             | -                | -              | ~11.5%  | -28.4% |
| Shared Base        | ✓               | -           | -             | -                | -              |         |        |
| Shared + Filter    | ✓               | ✓           | -             | -                | -              |         |        |
| Shared + Allocator | ✓               | -           | -             | ✓                | -              |         |        |
| Shared + EAR       | ✓               | -           | -             | -                | ✓              |         |        |
| Full CASCADES v4   | ✓               | ✓           | ✓             | ✓                | ✓              | 8.28%   | -1.48% |
