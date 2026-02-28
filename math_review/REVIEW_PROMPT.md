# Mathematical Review Request — CASCADES v9

You are reviewing the mathematical foundations of **CASCADES** (Continual Adaptation via Shared Adaptive Dynamic Evolving Subspaces), a continual learning system for neural networks that uses Riemannian geometry on the Stiefel manifold to prevent catastrophic forgetting.

## What CASCADES Does

CASCADES adapts a frozen pre-trained LLM by injecting lightweight adapters of the form:

$$\Delta W = U \cdot S_k \cdot V$$

where $U \in \mathbb{R}^{d_{out} \times r}$ and $V \in \mathbb{R}^{r \times d_{in}}$ are shared orthonormal bases constrained to the Stiefel manifold $\text{St}(r, d)$, and $S_k \in \mathbb{R}^{r \times r}$ is a task-specific "core" matrix selected from a learned pool via gating.

The key claim: by keeping $U, V$ on the Stiefel manifold and learning only $S_k$ per task, knowledge from previous tasks is preserved in the shared subspace while new tasks get their own core matrices.

## Files (read in order)

1. **1_math_ops.py** — Core Riemannian operations:
   - `qr_retraction()`: Projects back onto St(r,d) after gradient steps
   - `riemannian_gradient()`: Projects Euclidean gradients to the tangent space $T_U \text{St}(r,d)$
   - `stella_step()`: Combined Riemannian update (project + retract)
   - `ear_filter()`: Elastic Anchor Regularization — decomposes gradient into "anchor" (old task) and "free" (new task) subspaces, rescales the free component to preserve gradient norm
   - `deal_filter()`: Dimensional Entropic Attenuation Layer — gates gradient dimensions below a quantization noise floor
   - `svc_clip()`: Singular Value Clipping — soft thresholds singular values to prevent rank collapse

2. **2_adapters.py** — The adapter architecture:
   - `FunLoRA`: Rank-1 adapters using $\sin(Ax + b)$ nonlinear activation
   - `ResonantCore`: Gated core pool with $k$ core matrices and softmax attention gating
   - `CASCADES_v6_Adapter`: Full adapter with $U$, $V$ on Stiefel, liquid core pool, streaming PaCA masking, EMA buffers
   - `full_descent_step()`: The Riemannian descent — gradient → DEAL filter → EAR filter → Riemannian projection → QR retraction

3. **3_sleep.py** — Bio-inspired offline consolidation (runs between tasks):
   - SVD pruning of weak core dimensions
   - Cross-adapter redundancy detection and merging
   - Stiefel re-orthogonalization (QR correction for numerical drift)
   - Synaptic Homeostasis Scaling (prevents runaway core norms)

4. **4_config.py** — Ablation configuration (which components are active)
5. **5_injection.py** — How adapters are injected into the transformer
6. **6_hf_cascades_reasoning.py** — Training loop orchestration
7. **7_test_math.py** — Tests verifying manifold properties
8. **8_test_adapters_v9.py** — Tests verifying adapter correctness

## What I Need You To Analyze

### A. Correctness of Manifold Operations

1. Does `riemannian_gradient()` correctly project to $T_U \text{St}(r,d)$? The formula used is $G - U(U^T G + G^T U)/2$. Verify this is the correct orthogonal projection onto the tangent space.
2. Does `qr_retraction()` correctly retract back to the Stiefel manifold? Is QR retraction a valid retraction on St(r,d)?
3. In `stella_step()`, is the order of operations (project → step → retract) mathematically sound?
4. Are there numerical stability concerns with any of these operations (e.g., near-singular R in QR)?

### B. EAR Filter Analysis

The Elastic Anchor Regularization decomposes the gradient into an "anchor" component (projection onto previous task directions) and a "free" component, then rescales the free component to preserve the total gradient norm.

1. Is this decomposition mathematically valid?
2. Does the norm preservation property actually hold?
3. Could this cause issues with gradient descent convergence?

### C. Sleep Consolidation Correctness

1. **Re-orthogonalization**: After QR decomposition $U = QR$, cores are transformed as $S \leftarrow R \cdot S$. Does this correctly preserve $\Delta W = U S V^T = Q(RS)V^T$?
2. **V_shared handling**: $V$ is $(r, d_{in})$ with orthonormal rows. The code decomposes $V^T = Q_{V^T} R_{V^T}$, then sets $V_{new} = Q_{V^T}^T$. Is this the correct re-orthogonalization for row-orthonormal matrices?
3. **SVD pruning**: Weak dimensions are decayed by $0.1\times$. Does this soft decay preserve the subspace correctly, or should hard pruning (zeroing out) be used?

### D. Theoretical Gaps or Improvements

1. Is the Stiefel manifold the right choice, or would the Grassmann manifold $\text{Gr}(r, d)$ be more natural for subspace learning?
2. The EAR filter and DEAL filter are applied sequentially before the Riemannian projection. Could these filters interfere with the manifold constraint?
3. Are there any conditions under which the system could diverge, get stuck, or lose rank?
4. The streaming PaCA (Principal Angle Component Analysis) uses an EMA sketch to estimate principal angles. Is this a valid approximation?

### E. Suggestions

What mathematical improvements would you make? Consider:

- Better retraction methods (Cayley, geodesic)
- Alternative manifold choices
- Convergence guarantees
- Tighter bounds on backward transfer

## Current Results

- **Positive BWT (+2.31%)**: The model improves on old tasks after learning new ones
- **166 tests pass** including manifold property verification
- Training on 4B parameter Qwen model with rank-8 adapters
