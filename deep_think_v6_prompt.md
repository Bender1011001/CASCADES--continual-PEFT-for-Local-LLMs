# CASCADES v6: Boundary-less Streaming Architecture — Expert Deep Think Prompt

**Instructions for User:**
Attach the following files to your message when prompting the deep think model:
1. `cascades_exp/hf_cascades_reasoning.py` (The current v5.1 implementation with all math fixes applied)
2. `papers/CASCADES_Draft_Paper_v1.md` (Mathematical theory and ablation context)

---

## Context: Where We Are Now

You previously audited the CASCADES framework and identified three fatal mathematical contradictions and three VRAM autograd leaks. **All six have been patched and verified:**

1. **Double-Optimizer Annihilation** → `U_shared` and `V_shared` are now excluded from Adam. Only updated via Riemannian `full_descent_step()`.
2. **EAR/StelLA Non-Commutativity** → The descent pipeline now maps to Stiefel Tangent Space *first*, then applies EAR null-space constraint strictly within that tangent space, then re-projects to correct numerical drift.
3. **Basis-Destruction** → Historical task cores `Lam` are counter-rotated via `R_U @ Lam @ R_V.T` after every QR retraction.
4. **float32 Autograd Trap** → Eliminated. We now cast tiny adapter weights `.to(x.dtype)` instead of the massive activation tensor.
5. **FunLoRA Activation Bloat** → Replaced with `FunLoRA_Activation(torch.autograd.Function)` that analytically fuses `sigmoid`/`tanh` derivatives.
6. **cuSOLVER Sync Thrashing** → SVC SVD is amortized to every 50 steps. Null-space QR is precomputed once per task boundary.

The patched framework now trains flawlessly on an 8GB VRAM GPU (Qwen3-4B 4-bit) at 256 context length with batch_size=1. Loss converges smoothly across all three tasks.

## Your Mission: Design Production-Ready Code for CASCADES v6

The current architecture still relies on **explicit task boundaries** (`task_id`, `store_task_gradients()`, `update_null_sketch()`). Real-world continual learning towards autonomous reasoning does not have these clean separations. Your job is to design the three architectural innovations that evolve CASCADES into a **boundary-less streaming system**.

I need you to produce **complete, production-ready PyTorch code** (not pseudocode) for each component, designed to drop directly into the existing `CASCADES_v3_Adapter` class. Each component must:
- Work within 8GB VRAM (Qwen3-4B 4-bit quantized, rank=8, 14 critical + 130 FunLoRA layers)
- Use `torch.no_grad()` where appropriate to prevent graph leaks
- Respect the Stiefel manifold constraint (all updates to `U_shared`/`V_shared` must stay on the manifold)
- Be mathematically rigorous with citations to the underlying theory

---

### Component 1: Contextual Dynamic Routing ("Liquid Core")

**Current Problem:** The adapter maintains a `ModuleDict` of discrete task cores `task_lambdas["0"]`, `task_lambdas["1"]`, etc. This requires knowing the task ID at inference and cannot handle distribution drift within a single task.

**Your Design Brief:**
- Replace the discrete `task_lambdas` dictionary with a **fixed pool of K shared Stiefel-cores** (e.g., K=4).
- During the forward pass, implement a lightweight **attention-pooling** mechanism that extracts the semantic centroid of the current hidden state `x`.
- Route this centroid through a **continuous gating network** (Dirichlet-parameterized softmax) that outputs interpolation weights `w_1, ..., w_K` summing to 1.
- Dynamically assemble the active task core on the fly: `Lam_active = sum(w_k * Core_k)`.
- The gating network must be tiny (< 10K parameters) and its backward pass must not leak through the Stiefel bases.

**Constraints:**
- The `full_descent_step()` must now update ALL K cores, but the counter-rotation `R_U @ Core_k @ R_V.T` must apply to each.
- The gating weights should be included in the Adam optimizer (they are Euclidean, not Riemannian).
- At initialization, the K cores should be orthogonally diversified (not identical copies).

**Deliverable:** Complete `__init__`, `forward`, and updated `full_descent_step` methods for the refactored adapter.

---

### Component 2: Streaming Frequent Directions ("Continuous EAR")

**Current Problem:** The null-space sketch `null_sketch_U` is only updated at explicit task boundaries via `update_null_sketch()`. Between boundaries, the model has no protection against forgetting. After a boundary, ALL directions are frozen equally regardless of recency.

**Your Design Brief:**
- Implement **Oja's Rule Streaming Sketch** that continuously digests the Riemannian tangent gradient into the null-sketch covariance matrix at every training step.
- Use a low-rank exponential moving average update: `C ← β_C * C + (1 - β_C) * g_tangent @ g_tangent.T`, where `C` is the covariance approximation of the occupied subspace.
- Periodically (every N steps), perform an eigendecomposition of `C` to extract the top-k principal directions as the "occupied subspace".
- This creates a **sliding window of plasticity**: recent knowledge is rigorously protected, but ancient, unused gradient directions slowly decay (simulating biological synaptic turnover).
- The decay rate `β_C` controls the "memory half-life". Propose a sensible default and explain its relationship to the number of training steps.

**Constraints:**
- The covariance matrix `C` is `(rank, rank)` — tiny. The eigendecomposition is on this small matrix, not the full `(d_out, rank)` basis.
- `C` must NOT be part of the autograd graph. Use `torch.no_grad()` and `register_buffer`.
- The eigendecomposition should be amortized (e.g., every 25 steps) to avoid sync thrashing.

**Deliverable:** Complete `streaming_ear_update()` method and the modified `precompute_null_space()` replacement. Show how it integrates into `full_descent_step`.

---

### Component 3: Temporal Dual-EMA Causal Masking ("Streaming PaCA")

**Current Problem:** The `paca_causal_mask` function compares the current gradient against a frozen list of `historical_U_grads` snapshots taken at task boundaries. This requires explicit task switches and grows linearly with the number of tasks.

**Your Design Brief:**
- Replace the discrete historical gradient list with two running EMAs:
  - **Fast EMA** (`ema_fast`): Tracks the immediate local data stream gradient direction over ~100 steps. Decay `β_fast ≈ 0.99`.
  - **Slow EMA** (`ema_slow`): Tracks the long-term structural gradient direction over ~5,000 steps. Decay `β_slow ≈ 0.9998`.
- At each step, compute the **temporal cosine similarity** between the incoming gradient and the Slow EMA.
- If this cosine similarity is **strongly negative** (i.e., the current data distribution is actively attacking historical knowledge), dynamically generate a PaCA-style causal mask that clamps plasticity on those specific gradient coordinates.
- The mask should be **soft** (sigmoid-scaled cosine similarity), not binary, to allow smooth transitions.
- Define a **conflict threshold** `τ` below which the mask activates. Propose a sensible default.

**Constraints:**
- Both EMAs are `register_buffer` (no autograd).
- The cosine similarity must be computed element-wise (per-coordinate masking), not globally.
- The mask must be applied BEFORE the Stiefel tangent projection in `full_descent_step` (since it operates on the raw Euclidean gradient).
- Memory cost: 2× the size of `ema_U` (one fast, one slow). This is acceptable.

**Deliverable:** Complete `streaming_paca_mask()` method, updated `__init__` buffers, and show exactly where it plugs into `full_descent_step`.

---

## Output Format

Structure your response as three clearly separated sections, one per component. For each:

1. **Theory** (2-3 sentences): The mathematical justification and any relevant citations.
2. **Code** (complete, copy-pasteable Python): The full class modifications, new methods, and updated `full_descent_step` integration.
3. **Integration Notes**: Exactly which lines of the existing `full_descent_step` pipeline need to change and in what order.

After all three components, provide a **Unified `full_descent_step`** that shows the complete pipeline with all three components integrated in the mathematically correct order:

```
Raw Gradient → Streaming PaCA Mask → DEAL Heat Kernel → EMA Update → Stiefel Tangent Projection → Streaming EAR (Continuous Null-Space) → QR Retraction + Core Counter-Rotation → Lazy SVC
```

Finally, address any **interaction effects** between the three components. For example: does the Liquid Core's multi-core counter-rotation interact with the Streaming EAR's covariance update? Does the Dual-EMA mask need to be computed separately for each core's gradient contribution?
