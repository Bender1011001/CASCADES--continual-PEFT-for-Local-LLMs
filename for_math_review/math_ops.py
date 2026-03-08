"""
CASCADES core mathematical operations — standalone, testable, GPU/CPU agnostic.

Each function here corresponds to a specific claim in the paper and has a
matching unit test in tests/test_math.py.
"""

import math
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# v10: GQA-Aware Metric Preconditioning (§7 — resolves 8B GQA Paradox)
# ---------------------------------------------------------------------------

def gqa_precondition_gradient(
    grad: torch.Tensor,
    gqa_ratio: float = 1.0,
) -> torch.Tensor:
    """Precondition K/V gradients to counteract GQA fan-out inflation.

    In GQA, K/V heads service multiple Q heads. During backprop, the resulting
    gradient norm for K/V is inflated by the broadcast ratio γ = H_q / H_kv.
    Projecting this asymmetric energy into an isotropic Stiefel tangent space
    warps the tangent vector, pushing QR retraction off the optimal geodesic.

    Fix: scale by 1/√γ before GORP EMA and tangent projection to restore
    geometric isotropy.

    Args:
        grad:      Raw Euclidean gradient for a K or V projection.
        gqa_ratio: H_q / H_kv (e.g. 8 for 1:8 GQA). Set to 1.0 for MHA.

    Returns:
        Preconditioned gradient with restored isotropy.
    """
    if gqa_ratio > 1.0:
        return grad / math.sqrt(gqa_ratio)
    return grad


# ---------------------------------------------------------------------------
# v10: Tikhonov-Regularized Soft-EAR (§3.3 — smooth gradient reassignment)
# ---------------------------------------------------------------------------

def soft_ear(
    grad: torch.Tensor,
    free_component: torch.Tensor,
    gamma: float = 1e-4,
) -> torch.Tensor:
    """Tikhonov-regularized smooth Energy-Accounted Reassignment.

    Replaces the hard cutoff `if free_norm < 1e-2 * grad_norm` with a smooth
    geometric bridging function:

        g_EAR = (‖g‖ / √(‖g_⊥‖² + γ²)) · g_⊥

    As ‖g_⊥‖ → 0, the multiplier smoothly saturates at ‖g‖/γ instead of
    causing either runaway amplification (hard EAR) or a discontinuous jump
    (cutoff EAR). γ should be tied to the quantization noise floor ε_quant.

    Args:
        grad:           Original gradient (for norm reference only).
        free_component: g_⊥ = (I - P) g, already projected away from occupied space.
        gamma:          Tikhonov damping factor (default: 1e-4, ≈ 0.1 × ε_quant).

    Returns:
        Smoothly-scaled free component with bounded multiplier.
    """
    free_norm = free_component.norm()
    grad_norm = grad.norm()
    multiplier = grad_norm / torch.sqrt(free_norm ** 2 + gamma ** 2)
    return multiplier * free_component


# ---------------------------------------------------------------------------
# A. Stiefel manifold: Riemannian gradient + QR retraction (StelLA, §3.1)
# ---------------------------------------------------------------------------

def riemannian_gradient(param: torch.Tensor, euclidean_grad: torch.Tensor) -> torch.Tensor:
    """Compute the Riemannian gradient on the Stiefel manifold.

    Given a point U ∈ St(n, r) and its Euclidean gradient G, the Riemannian
    gradient is:
        G_R = G - U · sym(U^T G),   sym(A) = (A + A^T) / 2

    This ensures the update stays tangent to the manifold.

    Args:
        param: Current point U ∈ ℝ^{n×r} with orthonormal columns.
        euclidean_grad: Euclidean gradient G ∈ ℝ^{n×r}.

    Returns:
        Riemannian gradient G_R ∈ ℝ^{n×r}.
    """
    sym = (param.T @ euclidean_grad + euclidean_grad.T @ param) / 2.0
    return euclidean_grad - param @ sym


def qr_retraction(point: torch.Tensor) -> torch.Tensor:
    """Retract a point back onto the Stiefel manifold via QR decomposition.

    Cost: O(n r²) — much cheaper than full SVD at O(n³).

    Args:
        point: Matrix ∈ ℝ^{n×r} (not necessarily orthonormal after a gradient step).

    Returns:
        Q factor ∈ St(n, r) — orthonormal columns.
    """
    Q, _ = torch.linalg.qr(point)
    return Q


def stella_riemannian_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    lr: float = 0.01,
) -> torch.Tensor:
    """Single Riemannian gradient step with QR retraction (in-place).

    Combines riemannian_gradient and qr_retraction into one update:
        U ← qr(U - lr · G_R)

    Args:
        param: Parameter tensor U ∈ St(n, r) — modified **in-place**.
        grad:  Euclidean gradient.
        lr:    Learning rate.

    Returns:
        Updated param (same tensor, for convenience).
    """
    with torch.no_grad():
        g_r = riemannian_gradient(param, grad)
        updated = param - lr * g_r
        param.copy_(qr_retraction(updated))
    return param


def is_orthonormal(matrix: torch.Tensor, tol: float = 1e-5) -> bool:
    """Check whether columns of *matrix* form an orthonormal basis.

    Verifies U^T U ≈ I_r.

    Args:
        matrix: ℝ^{n×r} matrix.
        tol:    Tolerance for Frobenius-norm deviation from identity.

    Returns:
        True if columns are orthonormal within *tol*.
    """
    r = matrix.shape[1]
    gram = matrix.T @ matrix
    identity = torch.eye(r, dtype=matrix.dtype, device=matrix.device)
    return (gram - identity).norm().item() < tol


# ---------------------------------------------------------------------------
# B. Energy-Accounted Reassignment — EAR (§3.3)
# ---------------------------------------------------------------------------

def energy_accounted_reassignment(
    grad: torch.Tensor,
    occupied_basis: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Project gradient away from occupied subspace, rescale to preserve norm.

    Mathematical guarantee (Proposition 1 in paper):
        ‖g_EAR‖₂ = ‖g‖₂   (norm preserved exactly when g_free ≠ 0)

    Derivation:
        g_free  = (I - P) g,          where P = Q Q^T, Q = qr(occupied_basis)
        g_EAR   = (‖g‖₂ / ‖g_free‖₂) · g_free

    Note: if g_free ≈ 0 (gradient entirely in occupied subspace), returns g_free
    unchanged (zero vector) — caller should handle this edge case.

    Args:
        grad:           Gradient vector/matrix g.
        occupied_basis: Basis matrix B ∈ ℝ^{d×k} spanning occupied directions.
        eps:            Numerical stability floor.

    Returns:
        g_EAR with ‖g_EAR‖₂ = ‖g‖₂ (or 0 if g was entirely in occupied space).
    """
    Q, _ = torch.linalg.qr(occupied_basis)
    occupied_component = Q @ (Q.T @ grad)
    g_free = grad - occupied_component

    grad_norm = grad.norm()
    free_norm = g_free.norm()

    if grad_norm > eps and free_norm > eps:
        # Relative guard: if the free component is < 1% of the gradient norm,
        # the gradient is effectively in the occupied subspace.  Amplifying it
        # would inject floating-point noise scaled up by ~(‖g‖/ε_machine).
        # In this regime we return g_free unchanged (near-zero vector).
        if free_norm < 1e-2 * grad_norm:
            return g_free
        return (grad_norm / (free_norm + eps)) * g_free

    return g_free


# ---------------------------------------------------------------------------
# C. Quantization-aware DEAL heat-kernel filter (§3.4)
# ---------------------------------------------------------------------------

def deal_heat_kernel_filter(
    grad: torch.Tensor,
    quant_noise_std: float = 0.0,
    t: float = 0.05,
    lambda_decay: float = 0.01,
) -> torch.Tensor:
    """Low-pass heat-kernel filter with quantization-aware noise floor.

    Two-stage filtering:
        Stage 1 — noise-floor gate: if ‖g‖ < ε_quant → return zeros
        Stage 2 — smooth decay:     g ← g · exp(-λ t)

    The threshold ε_quant is derived from the NF4 quantization variance:
        Var(quant error) ≈ Δ²/12  →  std ≈ Δ/√12
    We use a conservative fraction (0.01/√12) to avoid killing true signal.

    Args:
        grad:            Gradient tensor.
        quant_noise_std: Estimated std of quantized weights (proxy for Δ).
        t:               Heat-kernel time parameter.
        lambda_decay:    Decay rate.

    Returns:
        Filtered gradient.
    """
    if grad.dim() < 2:
        return grad

    eps_quant = max(1e-4, quant_noise_std * (0.01 / math.sqrt(12)))

    if grad.norm().item() < eps_quant:
        return torch.zeros_like(grad)

    decay = math.exp(-lambda_decay * t)
    return grad * decay


# ---------------------------------------------------------------------------
# D. SVC — Singular Value Calibration (§3.1, task-core regularisation)
# ---------------------------------------------------------------------------

def svc_calibration(
    lambda_matrix: torch.Tensor,
    svc_lambda: float = 0.01,
) -> torch.Tensor:
    """Shrink singular values of the task-core matrix to prevent accumulation.

    Update rule:
        S_i ← S_i / (1 + λ · S_i)

    This is a soft-thresholding analogue that pushes large singular values
    toward 1/λ, preventing any single spectral direction from dominating.

    Args:
        lambda_matrix: Task core matrix Λ_t ∈ ℝ^{r×r}.
        svc_lambda:    Regularisation strength λ.

    Returns:
        Calibrated matrix with same shape.
    """
    U, S, Vh = torch.linalg.svd(lambda_matrix, full_matrices=False)
    S_cal = S / (1.0 + svc_lambda * S)
    return U @ torch.diag(S_cal) @ Vh


# ---------------------------------------------------------------------------
# E. PaCA causal mask (§3.2 / CaLoRA)
# ---------------------------------------------------------------------------

def paca_causal_mask(
    grad: torch.Tensor,
    historical_grads: list[torch.Tensor],
    temperature: float = 0.1,
) -> torch.Tensor:
    """Compute element-wise protection mask for the current gradient.

    Components highly correlated with historical task gradients are masked
    toward 0 (protected); uncorrelated components are left at 1 (free to update).

    Args:
        grad:             Current gradient (flattened or 2-D).
        historical_grads: List of flattened gradients from prior tasks.
        temperature:      Sigmoid sharpness.

    Returns:
        Mask tensor ∈ (0, 1) with same shape as *grad*.
    """
    if not historical_grads:
        return torch.ones_like(grad)

    flat = grad.flatten()
    corr = torch.stack([
        F.cosine_similarity(flat, hg.flatten(), dim=0)
        for hg in historical_grads
    ]).mean()

    inv_importance = 1.0 - torch.sigmoid(
        (grad.abs() / (grad.abs().mean() + 1e-8)) * (1.0 + corr) / temperature
    )
    return inv_importance
