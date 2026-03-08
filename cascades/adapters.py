"""
CASCADES v10 core adapter modules.

This module contains the complete CASCADES v10 "Elastic Riemannian Ecosystem":
  - FunLoRA_Activation: Custom autograd activation with analytical derivatives
  - FunLoRA_Adapter: Zero-VRAM rank-1 adapter for non-critical layers
  - ResonantCore: Hebbian routed multi-core pool (v8 architecture)
  - CASCADESAdapter: Full boundary-less adapter with Stiefel manifold, EAR, PaCA, SVC
  - CASCADESLinear: Wrapper selecting critical (full adapter) vs non-critical (FunLoRA)

v10 advancements over v9:
  - GQA-aware metric preconditioning (resolves 8B GQA paradox)
  - Tikhonov-regularized soft-EAR (smooth gradient reassignment)
  - Principal tangent expansion (noise-free rank revival via EAR sketch)
  - Subspace-contrastive decoding (adapter-level CFG for inference)

All classes accept an AblationConfig to control which components are active.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cascades.config import AblationConfig, DEFAULT_CONFIG
from cascades.math_ops import deal_heat_kernel_filter as _deal_filter_lib
from cascades.math_ops import gqa_precondition_gradient
from cascades.math_ops import soft_ear


# ---------------------------------------------------------------------------
# A. FunLoRA — Rank-1 adapter for non-critical layers (D-MoLE bottom tier)
# ---------------------------------------------------------------------------

class FunLoRA_Activation(torch.autograd.Function):
    """Fused analytical activation: f(x) = x + sigmoid(x) + tanh(x).

    Avoids caching large intermediate tensors by computing derivatives analytically:
        f'(x) = 1 + σ(x)(1 - σ(x)) + (1 - tanh²(x))
    """

    @staticmethod
    def forward(ctx, x):
        sig_x = torch.sigmoid(x)
        tanh_x = torch.tanh(x)
        ctx.save_for_backward(sig_x, tanh_x)
        # v10.4 Fix: Zero-mean at x=0 prevents catastrophic covariate shift
        # during D-MoLE promotion. σ(0)=0.5, so we subtract 0.5 to center.
        return x + sig_x + tanh_x - 0.5

    @staticmethod
    def backward(ctx, grad_output):
        sig_x, tanh_x = ctx.saved_tensors
        grad_x = 1.0 + sig_x * (1.0 - sig_x) + (1.0 - tanh_x.square())
        return grad_output * grad_x


class FunLoRA_Adapter(nn.Module):
    """Zero-VRAM FunLoRA adapter: rank-1 outer product with fused analytical activation.

    Architecture: x → (x @ b^T) → FunLoRA_Activation → (· @ a^T)
    Total params: d_out + d_in (two vectors).

    VRAM fix: Casts tiny adapter weights (not massive activations) to match input dtype.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.a = nn.Parameter(torch.randn(out_features, 1) * 0.05)
        self.b = nn.Parameter(torch.randn(1, in_features) * 0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck = x @ self.b.to(x.dtype).T
        expanded = FunLoRA_Activation.apply(bottleneck)
        return expanded @ self.a.to(x.dtype).T


# ---------------------------------------------------------------------------
# B. Resonant Core — Hebbian multi-core routing (v8 architecture)
# ---------------------------------------------------------------------------

class ResonantCore(nn.Module):
    """Multi-core pool with zero-parameter Hebbian routing.

    Instead of a learned router (which would suffer meta-forgetting),
    uses cosine similarity against slowly-updating key vectors.
    Keys are updated via exponential Hebbian averaging, detached from autograd.

    Args:
        in_features: Input dimension (for key vectors).
        rank: Dimension of each core matrix (rank × rank).
        num_cores: Number of expert cores in the pool.
    """

    def __init__(self, in_features: int, rank: int = 8, num_cores: int = 4):
        super().__init__()
        self.num_cores = num_cores

        cores = [torch.empty(rank, rank) for _ in range(num_cores)]
        for c in cores:
            nn.init.orthogonal_(c)
        self.core_pool = nn.Parameter(torch.stack(cores))

        # Zero-parameter router: no Adam, no meta-forgetting
        self.register_buffer('core_keys', torch.randn(num_cores, in_features))
        self.router_beta = 0.999  # Slow Hebbian trace

    def forward(
        self,
        x: torch.Tensor,
        V_shared: torch.Tensor,
        U_shared: torch.Tensor,
        gate_proj: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        x_float = x.to(torch.float32)
        centroid = 0.7 * x_float[:, -1, :] + 0.3 * x_float.mean(dim=1)

        # 1. Resonant routing (zero-grad cosine similarity)
        sim = F.cosine_similarity(
            centroid.unsqueeze(1), self.core_keys.unsqueeze(0), dim=-1
        )
        route_weights = F.softmax(sim / 0.05, dim=-1)  # (B, K)

        # 2. Continuous Hebbian key update (detached from autograd)
        if self.training:
            with torch.no_grad():
                for k in range(self.num_cores):
                    weight_k = route_weights[:, k].unsqueeze(1)  # (B, 1)
                    if weight_k.sum() > 0.1:
                        weighted_centroid = (centroid * weight_k).sum(dim=0) / (
                            weight_k.sum() + 1e-8
                        )
                        self.core_keys[k].lerp_(weighted_centroid, 1 - self.router_beta)
                        self.core_keys[k].copy_(
                            F.normalize(self.core_keys[k], p=2, dim=0)
                        )

        # 3. Dynamic core assembly
        Lam_active = torch.einsum(
            'bk,kij->bij', route_weights.to(x.dtype), self.core_pool.to(x.dtype)
        )

        x_V = x @ V_shared.T.to(x.dtype)
        x_V_Lam = torch.bmm(x_V, Lam_active.transpose(1, 2))
        adapt_out = x_V_Lam @ U_shared.T.to(x.dtype)

        # Optional GainLoRA gate
        if gate_proj is not None:
            gate_input = torch.cat([
                U_shared.mean(dim=0).detach(),
                V_shared.mean(dim=1).detach(),
            ], dim=0)
            adapt_out = (
                torch.sigmoid(gate_proj(gate_input).unsqueeze(0).unsqueeze(0))
                * adapt_out
            )

        return adapt_out


# ---------------------------------------------------------------------------
# C. CASCADES v6 Adapter — full boundary-less adapter with Stiefel manifold
# ---------------------------------------------------------------------------

def _deal_filter(grad, quant_noise_std, config: AblationConfig):
    """Apply DEAL heat kernel filter if enabled."""
    if not config.enable_deal or grad.dim() < 2:
        return grad
    return _deal_filter_lib(grad, quant_noise_std=quant_noise_std)


def _cllora_reassign(grad, null_sketch, config: AblationConfig, frozen_basis=None):
    """CL-LoRA EAR gradient reassignment with strict frozen protection.

    v10.3 BWT Fix: Separated two-phase lockdown.
    Phase 1 — STRICT: Hard zero-tolerance projection for frozen (prior-task)
    directions. No gradient energy is allowed to leak into past tasks.
    Phase 2 — SOFT: Tikhonov-regularized EAR penalty for active sketch
    (current-task exploration dampening).

    Uses SVD to extract valid bases (handles linear dependencies gracefully).
    """
    # 1. STRICT lockdown for past tasks (Absolute Zero Tolerance)
    if frozen_basis is not None and frozen_basis.shape[1] > 0:
        U_f, S_f, _ = torch.linalg.svd(frozen_basis, full_matrices=False)
        Q_f = U_f[:, S_f > 1e-6]
        if Q_f.shape[1] > 0:
            grad = grad - Q_f @ (Q_f.T @ grad)

    # 2. SOFT penalty for active sketch (Exploration penalty)
    if not config.enable_cllora_reassign or null_sketch is None:
        return grad

    U_n, S_n, _ = torch.linalg.svd(null_sketch, full_matrices=False)
    Q_n = U_n[:, S_n > 1e-6]

    if Q_n.shape[1] == 0:
        return grad

    occupied = Q_n @ (Q_n.T @ grad)
    free = grad - occupied

    if config.enable_soft_ear:
        return soft_ear(grad, free, gamma=config.ear_gamma)

    # Legacy hard-cutoff EAR (v9 fallback)
    grad_energy = grad.norm()
    free_energy = free.norm()
    if grad_energy > 1e-8 and free_energy > 1e-8:
        return (grad_energy / (free_energy + 1e-8)) * free
    return free


class CASCADESAdapter(nn.Module):
    """Full CASCADES v9 boundary-less adapter for critical layers.

    Implements the complete 5-pillar architecture:
      1. Shared Stiefel subspace (U_shared, V_shared) with QR retraction
      2. Streaming dual-EMA PaCA causal masking
      3. Energy-accounted gradient reassignment (EAR) via streaming sketch
      4. Autopoietic rank expansion when EAR subspace is saturated
      5. Riemannian freeze for unrouted experts

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        rank: Initial Stiefel manifold rank.
        svc_lambda: Singular value calibration strength.
        config: Ablation configuration controlling which components are active.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        svc_lambda: float = 0.01,
        config: AblationConfig = DEFAULT_CONFIG,
    ):
        super().__init__()
        self.r = rank
        self.svc_lambda = svc_lambda
        self.in_features = in_features
        self.out_features = out_features
        self.config = config

        # Pillar 1: Shared Stiefel subspace (StelLA USV^T decomposition)
        self.U_shared = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.V_shared = nn.Parameter(torch.randn(rank, in_features) * 0.01)

        # Resonant core pool (v8 Hebbian routing)
        self.liquid_core = ResonantCore(in_features, rank=rank)

        # GORP EMA gradient tracking
        self.register_buffer('ema_U', torch.zeros(out_features, rank))
        self.register_buffer('ema_V', torch.zeros(rank, in_features))
        self.beta1 = 0.9

        # GainLoRA learned interference gate
        if config.enable_gainlora_gate:
            self.gate_proj = nn.Linear(rank * 2, 1, bias=True)
            nn.init.constant_(self.gate_proj.bias, 1.0)  # start open

        # Streaming dual-EMA PaCA (temporal causal masking)
        # PaCA EMA trackers and causal event horizons (slower for v10.2 BWT)
        self.beta_fast = getattr(config, 'beta_fast', 0.95)
        self.beta_slow = getattr(config, 'beta_slow', 0.999)
        self.tau_conflict = getattr(config, 'tau_conflict', -0.1)  # Conflict threshold
        
        self.register_buffer('ema_fast_U', torch.zeros(out_features, rank))
        self.register_buffer('ema_slow_U', torch.zeros(out_features, rank))
        self.register_buffer('ema_fast_V', torch.zeros(rank, in_features))
        self.register_buffer('ema_slow_V', torch.zeros(rank, in_features))

        # Streaming frequent directions (continuous EAR)
        self.beta_ear = 0.999  # ~1000 step memory half-life (v10: was 0.99)
        self.register_buffer('streaming_sketch_U', torch.zeros(out_features, rank))
        self.register_buffer(
            'Q_null_U', torch.zeros(out_features, max(1, rank // 2))  # Fix 4: half-rank preserves plasticity
        )
        self.ear_initialized = False

        # v10 BWT fix: Accumulated frozen null-space from completed tasks.
        # At each task boundary, the occupied subspace is snapshotted and
        # concatenated here. During training, gradients are projected out
        # of BOTH the streaming null-space AND this frozen basis.
        self.register_buffer('frozen_null_basis', torch.zeros(out_features, 0))

        # v10.4: V subspace tracking — protects input-side directions.
        # Without this, V freely rotates to accommodate new tasks, altering
        # how past task inputs project into the cores.
        self.register_buffer('streaming_sketch_V', torch.zeros(in_features, rank))
        self.register_buffer(
            'Q_null_V', torch.zeros(in_features, max(1, rank // 2))
        )
        self.register_buffer('frozen_null_basis_V', torch.zeros(in_features, 0))

        self.num_frozen_tasks = 0

        # Quantization noise estimate (updated during training)
        self.quant_noise_std = torch.tensor(1e-3)
        self.step_counter = 0
        self.contracted_this_step = False  # Signal for optimizer hook

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.liquid_core(
            x, self.V_shared, self.U_shared, getattr(self, 'gate_proj', None)
        )

    def cfg_boost(self, x: torch.Tensor, lambda_cfg: float = 1.5) -> torch.Tensor:
        """v10: Subspace-Contrastive Decoding (adapter-level CFG).

        During inference, scale the adapter contribution by λ_cfg > 1.0 to
        amplify the reasoning patterns stored in the Stiefel manifold,
        overpowering the base model's conversational prior ("Certainly!").

        This is conceptually Classifier-Free Guidance at the adapter level:
            logits_final = logits_base + λ_cfg × adapter_output

        Args:
            x:          Input hidden states.
            lambda_cfg: CFG strength. 1.0 = neutral, 1.5-2.0 = strong steering.

        Returns:
            Boosted adapter output.
        """
        return lambda_cfg * self.forward(x)

    def streaming_paca_mask(
        self, grad_U: torch.Tensor, grad_V: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate coordinate-wise causal mask based on temporal gradient conflict."""
        with torch.no_grad():
            # 1. Temporal dual-updates
            self.ema_fast_U.lerp_(grad_U, 1 - self.beta_fast)
            self.ema_slow_U.lerp_(grad_U, 1 - self.beta_slow)
            self.ema_fast_V.lerp_(grad_V, 1 - self.beta_fast)
            self.ema_slow_V.lerp_(grad_V, 1 - self.beta_slow)

            # 2. Element-wise temporal cosine similarity
            sim_U = F.cosine_similarity(
                grad_U, self.ema_slow_U, dim=1, eps=1e-8
            ).unsqueeze(1)
            sim_V = F.cosine_similarity(
                grad_V, self.ema_slow_V, dim=0, eps=1e-8
            ).unsqueeze(0)

            # 3. Soft mask activation
            mask_U = torch.sigmoid((sim_U - self.tau_conflict) * 15.0)
            mask_V = torch.sigmoid((sim_V - self.tau_conflict) * 15.0)

            return mask_U, mask_V

    def streaming_ear_update(self, tangent_U: torch.Tensor, tangent_V: torch.Tensor | None = None) -> None:
        """Oja's rule covariance digest of the instantaneous Riemannian tangent."""
        with torch.no_grad():
            self.streaming_sketch_U.lerp_(tangent_U, 1 - self.beta_ear)
            # v10.4: Track V tangent for input-side subspace protection
            if tangent_V is not None and hasattr(self, 'streaming_sketch_V'):
                self.streaming_sketch_V.lerp_(tangent_V.T, 1 - self.beta_ear)

    def freeze_current_subspace(self) -> None:
        """Snapshot the current occupied subspace and freeze it for future tasks.

        Called at task boundaries. The streaming EAR sketch captures which
        gradient directions were important during the current task. This method
        extracts those directions, orthogonalizes them, and appends them to
        the frozen_null_basis — a permanent record of "directions that must
        not be overwritten."

        The frozen basis grows by at most `rank` columns per task. To prevent
        unbounded growth, we cap it at `out_features // 2` columns and use
        SVD truncation if needed.
        """
        with torch.no_grad():
            S = self.streaming_sketch_U  # (d_out, rank)
            if S.norm() < 1e-8:
                return

            # v10.2 BWT fix: Normalized Covariance
            # Normalize sketch columns so covariance elements are strictly bounded [-1, 1].
            # This makes the eigenvalues represent pure structural variance percentage,
            # untainted by extreme gradient magnitude spikes.
            S_norm = S / (S.norm(dim=0, keepdim=True) + 1e-8)

            # Extract occupied directions from the normalized correlation matrix
            C = S_norm.T @ S_norm  # (rank, rank)
            eigvals, eigvecs = torch.linalg.eigh(C)

            # Keep directions with significant percentage of structural variance
            total_variance = eigvals.sum()
            if total_variance < 1e-6:
                return

            # Keep eigenvectors capturing >= 5% of variance
            keep_mask = eigvals / total_variance > 0.05
            if not keep_mask.any():
                # At least keep the top structural direction
                keep_mask[-1] = True

            kept_vecs = eigvecs[:, keep_mask]  # (rank, k)

            # Fix 5: Use S_norm to preserve structural variance properties.
            # The eigenvectors came from normalized S, so we must project
            # through normalized S to get geometrically consistent directions.
            new_dirs = S_norm @ kept_vecs  # (d_out, k)
            Q_new, _ = torch.linalg.qr(new_dirs)  # (d_out, k)

            # Concatenate with existing frozen basis
            if self.frozen_null_basis.shape[1] > 0:
                combined = torch.cat([self.frozen_null_basis, Q_new], dim=1)
                # Fix 5: Use SVD to merge bases — handles linear dependencies
                # gracefully unlike QR which can produce artificial columns.
                U_svd, S_svd, _ = torch.linalg.svd(combined, full_matrices=False)
                Q_combined = U_svd[:, S_svd > 1e-6]
            else:
                Q_combined = Q_new

            # Cap at out_features // 2 to prevent oversized null-space
            max_cols = self.U_shared.shape[0] // 2
            if Q_combined.shape[1] > max_cols:
                # Fix 5: FIFO drop oldest columns. SVD truncation on an
                # orthogonal matrix has all singular values ≈ 1.0, making
                # it an arbitrary rotation that drops random directions.
                Q_combined = Q_combined[:, -max_cols:]

            self.frozen_null_basis = Q_combined

            # ------ V SUBSPACE FREEZE (v10.4) ------
            if hasattr(self, 'streaming_sketch_V'):
                S_V = self.streaming_sketch_V
                if S_V.norm() > 1e-8:
                    S_norm_V = S_V / (S_V.norm(dim=0, keepdim=True) + 1e-8)
                    U_sV, S_sV, _ = torch.linalg.svd(S_norm_V, full_matrices=False)
                    tot_var_V = (S_sV ** 2).sum()
                    if tot_var_V > 1e-6:
                        keep_V = (S_sV ** 2) / tot_var_V > 0.05
                        if not keep_V.any():
                            keep_V[0] = True
                        Q_new_V = U_sV[:, keep_V]

                        if (
                            getattr(self, 'frozen_null_basis_V', None) is not None
                            and self.frozen_null_basis_V.shape[1] > 0
                        ):
                            combined_V = torch.cat(
                                [self.frozen_null_basis_V, Q_new_V], dim=1
                            )
                            V_c, Sv_c, _ = torch.linalg.svd(
                                combined_V, full_matrices=False
                            )
                            Q_comb_V = V_c[:, Sv_c > 1e-6]
                        else:
                            Q_comb_V = Q_new_V

                        max_cols_V = self.V_shared.shape[1] // 2
                        if Q_comb_V.shape[1] > max_cols_V:
                            Q_comb_V = Q_comb_V[:, :max_cols_V]
                        self.frozen_null_basis_V = Q_comb_V

            self.num_frozen_tasks += 1

    def full_descent_step(self, lr: float = 0.01) -> None:
        """Unified CASCADES v9 Ecosystem Descent Pipeline.

        This is the mathematically critical function that integrates all 5 pillars.
        Order of operations is non-commutative — do NOT rearrange:
          1. Pop gradients → 2. PaCA mask → 3. DEAL filter → 4. GORP EMA →
          5. Tangent projection → 6. EAR → 7. QR retraction → 8. Ripple fix
        """
        cfg = self.config

        with torch.no_grad():
            if self.U_shared.grad is None or self.V_shared.grad is None:
                return

            # 1. Pop raw Euclidean gradients to free autograd graph
            grad_U, self.U_shared.grad = self.U_shared.grad.clone(), None
            grad_V, self.V_shared.grad = self.V_shared.grad.clone(), None

            self.step_counter += 1

            # 1.5. v10: GQA-aware metric preconditioning
            # For K/V projections, scale gradients by 1/√(H_q/H_kv) to
            # counteract fan-out inflation before entering the manifold.
            gqa_ratio = getattr(self, 'gqa_ratio', cfg.gqa_ratio)
            if gqa_ratio > 1.0:
                grad_U = gqa_precondition_gradient(grad_U, gqa_ratio)
                grad_V = gqa_precondition_gradient(grad_V, gqa_ratio)

            # Riemannian freeze: bypass EMA decay and retraction for unrouted experts
            if grad_U.norm() < 1e-8 and grad_V.norm() < 1e-8:
                return

            # Track EMA of gradient norm (for relative expansion checking)
            if not hasattr(self, 'ema_grad_norm'):
                self.ema_grad_norm = grad_U.norm()
            else:
                self.ema_grad_norm = 0.99 * self.ema_grad_norm + 0.01 * grad_U.norm()

            # 2. Streaming PaCA (temporal causal mask)
            if cfg.enable_paca and self.step_counter > 100:
                mask_U, mask_V = self.streaming_paca_mask(grad_U, grad_V)
                grad_U *= mask_U
                grad_V *= mask_V
            elif cfg.enable_paca:
                self.ema_fast_U.lerp_(grad_U, 1 - self.beta_fast)
                self.ema_slow_U.lerp_(grad_U, 1 - self.beta_slow)
                self.ema_fast_V.lerp_(grad_V, 1 - self.beta_fast)
                self.ema_slow_V.lerp_(grad_V, 1 - self.beta_slow)

            # 3. DEAL heat kernel (quantization-aware)
            ns = (
                self.quant_noise_std.item()
                if isinstance(self.quant_noise_std, torch.Tensor)
                else self.quant_noise_std
            )
            grad_U = _deal_filter(grad_U, ns, cfg)
            grad_V = _deal_filter(grad_V, ns, cfg)

            # 4. GORP Euclidean EMA smoothing
            self.ema_U.lerp_(grad_U, 1 - self.beta1)
            self.ema_V.lerp_(grad_V, 1 - self.beta1)

            # 4.5 PHANTOM REPULSION FIX (v10.4)
            # Keep EMA buffers clean of frozen bases. Without this, the EMA
            # accumulates energy in frozen directions over hundreds of steps,
            # and each tangent map leaks a fraction through numerical error.
            if getattr(self, 'frozen_null_basis', None) is not None and self.frozen_null_basis.shape[1] > 0:
                Q_f = self.frozen_null_basis
                self.ema_U.copy_(self.ema_U - Q_f @ (Q_f.T @ self.ema_U))

            if getattr(self, 'frozen_null_basis_V', None) is not None and self.frozen_null_basis_V.shape[1] > 0:
                Q_fV = self.frozen_null_basis_V
                self.ema_V.copy_(self.ema_V - self.ema_V @ Q_fV @ Q_fV.T)

            # 5. Map safely to Stiefel tangent space
            sym_U = 0.5 * (
                self.U_shared.T @ self.ema_U + self.ema_U.T @ self.U_shared
            )
            tangent_U = self.ema_U - self.U_shared @ sym_U

            sym_V = 0.5 * (
                self.V_shared @ self.ema_V.T + self.ema_V @ self.V_shared.T
            )
            tangent_V = self.ema_V - sym_V @ self.V_shared

            # 6. Apply EAR + STRICT Frozen Protection to Tangent Vectors
            if cfg.enable_coso_nullspace:
                self.streaming_ear_update(tangent_U, tangent_V)

                if getattr(self, 'ear_initialized', False):
                    # Project Tangent U
                    safe_tangent_U = _cllora_reassign(
                        tangent_U,
                        getattr(self, 'Q_null_U', None),
                        cfg,
                        frozen_basis=getattr(self, 'frozen_null_basis', None),
                    )
                    # Project Tangent V (transpose to treat as column vectors)
                    safe_tangent_V_T = _cllora_reassign(
                        tangent_V.T,
                        getattr(self, 'Q_null_V', None),
                        cfg,
                        frozen_basis=getattr(self, 'frozen_null_basis_V', None),
                    )
                    safe_tangent_V = safe_tangent_V_T.T

                    n_orig_U = tangent_U.norm()
                    n_free_U = safe_tangent_U.norm()
                    n_occ_U = (tangent_U - safe_tangent_U).norm()
                    n_orig_V = tangent_V.norm()
                    n_free_V = safe_tangent_V.norm()

                    # Autopoiesis: dynamic rank recycling
                    n_dead = getattr(self, '_dead_ranks', 0)
                    if (
                        n_free_U / (n_occ_U + n_free_U + 1e-8) < 0.20
                        and n_dead > 0
                    ):
                        with torch.no_grad():
                            revive_idx = n_dead - 1
                            print(
                                f"🧬 Autopoiesis Triggered! Recycling dead rank "
                                f"(reviving dim {revive_idx}, effective "
                                f"{self.U_shared.shape[1] - n_dead} -> "
                                f"{self.U_shared.shape[1] - n_dead + 1})"
                            )

                            S = self.streaming_sketch_U
                            U = self.U_shared

                            if cfg.enable_principal_expansion and S.norm() > 1e-6:
                                u_new = torch.randn(
                                    U.shape[0], 1,
                                    device=U.device, dtype=U.dtype,
                                )
                                for _ in range(3):
                                    u_new = u_new - U @ (U.T @ u_new)
                                    u_new = S @ (S.T @ u_new)
                                    u_new = u_new - U @ (U.T @ u_new)
                                    u_new = u_new / (u_new.norm() + 1e-8)
                                u_new = u_new.squeeze()
                            else:
                                u_new = grad_U.mean(dim=1)
                                u_new = u_new - U @ (U.T @ u_new)
                                u_new = u_new / (u_new.norm() + 1e-8)

                            v_new = grad_V.mean(dim=0)
                            v_new = v_new - (v_new @ self.V_shared.T) @ self.V_shared
                            v_new = v_new / (v_new.norm() + 1e-8)

                            self.U_shared.data[:, revive_idx] = u_new
                            self.V_shared.data[revive_idx, :] = v_new

                            self._dead_ranks = n_dead - 1
                            self.contracted_this_step = True

                    # Rescale to preserve gradient energy
                    if n_orig_U > 1e-8 and n_free_U > 1e-8:
                        tangent_U = safe_tangent_U * min(n_orig_U / n_free_U, 5.0)
                    else:
                        tangent_U = safe_tangent_U

                    if n_orig_V > 1e-8 and n_free_V > 1e-8:
                        tangent_V = safe_tangent_V * min(n_orig_V / n_free_V, 5.0)
                    else:
                        tangent_V = safe_tangent_V

            # 6.5 CORE GRADIENT PROTECTION (v10.4)
            # Prevent Adam from writing over latent matrices in directions
            # past tasks rely on. Project core grads orthogonal to frozen
            # subspaces expressed in the core's r×r coordinate frame.
            if self.liquid_core.core_pool.grad is not None:
                grad_C = self.liquid_core.core_pool.grad  # (K, r, r)

                # Protect Output Directions (U-side)
                if getattr(self, 'frozen_null_basis', None) is not None and self.frozen_null_basis.shape[1] > 0:
                    Z_U = self.U_shared.T @ self.frozen_null_basis  # (r, n_frozen)
                    Q_ZU, R_ZU = torch.linalg.qr(Z_U)
                    Q_ZU = Q_ZU[:, R_ZU.diag().abs() > 1e-6]
                    if Q_ZU.shape[1] > 0:
                        grad_C = grad_C - torch.matmul(Q_ZU @ Q_ZU.T, grad_C)

                # Protect Input Directions (V-side)
                if getattr(self, 'frozen_null_basis_V', None) is not None and self.frozen_null_basis_V.shape[1] > 0:
                    Z_V = self.V_shared @ self.frozen_null_basis_V  # (r, n_frozen_v)
                    Q_ZV, R_ZV = torch.linalg.qr(Z_V)
                    Q_ZV = Q_ZV[:, R_ZV.diag().abs() > 1e-6]
                    if Q_ZV.shape[1] > 0:
                        grad_C = grad_C - torch.matmul(grad_C, Q_ZV @ Q_ZV.T)

                self.liquid_core.core_pool.grad = grad_C

            # GQA asymmetry: dimension-calibrated learning rates
            lr_U = lr * math.sqrt(2560.0 / self.U_shared.shape[0])
            lr_V = lr * math.sqrt(2560.0 / self.V_shared.shape[1])

            # 7. QR retraction
            Q_U, R_U = torch.linalg.qr(self.U_shared - lr_U * tangent_U)
            self.U_shared.copy_(Q_U)

            Q_V, R_V = torch.linalg.qr((self.V_shared - lr_V * tangent_V).T)
            self.V_shared.copy_(Q_V.T)

            # 8. SYSTEM-WIDE BASIS COUNTER-ROTATION (The Ripple Fix)
            R_U_inv = torch.linalg.pinv(R_U)
            R_V_inv = torch.linalg.pinv(R_V)

            # Contravariant cores: mix by R
            for k in range(self.liquid_core.num_cores):
                self.liquid_core.core_pool[k].copy_(
                    R_U @ self.liquid_core.core_pool[k] @ R_V.T
                )

            # Covariant buffers: mix by R^T
            self.ema_U.copy_(self.ema_U @ R_U.T)
            self.ema_V.copy_(R_V @ self.ema_V)

            if cfg.enable_paca:
                self.ema_fast_U.copy_(self.ema_fast_U @ R_U.T)
                self.ema_slow_U.copy_(self.ema_slow_U @ R_U.T)
                self.ema_fast_V.copy_(R_V @ self.ema_fast_V)
                self.ema_slow_V.copy_(R_V @ self.ema_slow_V)

            if cfg.enable_coso_nullspace:
                # Tangent vectors transform contravariantly via R^{-1}
                self.streaming_sketch_U.copy_(self.streaming_sketch_U @ R_U_inv)
                # v10.4: V sketch also needs contravariant transform
                if hasattr(self, 'streaming_sketch_V'):
                    self.streaming_sketch_V.copy_(self.streaming_sketch_V @ R_V_inv)


# ---------------------------------------------------------------------------
# D. CASCADES Linear Wrapper — selects critical (full) vs non-critical (FunLoRA)
# ---------------------------------------------------------------------------

class CASCADESLinear(nn.Module):
    """Wraps a base linear layer with either full CASCADES or FunLoRA adapter.

    For critical layers (high D-MoLE importance): uses full CASCADESAdapter.
    For non-critical layers: uses lightweight FunLoRA_Adapter (rank-1).

    Supports dynamic promotion (FunLoRA → full) and demotion (full → FunLoRA)
    via Dormant Core Distillation for phase-transition hardware.

    Args:
        base_layer: The original nn.Linear or Linear4bit to wrap.
        rank: Stiefel manifold rank for critical adapters.
        is_critical: Whether this layer receives a full adapter.
        config: Ablation configuration.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        rank: int = 8,
        is_critical: bool = True,
        config: AblationConfig = DEFAULT_CONFIG,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.is_critical = is_critical
        self.config = config

        if is_critical:
            self.adapter = CASCADESAdapter(
                self.in_features, self.out_features, rank=rank, config=config
            )
        else:
            self.adapter = FunLoRA_Adapter(self.in_features, self.out_features)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        base_out = (
            self.base_layer(x, *args, **kwargs)
            if not isinstance(self.base_layer, nn.Linear)
            else self.base_layer(x)
        )
        adapt_out = self.adapter(x)
        # Ensure adapter output dtype matches base output (prevents SDPA mismatch)
        return base_out + 0.1 * adapt_out.to(base_out.dtype)

    def forward_with_cfg(self, x: torch.Tensor, lambda_cfg: float = 1.5,
                         *args, **kwargs) -> torch.Tensor:
        """v10: Forward pass with adapter-level Classifier-Free Guidance.

        Multiplies the adapter contribution by λ_cfg during inference
        to steer away from the base model's conversational prior.
        Use during evaluation/generation only — not training.

        Args:
            x:          Input tensor.
            lambda_cfg: CFG boost strength (1.5 default, 1.0 = neutral).
        """
        base_out = (
            self.base_layer(x, *args, **kwargs)
            if not isinstance(self.base_layer, nn.Linear)
            else self.base_layer(x)
        )
        if self.is_critical and hasattr(self.adapter, 'cfg_boost'):
            adapt_out = self.adapter.cfg_boost(x, lambda_cfg=lambda_cfg)
        else:
            adapt_out = self.adapter(x)
        return base_out + 0.1 * adapt_out.to(base_out.dtype)

    def promote(self, rank: int = 8) -> bool:
        """Promote FunLoRA → full ResonantCore adapter.

        Fix 2: Distills the trained FunLoRA rank-1 weights (a, b) into
        the first dimension of the new Stiefel manifold, preserving
        learned knowledge instead of injecting random noise.
        """
        if self.is_critical:
            return False

        self.is_critical = True
        new_adapter = CASCADESAdapter(
            self.in_features, self.out_features, rank=rank, config=self.config
        )
        new_adapter = new_adapter.to(self.adapter.a.device)

        # v10.4: Distill FunLoRA weights + QR parity correction
        with torch.no_grad():
            a_norm = self.adapter.a.norm() + 1e-8
            b_norm = self.adapter.b.norm() + 1e-8

            # Inject FunLoRA directions into dim 0 of Stiefel bases
            new_adapter.U_shared.data[:, 0:1] = self.adapter.a / a_norm
            new_adapter.V_shared.data[0:1, :] = self.adapter.b / b_norm

            # STRICT QR orthogonalization to prevent Stiefel shattering.
            # QR may flip signs — we must track parity to fix the core.
            Q_u, R_u = torch.linalg.qr(new_adapter.U_shared.data)
            new_adapter.U_shared.data.copy_(Q_u)
            Q_v, R_v = torch.linalg.qr(new_adapter.V_shared.data.T)
            new_adapter.V_shared.data.copy_(Q_v.T)

            # QR parity correction: if QR flipped dim 0, flip the core too
            sign_u = torch.sign(R_u[0, 0]).item() or 1.0
            sign_v = torch.sign(R_v[0, 0]).item() or 1.0
            distilled_val = (a_norm * b_norm * 2.25).item() * sign_u * sign_v

            # Transfer FunLoRA magnitude into liquid core dimension 0
            for k in range(new_adapter.liquid_core.num_cores):
                new_adapter.liquid_core.core_pool.data[k] *= 0.01  # Suppress noise
                new_adapter.liquid_core.core_pool.data[k, 0, 0] = distilled_val

        self.adapter = new_adapter
        return True

    def demote(self) -> bool:
        """Demote ResonantCore → FunLoRA via Dormant Core Distillation.

        Performs exact Stiefel-lifted SVD to extract the top principal
        component before demotion, preserving learned knowledge.
        """
        if not self.is_critical:
            return False

        with torch.no_grad():
            # 1. Structural consensus: expected K-core behavior
            mean_core = self.adapter.liquid_core.core_pool.mean(dim=0)

            # 2. Latent isometry SVD
            U_c, S_c, Vh_c = torch.linalg.svd(mean_core, full_matrices=False)

            # 3. Extract top principal component
            sigma_1 = S_c[0]
            u_1 = U_c[:, 0:1]    # (r, 1) — preserves 2D for broadcasting
            vh_1 = Vh_c[0:1, :]  # (1, r)

            # 4. Push-forward to ambient coordinate space
            a_ambient = self.adapter.U_shared @ u_1  # (d_out, 1)
            b_ambient = vh_1 @ self.adapter.V_shared  # (1, d_in)

            # 5. Non-linear gain compensation (FunLoRA f'(0) = 2.25)
            scale = math.sqrt(sigma_1.item() / 2.25)

            # Fold gate state into distilled weights
            if getattr(self.adapter, 'gate_proj', None) is not None:
                gate_in = torch.cat([
                    self.adapter.U_shared.mean(0),
                    self.adapter.V_shared.mean(1),
                ])
                scale *= math.sqrt(
                    torch.sigmoid(self.adapter.gate_proj(gate_in)).item()
                )

            a_distilled = a_ambient * scale
            b_distilled = b_ambient * scale

        self.is_critical = False
        new_adapter = FunLoRA_Adapter(self.in_features, self.out_features)

        # Inject distilled memories
        with torch.no_grad():
            new_adapter.a.copy_(a_distilled)
            new_adapter.b.copy_(b_distilled)

        new_adapter = new_adapter.to(self.adapter.U_shared.device)
        self.adapter = new_adapter
        return True
