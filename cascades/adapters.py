"""
CASCADES v9 core adapter modules — extracted from cascades_exp/hf_cascades_reasoning.py.

This module contains the complete CASCADES v9 Pro architecture:
  - FunLoRA_Activation: Custom autograd activation with analytical derivatives
  - FunLoRA_Adapter: Zero-VRAM rank-1 adapter for non-critical layers
  - ResonantCore: Hebbian routed multi-core pool (v8 architecture)
  - CASCADESAdapter: Full boundary-less adapter with Stiefel manifold, EAR, PaCA, SVC
  - CASCADESLinear: Wrapper selecting critical (full adapter) vs non-critical (FunLoRA)

All classes accept an AblationConfig to control which components are active,
replacing the old global ENABLE_* flags pattern.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cascades.config import AblationConfig, DEFAULT_CONFIG
from cascades.math_ops import deal_heat_kernel_filter as _deal_filter_lib


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
        return x + sig_x + tanh_x

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


def _cllora_reassign(grad, null_sketch, config: AblationConfig):
    """CL-LoRA EAR gradient reassignment if enabled."""
    if not config.enable_cllora_reassign or null_sketch is None:
        return grad
    Q, _ = torch.linalg.qr(null_sketch)
    occupied = Q @ (Q.T @ grad)
    free = grad - occupied

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
        self.beta_fast = 0.99     # ~100 step local trajectory horizon
        self.beta_slow = 0.9998   # ~5000 step structural consensus horizon
        self.tau_conflict = -0.1  # Conflict threshold

        self.register_buffer('ema_fast_U', torch.zeros(out_features, rank))
        self.register_buffer('ema_slow_U', torch.zeros(out_features, rank))
        self.register_buffer('ema_fast_V', torch.zeros(rank, in_features))
        self.register_buffer('ema_slow_V', torch.zeros(rank, in_features))

        # Streaming frequent directions (continuous EAR)
        self.beta_ear = 0.99  # ~100 step memory half-life
        self.register_buffer('streaming_sketch_U', torch.zeros(out_features, rank))
        self.register_buffer(
            'Q_null_U', torch.zeros(out_features, max(1, rank // 2))
        )
        self.ear_initialized = False

        # Quantization noise estimate (updated during training)
        self.quant_noise_std = torch.tensor(1e-3)
        self.step_counter = 0
        self.contracted_this_step = False  # Signal for optimizer hook

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.liquid_core(
            x, self.V_shared, self.U_shared, getattr(self, 'gate_proj', None)
        )

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

    def streaming_ear_update(self, tangent_U: torch.Tensor) -> None:
        """Oja's rule covariance digest of the instantaneous Riemannian tangent."""
        with torch.no_grad():
            self.streaming_sketch_U.lerp_(tangent_U, 1 - self.beta_ear)

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

            # 5. Map to Stiefel tangent space FIRST (crucial for commutativity)
            sym_U = 0.5 * (
                self.U_shared.T @ self.ema_U + self.ema_U.T @ self.U_shared
            )
            tangent_U = self.ema_U - self.U_shared @ sym_U

            sym_V = 0.5 * (
                self.V_shared @ self.ema_V.T + self.ema_V @ self.V_shared.T
            )
            tangent_V = self.ema_V - sym_V @ self.V_shared

            # 6. Streaming EAR update and constraint
            if cfg.enable_coso_nullspace:
                self.streaming_ear_update(tangent_U)

                if self.ear_initialized:
                    free_U = _cllora_reassign(tangent_U, self.Q_null_U, cfg)

                    n_orig, n_free = tangent_U.norm(), free_U.norm()
                    n_occ = (tangent_U - free_U).norm()

                    # Autopoiesis: dynamic rank expansion
                    if (
                        n_free / (n_occ + n_free + 1e-8) < 0.10
                        and self.U_shared.shape[1] < 32
                    ):
                        with torch.no_grad():
                            print(
                                f"🧬 Autopoiesis Triggered! Expanding Rank "
                                f"{self.U_shared.shape[1]} -> "
                                f"{self.U_shared.shape[1] + 1}"
                            )

                            # Extract strongest un-represented gradient direction
                            u_new = grad_U.mean(dim=1, keepdim=True)
                            u_new = u_new - self.U_shared @ (
                                self.U_shared.T @ u_new
                            )  # Gram-Schmidt
                            u_new = u_new / (u_new.norm() + 1e-8)

                            v_new = grad_V.mean(dim=0, keepdim=True)
                            v_new = v_new - (v_new @ self.V_shared.T) @ self.V_shared
                            v_new = v_new / (v_new.norm() + 1e-8)

                            # Physically expand Stiefel bases
                            self.U_shared = nn.Parameter(
                                torch.cat([self.U_shared, u_new], dim=1)
                            )
                            self.V_shared = nn.Parameter(
                                torch.cat([self.V_shared, v_new], dim=0)
                            )

                            # Zero-pad cores (maintains isometric immersion)
                            self.liquid_core.core_pool.data = F.pad(
                                self.liquid_core.core_pool.data,
                                (0, 1, 0, 1),
                                "constant",
                                0.0,
                            )

                            # Zero-pad all covariant tracking buffers
                            self.ema_U = F.pad(self.ema_U, (0, 1))
                            self.streaming_sketch_U = F.pad(
                                self.streaming_sketch_U, (0, 1)
                            )
                            self.ema_V = F.pad(self.ema_V, (0, 0, 0, 1))

                            if cfg.enable_paca:
                                self.ema_fast_U = F.pad(self.ema_fast_U, (0, 1))
                                self.ema_slow_U = F.pad(self.ema_slow_U, (0, 1))
                                self.ema_fast_V = F.pad(
                                    self.ema_fast_V, (0, 0, 0, 1)
                                )
                                self.ema_slow_V = F.pad(
                                    self.ema_slow_V, (0, 0, 0, 1)
                                )

                            # Expand tangents for QR retraction
                            tangent_U = torch.cat(
                                [free_U, torch.zeros_like(u_new)], dim=1
                            )
                            tangent_V = torch.cat(
                                [tangent_V, torch.zeros_like(v_new)], dim=0
                            )

                            n_orig, n_free = tangent_U.norm(), tangent_U.norm()
                            free_U = tangent_U

                    if n_orig > 1e-8 and n_free > 1e-8:
                        tangent_U = free_U * min(n_orig / n_free, 5.0)

                    # Re-project to correct numerical drift from EAR
                    sym_free_U = 0.5 * (
                        self.U_shared.T @ tangent_U + tangent_U.T @ self.U_shared
                    )
                    tangent_U = tangent_U - self.U_shared @ sym_free_U

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

            # Contravariant cores: mix by R
            for k in range(self.liquid_core.num_cores):
                self.liquid_core.core_pool[k].copy_(
                    R_U @ self.liquid_core.core_pool[k] @ R_V.T
                )

            # Covariant buffers: mix by R^T (NOT R^{-1} — the transpose parity fix)
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

    def promote(self, rank: int = 8) -> bool:
        """Promote FunLoRA → full ResonantCore adapter."""
        if self.is_critical:
            return False

        self.is_critical = True
        new_adapter = CASCADESAdapter(
            self.in_features, self.out_features, rank=rank, config=self.config
        )
        new_adapter = new_adapter.to(self.adapter.a.device)
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
