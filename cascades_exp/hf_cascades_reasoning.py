import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
import pandas as pd
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch.utils.data import DataLoader

# ============================================================================
# CASCADES REASONING PIPELINE (Think -> Plan -> Act)
# ============================================================================
# Sequential Domain Adaptation for formatting complex reasoning chains
# Task 0: Logic (GSM8k)
# Task 1: Decomp (Arch Planning)
# Task 2: Exec (Tool Use)
# ============================================================================

# --- Ablation Flags ---
ENABLE_PACA = True            # Intersection B: CaLoRA causal attribution
ENABLE_DEAL = True            # Intersection B: heat kernel filter (quant-aware)
ENABLE_GAINLORA_GATE = True   # Intersection B: GainLoRA learned interference gate
ENABLE_COSO_NULLSPACE = True  # Intersection C: CoSO Frequent Directions
ENABLE_CLLORA_REASSIGN = True # Intersection C: CL-LoRA gradient reassignment
ENABLE_SVC = True             # Intersection C: Singular Value Calibration
ENABLE_DMOLE_SELECT = True    # Intersection D: D-MoLE layer importance selection
ENABLE_FUNLORA = True         # Intersection D: FunLoRA rank-1 for non-critical layers


# --- Intersection A: StelLA-style Riemannian step (NeurIPS'25 Spotlight) ---
def stella_riemannian_step(param, grad, lr=0.01):
    """StelLA's modular Euclidean→Riemannian conversion.
    Computes the Riemannian gradient on the Stiefel manifold and retracts via QR.
    Replaces the ad-hoc QR retraction from v1/v2 with the principled StelLA formulation."""
    with torch.no_grad():
        # Riemannian gradient: grad_R = grad - param @ sym(param^T @ grad)
        sym = (param.T @ grad + grad.T @ param) / 2.0
        riemannian_grad = grad - param @ sym
        # QR retraction (O(dr²))
        updated = param - lr * riemannian_grad
        Q, _ = torch.linalg.qr(updated)
        param.copy_(Q)


# --- Intersection B: CaLoRA PaCA causal mask ---
def paca_causal_mask(grad, historical_grads, temperature=0.1):
    """CaLoRA NeurIPS'25: parameter-level counterfactual attribution.
    Identifies which gradient components causally benefit past task performance.
    High correlation to historical grads → protect (mask → 0); low → update freely (mask → 1)."""
    if not ENABLE_PACA or len(historical_grads) == 0:
        return torch.ones_like(grad)
    flat = grad.flatten()
    corr = torch.stack([
        F.cosine_similarity(flat, hg.flatten(), dim=0)
        for hg in historical_grads
    ]).mean()
    inv_importance = 1.0 - torch.sigmoid(
        (grad.abs() / (grad.abs().mean() + 1e-8)) * (1 + corr) / temperature
    )
    return inv_importance


# --- Intersection B: DEAL heat kernel (quantization-aware v3 fix) ---
def deal_heat_kernel_filter(grad, quant_noise_std=0.0, t=0.05, lambda_decay=0.01):
    """DEAL arXiv'25: heat-kernel low-pass filter on gradients.
    v4 fix: adaptive threshold ε_quant based on quantization noise floor.
    Filtering operates strictly within coordinates avoiding explicit O(d^3) SVD."""
    if not ENABLE_DEAL or grad.dim() < 2:
        return grad
    
    # Quantization-aware noise floor proxy parameterized by quantization step
    eps_quant = max(1e-4, quant_noise_std * (0.01 / math.sqrt(12))) 
    
    grad_norm = grad.norm()
    # Filter out pure noise regimes
    if grad_norm < eps_quant:
        return torch.zeros_like(grad)
        
    decay = math.exp(-lambda_decay * t)
    return grad * decay


# --- Intersection C: CL-LoRA gradient reassignment ---
def cllora_gradient_reassign(grad, null_sketch, alpha=1.0):
    """CL-LoRA CVPR'25 / CASCADES EAR: redirect blocked gradient energy into the free subspace.
    Uses exact Energy-Accounted Reassignment (EAR) to preserve ||g||_2 while staying
    in the feasible orthogonal subspace."""
    if not ENABLE_CLLORA_REASSIGN or null_sketch is None:
        return grad
    # Compute null-space projection P
    Q, _ = torch.linalg.qr(null_sketch)
    occupied_component = Q @ (Q.T @ grad)
    null_component = grad - occupied_component
    
    # EXACT EAR SCALING: g_EAR = ( ||g||_2 / (||g_free||_2 + eps) ) * g_free
    grad_energy = grad.norm()
    null_energy = null_component.norm()
    
    if grad_energy > 1e-8 and null_energy > 1e-8:
        return (grad_energy / (null_energy + 1e-8)) * null_component
        
    return null_component


class FunLoRA_Activation(torch.autograd.Function):
    """Fuses functional expansion to compute analytical derivatives, avoiding massive tensor caching."""
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

class FunLoRA_Adapter_Optimized(nn.Module):
    """Zero-VRAM FunLoRA Adapter using fused analytical activations and tiny weight casting."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.a = nn.Parameter(torch.randn(out_features, 1) * 0.05)
        self.b = nn.Parameter(torch.randn(1, in_features) * 0.05)

    def forward(self, x):
        # VRAM FIX: Cast tiny adapter weights instead of massive activation tensors
        bottleneck = x @ self.b.to(x.dtype).T
        # VRAM FIX: Fused analytical activation
        expanded = FunLoRA_Activation.apply(bottleneck)
        return expanded @ self.a.to(x.dtype).T


# --- CASCADES v8: Autopoietic Architecture ---
class CASCADES_v8_ResonantCore(nn.Module):
    def __init__(self, in_features, rank=8, num_cores=4):
        super().__init__()
        self.num_cores = num_cores
        
        cores = [torch.empty(rank, rank) for _ in range(num_cores)]
        for c in cores: nn.init.orthogonal_(c)
        self.core_pool = nn.Parameter(torch.stack(cores))
        
        # Zero-Parameter Router: No Adam, No Meta-Forgetting
        self.register_buffer('core_keys', torch.randn(num_cores, in_features))
        self.router_beta = 0.999  # Slow Hebbian trace

    def forward(self, x, V_shared, U_shared, gate_proj=None):
        x_float = x.to(torch.float32)
        centroid = 0.7 * x_float[:, -1, :] + 0.3 * x_float.mean(dim=1)
        
        # 1. Resonant Routing (Zero-grad Cosine Similarity)
        # Temperature scaling sharpens the routing decision
        sim = F.cosine_similarity(centroid.unsqueeze(1), self.core_keys.unsqueeze(0), dim=-1)
        route_weights = F.softmax(sim / 0.05, dim=-1) # (B, K)
        
        # 2. Continuous Hebbian Key Update (Detached from autograd)
        if self.training:
            with torch.no_grad():
                # Pull the keys toward the centroids that activated them
                for k in range(self.num_cores):
                    weight_k = route_weights[:, k].unsqueeze(1) # (B, 1)
                    if weight_k.sum() > 0.1: # If core was utilized
                        weighted_centroid = (centroid * weight_k).sum(dim=0) / (weight_k.sum() + 1e-8)
                        self.core_keys[k].lerp_(weighted_centroid, 1 - self.router_beta)
                        # Keep keys normalized on the unit hypersphere
                        self.core_keys[k].copy_(F.normalize(self.core_keys[k], p=2, dim=0))
        
        # 3. Dynamic Core Assembly
        Lam_active = torch.einsum('bk,kij->bij', route_weights.to(x.dtype), self.core_pool.to(x.dtype))
        
        x_V = x @ V_shared.T.to(x.dtype)
        x_V_Lam = torch.bmm(x_V, Lam_active.transpose(1, 2))
        adapt_out = x_V_Lam @ U_shared.T.to(x.dtype)
        
        # Optional GainLoRA Gate execution 
        if gate_proj is not None:
            gate_input = torch.cat([
                U_shared.mean(dim=0).detach(),
                V_shared.mean(dim=1).detach()
            ], dim=0)
            adapt_out = torch.sigmoid(gate_proj(gate_input).unsqueeze(0).unsqueeze(0)) * adapt_out
            
        return adapt_out


# --- Full CASCADES v6 Adapter (critical layers) ---
class CASCADES_v6_Adapter(nn.Module):
    """Full boundary-less v6 adapter for critical layers."""

    def __init__(self, in_features, out_features, rank=8, svc_lambda=0.01):
        super().__init__()
        self.r = rank
        self.svc_lambda = svc_lambda
        self.in_features = in_features
        self.out_features = out_features

        # Intersection A: Shared Stiefel subspace (StelLA USV^T decomposition)
        self.U_shared = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.V_shared = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        
        # CASCADES v8: Hebbian Resonant Routing
        self.liquid_core = CASCADES_v8_ResonantCore(in_features, rank=rank)

        # Intersection A: GORP EMA gradient tracking
        self.register_buffer('ema_U', torch.zeros(out_features, rank))
        self.register_buffer('ema_V', torch.zeros(rank, in_features))
        self.beta1 = 0.9

        # Intersection B: GainLoRA learned interference gate
        if ENABLE_GAINLORA_GATE:
            self.gate_proj = nn.Linear(rank * 2, 1, bias=True)
            nn.init.constant_(self.gate_proj.bias, 1.0)  # start open

        # CASCADES v6: Temporal Dual-EMA Causal Masking (Streaming PaCA)
        self.beta_fast = 0.99     # ~100 step local trajectory horizon
        self.beta_slow = 0.9998   # ~5000 step structural consensus horizon
        self.tau_conflict = -0.1  # Conflict threshold
        
        self.register_buffer('ema_fast_U', torch.zeros(out_features, rank))
        self.register_buffer('ema_slow_U', torch.zeros(out_features, rank))
        self.register_buffer('ema_fast_V', torch.zeros(rank, in_features))
        self.register_buffer('ema_slow_V', torch.zeros(rank, in_features))

        # CASCADES v6: Streaming Frequent Directions (Continuous EAR)
        self.beta_ear = 0.99  # ~100 step memory half-life
        self.register_buffer('streaming_sketch_U', torch.zeros(out_features, rank))
        self.register_buffer('Q_null_U', torch.zeros(out_features, max(1, rank // 2)))
        self.ear_initialized = False

        # Quantization noise estimate (updated during training)
        self.quant_noise_std = torch.tensor(1e-3)
        self.step_counter = 0
        self.contracted_this_step = False # CASCADES v9: Signal for optimizer hook


    def forward(self, x):
        return self.liquid_core(x, self.V_shared, self.U_shared, getattr(self, 'gate_proj', None))

    def streaming_paca_mask(self, grad_U, grad_V):
        """Generates coordinate-wise causal mask based on temporal gradient conflict."""
        with torch.no_grad():
            # 1. Temporal dual-updates
            self.ema_fast_U.lerp_(grad_U, 1 - self.beta_fast)
            self.ema_slow_U.lerp_(grad_U, 1 - self.beta_slow)
            self.ema_fast_V.lerp_(grad_V, 1 - self.beta_fast)
            self.ema_slow_V.lerp_(grad_V, 1 - self.beta_slow)
            
            # 2. Element-wise Temporal Cosine Similarity (+1 agree, -1 conflict)
            # Computed between the incoming gradient and the Slow EMA structure
            sim_U = F.cosine_similarity(grad_U, self.ema_slow_U, dim=1, eps=1e-8).unsqueeze(1)
            sim_V = F.cosine_similarity(grad_V, self.ema_slow_V, dim=0, eps=1e-8).unsqueeze(0)
            
            # 3. Soft Mask Activation
            # Clamps to ~0 if conflict < tau; saturates near ~1 otherwise
            mask_U = torch.sigmoid((sim_U - self.tau_conflict) * 15.0)  # steeper for reasoning sharpness
            mask_V = torch.sigmoid((sim_V - self.tau_conflict) * 15.0)
            
            return mask_U, mask_V

    def streaming_ear_update(self, tangent_U):
        """Oja's rule covariance digest of the instantaneous Riemannian tangent."""
        with torch.no_grad():
            self.streaming_sketch_U.lerp_(tangent_U, 1 - self.beta_ear)



    def full_descent_step(self, lr=0.01):
        """Unified CASCADES v9 Ecosystem Descent Pipeline"""
        with torch.no_grad():
            if self.U_shared.grad is None or self.V_shared.grad is None: return

            # 1. Pop raw Euclidean gradients immediately to free the autograd graph
            grad_U, self.U_shared.grad = self.U_shared.grad.clone(), None
            grad_V, self.V_shared.grad = self.V_shared.grad.clone(), None

            self.step_counter += 1

            # === CASCADES v9: MOE INTERFERENCE FIX (RIEMANNIAN FREEZE) ===
            # Detect if this expert was unrouted (zero gradient energy). 
            # If so, we bypass EMA decay and Retraction entirely to perfectly 
            # lock the geometric Stiefel tracking buffers in hibernation.
            if grad_U.norm() < 1e-8 and grad_V.norm() < 1e-8:
                return 

            # Track EMA of Gradient Norm (for relative Expansion checking)
            if not hasattr(self, 'ema_grad_norm'):
                self.ema_grad_norm = grad_U.norm()
            else:
                self.ema_grad_norm = 0.99 * self.ema_grad_norm + 0.01 * grad_U.norm()

            # 2. Component 3: Streaming PaCA (Temporal Causal Mask)
            # Apply after a small warmup to allow Slow EMA to populate
            if ENABLE_PACA and self.step_counter > 100:
                mask_U, mask_V = self.streaming_paca_mask(grad_U, grad_V)
                grad_U *= mask_U
                grad_V *= mask_V
            elif ENABLE_PACA:
                self.ema_fast_U.lerp_(grad_U, 1 - self.beta_fast)
                self.ema_slow_U.lerp_(grad_U, 1 - self.beta_slow)
                self.ema_fast_V.lerp_(grad_V, 1 - self.beta_fast)
                self.ema_slow_V.lerp_(grad_V, 1 - self.beta_slow)

            # 3. DEAL Heat Kernel (Quantization-aware)
            ns = self.quant_noise_std.item() if isinstance(self.quant_noise_std, torch.Tensor) else self.quant_noise_std
            grad_U = deal_heat_kernel_filter(grad_U, quant_noise_std=ns)
            grad_V = deal_heat_kernel_filter(grad_V, quant_noise_std=ns)

            # 4. GORP Euclidean EMA smoothing
            self.ema_U.lerp_(grad_U, 1 - self.beta1)
            self.ema_V.lerp_(grad_V, 1 - self.beta1)

            # 5. Map to Stiefel Tangent Space FIRST (Crucial for Commutativity)
            sym_U = 0.5 * (self.U_shared.T @ self.ema_U + self.ema_U.T @ self.U_shared)
            tangent_U = self.ema_U - self.U_shared @ sym_U
            
            sym_V = 0.5 * (self.V_shared @ self.ema_V.T + self.ema_V @ self.V_shared.T)
            tangent_V = self.ema_V - sym_V @ self.V_shared

            # 6. Component 2: Streaming EAR Update and Constraint
            if ENABLE_COSO_NULLSPACE:
                self.streaming_ear_update(tangent_U)
                
                # NOTE: Amortized null-space extraction is now batched globally per epoch.

                if self.ear_initialized:
                    occ_U = self.Q_null_U @ (self.Q_null_U.T @ tangent_U)
                    free_U = tangent_U - occ_U
                    
                    n_orig, n_free = tangent_U.norm(), free_U.norm()
                    
                    # === V8 AUTOPOIESIS (DYNAMIC RANK EXPANSION) ===
                    # If the null-space free energy is suffocating learning relative to historical norm
                    if n_free < 0.10 * self.ema_grad_norm and self.U_shared.shape[1] < 16: # Max rank safety cap
                        with torch.no_grad():
                            print(f"🧬 Autopoiesis Triggered! Expanding Rank {self.U_shared.shape[1]} -> {self.U_shared.shape[1] + 1}")
                            
                            # 1. Extract the strongest un-represented gradient direction in ambient space
                            u_new = grad_U.mean(dim=1, keepdim=True)
                            u_new = u_new - self.U_shared @ (self.U_shared.T @ u_new) # Gram-Schmidt
                            u_new = u_new / (u_new.norm() + 1e-8)
                            
                            v_new = grad_V.mean(dim=0, keepdim=True) 
                            v_new = v_new - (v_new @ self.V_shared.T) @ self.V_shared
                            v_new = v_new / (v_new.norm() + 1e-8)
                            
                            # 2. Physically expand the Stiefel Bases in VRAM
                            self.U_shared = nn.Parameter(torch.cat([self.U_shared, u_new], dim=1))
                            self.V_shared = nn.Parameter(torch.cat([self.V_shared, v_new], dim=0))
                            
                            # 3. Zero-pad Cores (Maintains exact isometric immersion)
                            # Pad the last two dimensions (rank, rank) with (0, 1) to add a row and column for each core
                            self.liquid_core.core_pool.data = F.pad(self.liquid_core.core_pool.data, (0, 1, 0, 1), "constant", 0.0)
                                
                            # 4. Zero-pad all Covariant Tracking Buffers
                            self.ema_U = F.pad(self.ema_U, (0, 1))
                            self.streaming_sketch_U = F.pad(self.streaming_sketch_U, (0, 1))
                            self.ema_V = F.pad(self.ema_V, (0, 0, 0, 1)) # Note transpose convention for V
                            
                            if ENABLE_PACA:
                                self.ema_fast_U = F.pad(self.ema_fast_U, (0, 1))
                                self.ema_slow_U = F.pad(self.ema_slow_U, (0, 1))
                                self.ema_fast_V = F.pad(self.ema_fast_V, (0, 0, 0, 1))
                                self.ema_slow_V = F.pad(self.ema_slow_V, (0, 0, 0, 1))
                            
                            # 5. Expand tangents so QR retraction succeeds on the new shape
                            # We use free_U for U to preserve the projected direction 
                            tangent_U = torch.cat([free_U, torch.zeros_like(u_new)], dim=1)
                            tangent_V = torch.cat([tangent_V, torch.zeros_like(v_new)], dim=0)
                            
                            # Override norms to prevent anomalous scaling after expansion
                            n_orig, n_free = tangent_U.norm(), tangent_U.norm()
                            free_U = tangent_U
                            
                    if n_orig > 1e-8 and n_free > 1e-8:
                        tangent_U = free_U * min(n_orig / n_free, 5.0)

                    # Re-project to correct numerical drift from EAR projection
                    sym_free_U = 0.5 * (self.U_shared.T @ tangent_U + tangent_U.T @ self.U_shared)
                    tangent_U = tangent_U - self.U_shared @ sym_free_U

            # 7. QR Retraction
            Q_U, R_U = torch.linalg.qr(self.U_shared - lr * tangent_U)
            self.U_shared.copy_(Q_U)
            
            Q_V, R_V = torch.linalg.qr((self.V_shared - lr * tangent_V).T)
            self.V_shared.copy_(Q_V.T)
            
            # 8. SYSTEM-WIDE BASIS COUNTER-ROTATION (The Ripple Fix)
            # Use pseudo-inverse to prevent CUDA deadlocks if R is ill-conditioned (which happens in baseline ablation)
            R_U_inv = torch.linalg.pinv(R_U)
            R_V_inv = torch.linalg.pinv(R_V)

            # RULE 2: Contravariant Cores (Mix by R)
            for k in range(self.liquid_core.num_cores):
                self.liquid_core.core_pool[k].copy_(R_U @ self.liquid_core.core_pool[k] @ R_V.T)
                
            # RULE 1: Covariant Ambient Buffers (Mix by R_inv)
            self.ema_U.copy_(self.ema_U @ R_U_inv)
            self.ema_V.copy_(R_V_inv @ self.ema_V) # Left side for transposed V space
            
            if ENABLE_PACA:
                self.ema_fast_U.copy_(self.ema_fast_U @ R_U_inv)
                self.ema_slow_U.copy_(self.ema_slow_U @ R_U_inv)
                self.ema_fast_V.copy_(R_V_inv @ self.ema_fast_V)
                self.ema_slow_V.copy_(R_V_inv @ self.ema_slow_V)
                
            if ENABLE_COSO_NULLSPACE:
                self.streaming_sketch_U.copy_(self.streaming_sketch_U @ R_U_inv)
            
            # NOTE: Autopoietic Contraction and SVC are now batched globally for efficiency.



# --- Adaptive Linear Wrapper ---
class CASCADES_v6_Linear(nn.Module):
    """Wraps base layer with either full CASCADES (critical) or FunLoRA (non-critical)."""

    def __init__(self, base_layer, rank=8, is_critical=True):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.is_critical = is_critical

        if is_critical:
            self.adapter = CASCADES_v6_Adapter(self.in_features, self.out_features, rank=rank)
        else:
            self.adapter = FunLoRA_Adapter_Optimized(self.in_features, self.out_features)

    def forward(self, x, *args, **kwargs):
        base_out = self.base_layer(x, *args, **kwargs) if not isinstance(self.base_layer, nn.Linear) else self.base_layer(x)
        adapt_out = self.adapter(x)
        # Ensure adapter output dtype matches base output to prevent SDPA mismatch
        return base_out + 0.1 * adapt_out.to(base_out.dtype)
        
    def promote(self, rank=8):
        """Phase-Transition leaps: Promotes a rank-1 FunLoRA to a full Resonant core.
           Safely handles transferring the activation weight onto the new structure."""
        if self.is_critical: return False
        
        self.is_critical = True
        new_adapter = CASCADES_v6_Adapter(self.in_features, self.out_features, rank=rank)
        new_adapter = new_adapter.to(self.adapter.a.device)
        self.adapter = new_adapter
        return True
        
    def demote(self):
        """Phase-Transition leaps: Distills the dormant Resonant core back down 
           to a rank-1 FunLoRA via exact Stiefel-lifted SVD to prevent forgetting."""
        if not self.is_critical: return False
        
        # --- CASCADES v9: Dormant Core Distillation (The Sleep Cycle) ---
        with torch.no_grad():
            # 1. Structural Consensus: Expected K-Core behavior (r x r)
            mean_core = self.adapter.liquid_core.core_pool.mean(dim=0)
            
            # 2. Latent Isometry SVD 
            U_c, S_c, Vh_c = torch.linalg.svd(mean_core, full_matrices=False)
            
            # 3. Extract Top Principal Component
            sigma_1 = S_c[0]
            u_1 = U_c[:, 0:1]    # (r, 1) preserves 2D shape for exact broadcasting
            vh_1 = Vh_c[0:1, :]  # (1, r)
            
            # 4. Push-forward to ambient coordinate space
            a_ambient = self.adapter.U_shared @ u_1  # (d_out, 1)
            b_ambient = vh_1 @ self.adapter.V_shared # (1, d_in)
            
            # 5. Non-Linear Gain Compensation (FunLoRA f'(0) = 2.25)
            scale = math.sqrt(sigma_1.item() / 2.25)
            
            # (Optional) Fold the Gate state directly into the distilled weights
            if getattr(self.adapter, 'gate_proj', None) is not None:
                gate_in = torch.cat([self.adapter.U_shared.mean(0), self.adapter.V_shared.mean(1)])
                scale *= math.sqrt(torch.sigmoid(self.adapter.gate_proj(gate_in)).item())

            a_distilled = a_ambient * scale
            b_distilled = b_ambient * scale

        self.is_critical = False
        new_adapter = FunLoRA_Adapter_Optimized(self.in_features, self.out_features)
        
        # 6. Safely inject the distilled memories
        with torch.no_grad():
            new_adapter.a.copy_(a_distilled)
            new_adapter.b.copy_(b_distilled)
        
        new_adapter = new_adapter.to(self.adapter.U_shared.device)
        self.adapter = new_adapter
        return True


# --- Intersection D: D-MoLE Layer Importance Selection ---
def compute_layer_importance(model, dataloader, device, threshold=0.15):
    """D-MoLE ICML'25: gradient-based layer importance scoring.
    For 4-bit models, uses activation-based heuristic (output variance as proxy)
    since 4-bit weights don't support requires_grad. For fp16/fp32 models, uses
    gradient norms directly."""
    if not ENABLE_DMOLE_SELECT:
        return {}  # all critical

    importance = {}
    model.eval()
    activation_stats = {}
    hooks = []

    # Register forward hooks to capture activation variance per layer
    for name, module in model.named_modules():
        if isinstance(module, CASCADES_v6_Linear) or isinstance(module, nn.Linear) or type(module).__name__ == "Linear4bit":
            activation_stats[name] = 0.0
            def make_hook(layer_name):
                def hook_fn(mod, inp, out):
                    if isinstance(out, torch.Tensor):
                        activation_stats[layer_name] += out.float().var().item()
                    elif isinstance(out, tuple) and isinstance(out[0], torch.Tensor):
                        activation_stats[layer_name] += out[0].float().var().item()
                return hook_fn
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Multi-batch for importance estimation stability
    max_batches = 3
    batches_processed = 0
    with torch.no_grad():
        for batch in dataloader:
            if batches_processed >= max_batches:
                break
            input_ids, attention_mask = batch
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            model(input_ids=input_ids, attention_mask=attention_mask)
            batches_processed += 1

    # Remove hooks
    for h in hooks:
        h.remove()

    if batches_processed > 0:
        for k in activation_stats:
            activation_stats[k] /= batches_processed

    if not activation_stats:
        return {}

    max_var = max(activation_stats.values())
    if max_var > 0:
        importance = {k: v / max_var for k, v in activation_stats.items()}
    else:
        importance = activation_stats

    # Layers above threshold are critical (high variance = more forgetting pressure)
    critical = {k: v >= threshold for k, v in importance.items()}
    n_critical = sum(1 for v in critical.values() if v)
    n_total = len(critical)
    print(f"D-MoLE: {n_critical}/{n_total} layers marked critical (threshold={threshold})")
    return critical


def inject_cascades_v3(model, rank=8, target_modules=None, layer_importance=None):
    """Inject CASCADES v3.1 adapters with D-MoLE selective allocation."""
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "up_proj", "down_proj"]

    adapters_critical = []
    adapters_funlora = []

    for name, module in dict(model.named_modules()).items():
        if any(t in name for t in target_modules) and (isinstance(module, nn.Linear) or type(module).__name__ == "Linear4bit"):
            parts = name.split(".")
            parent_name = ".".join(parts[:-1])
            child_name = parts[-1]
            try:
                parent = model.get_submodule(parent_name)

                # D-MoLE: determine if this layer is critical
                is_critical = True
                if ENABLE_DMOLE_SELECT and ENABLE_FUNLORA and layer_importance:
                    is_critical = layer_importance.get(name, True)

                new_module = CASCADES_v6_Linear(module, rank=rank, is_critical=is_critical)
                new_module = new_module.to(module.weight.device)
                setattr(parent, child_name, new_module)

                if is_critical:
                    adapters_critical.append(new_module.adapter)
                else:
                    adapters_funlora.append(new_module.adapter)
            except AttributeError:
                pass

    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze adapters
    for adapter in adapters_critical + adapters_funlora:
        for param in adapter.parameters():
            param.requires_grad = True

    print(f"Injected: {len(adapters_critical)} full CASCADES + {len(adapters_funlora)} FunLoRA rank-1")
    return adapters_critical, adapters_funlora


def prepare_data(tokenizer, task_number, base_seed=42):
    """Loads domain-specific JSONL prompts for reasoning adaptation."""
    import os
    torch.manual_seed(base_seed + task_number)

    files = [
        "cascades_exp/task0_logic.jsonl",
        "cascades_exp/task1_decomp.jsonl",
        "cascades_exp/task2_action.jsonl"
    ]
    file_path = files[task_number % len(files)]

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing dataset: {file_path}. Please generate it using the prompt and save it here.")

    df = pd.read_json(file_path, lines=True)
    # Concatenate prompt and response to train causal LM
    prompts = [f"USER: {p}\nMODEL: {r}" for p, r in zip(df['prompt'], df['response'])]

    encodings = tokenizer(prompts, return_tensors="pt", truncation=True, padding=True, max_length=256)
    dataset = torch.utils.data.TensorDataset(encodings.input_ids, encodings.attention_mask)
    return DataLoader(dataset, batch_size=1, shuffle=True)


def estimate_quant_noise(model):
    """Estimate quantization noise floor from 4-bit weight statistics."""
    stds = []
    for name, module in model.named_modules():
        if type(module).__name__ == "Linear4bit" and hasattr(module, 'weight'):
            try:
                w = module.weight.data.float()
                stds.append(w.std().item())
            except Exception:
                pass
    return np.mean(stds) if stds else 0.0


def batched_null_space_extraction(adapters):
    if not adapters: return
    with torch.no_grad():
        C_Us = torch.stack([a.streaming_sketch_U.T @ a.streaming_sketch_U for a in adapters])
        _, V_eigs = torch.linalg.eigh(C_Us)
        for i, a in enumerate(adapters):
            k = a.Q_null_U.shape[1]
            occ_ambient_U = a.streaming_sketch_U @ V_eigs[i, :, -k:]
            if occ_ambient_U.norm() > 1e-8:
                Q_U, _ = torch.linalg.qr(occ_ambient_U)
                a.Q_null_U.copy_(Q_U)
                a.ear_initialized = True

def batched_autopoiesis_and_svc(adapters, ENABLE_SVC, ENABLE_PACA, ENABLE_COSO_NULLSPACE):
    if not adapters: return
    with torch.no_grad():
        # --- A. CONTINUOUS RANK CONTRACTION (EXHALE) ---
        C_Ls, C_Rs, exhaling_adapters = [], [], []
        for a in adapters:
            current_r = a.U_shared.shape[1]
            if current_r > 2:
                C_L = sum(c @ c.T for c in a.liquid_core.core_pool)
                C_R = sum(c.T @ c for c in a.liquid_core.core_pool)
                C_Ls.append(C_L)
                C_Rs.append(C_R)
                exhaling_adapters.append(a)
                
        if exhaling_adapters:
            # Batched Eigh
            D_Us, O_Us = torch.linalg.eigh(torch.stack(C_Ls))
            D_Vs, O_Vs = torch.linalg.eigh(torch.stack(C_Rs))
            
            for i, a in enumerate(exhaling_adapters):
                if (D_Us[i, 0] / (D_Us[i].sum() + 1e-8) < 0.0001) and (D_Vs[i, 0] / (D_Vs[i].sum() + 1e-8) < 0.0001):
                    current_r = a.U_shared.shape[1]
                    print(f"🫁 Breathing Manifold Triggered! Slicing dead rank {current_r} -> {current_r - 1}")
                    O_U, O_V = O_Us[i], O_Vs[i]
                    a.U_shared.data = a.U_shared.data @ O_U
                    a.V_shared.data = O_V.T @ a.V_shared.data
                    for k in range(a.liquid_core.num_cores):
                        a.liquid_core.core_pool.data[k] = O_U.T @ a.liquid_core.core_pool.data[k] @ O_V
                    a.ema_U.data = a.ema_U.data @ O_U
                    a.ema_V.data = O_V.T @ a.ema_V.data
                    if ENABLE_COSO_NULLSPACE: a.streaming_sketch_U.data = a.streaming_sketch_U.data @ O_U
                    if ENABLE_PACA:
                        a.ema_fast_U.data = a.ema_fast_U.data @ O_U
                        a.ema_slow_U.data = a.ema_slow_U.data @ O_U
                        a.ema_fast_V.data = O_V.T @ a.ema_fast_V.data
                        a.ema_slow_V.data = O_V.T @ a.ema_slow_V.data
                    
                    a.U_shared.data = a.U_shared.data[:, 1:]
                    a.V_shared.data = a.V_shared.data[1:, :]
                    a.liquid_core.core_pool.data = a.liquid_core.core_pool.data[:, 1:, 1:]
                    a.ema_U.data = a.ema_U.data[:, 1:]
                    a.ema_V.data = a.ema_V.data[1:, :]
                    if ENABLE_PACA:
                        a.ema_fast_U.data = a.ema_fast_U.data[:, 1:]
                        a.ema_slow_U.data = a.ema_slow_U.data[:, 1:]
                        a.ema_fast_V.data = a.ema_fast_V.data[1:, :]
                        a.ema_slow_V.data = O_V.T @ a.ema_slow_V.data
                        a.ema_fast_V.data = a.ema_fast_V.data[1:, :]
                        a.ema_slow_V.data = a.ema_slow_V.data[1:, :]
                    if ENABLE_COSO_NULLSPACE:
                        a.streaming_sketch_U.data = a.streaming_sketch_U.data[:, 1:]
                        a.ear_initialized = False
                        a.Q_null_U = torch.zeros(a.out_features, max(1, (a.U_shared.shape[1]) // 2), device=a.U_shared.device)
                    a.contracted_this_step = True
                    
        # --- B. SINGULAR VALUE CALIBRATION (SVC) BUGFIX ---
        if ENABLE_SVC:
            cores, core_refs = [], []
            for a in adapters:
                for k in range(a.liquid_core.num_cores):
                    cores.append(a.liquid_core.core_pool[k].data)
                    core_refs.append((a, k))
            if cores:
                U_s, S_s, V_h = torch.linalg.svd(torch.stack(cores), full_matrices=False)
                svc_lambda = core_refs[0][0].svc_lambda
                S_s = S_s / (1 + svc_lambda * S_s)
                reconstructed = U_s @ torch.diag_embed(S_s) @ V_h
                for idx, (a, k) in enumerate(core_refs):
                    a.liquid_core.core_pool.data[k] = reconstructed[idx]

def train_cascades_v3(seed=42):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set basic global seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Print ablation config
    print("=" * 60)
    print("CASCADES REASONING PIPELINE")
    print("=" * 60)
    print(f"Ablation flags:")
    print(f"  PaCA (CaLoRA):        {ENABLE_PACA}")
    print(f"  DEAL heat kernel:     {ENABLE_DEAL}")
    print(f"  GainLoRA gate:        {ENABLE_GAINLORA_GATE}")
    print(f"  CoSO null-space:      {ENABLE_COSO_NULLSPACE}")
    print(f"  CL-LoRA reassignment: {ENABLE_CLLORA_REASSIGN}")
    print(f"  SVC calibration:      {ENABLE_SVC}")
    print(f"  D-MoLE layer select:  {ENABLE_DMOLE_SELECT}")
    print(f"  FunLoRA rank-1:       {ENABLE_FUNLORA}")
    print(f"  Device: {device}")
    print("=" * 60)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    model_id = "p-e-w/Qwen3-4B-Instruct-2507-heretic"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
    )

    # Estimate quantization noise floor for DEAL filter
    quant_noise = estimate_quant_noise(model)
    print(f"Estimated quantization noise std: {quant_noise:.6f}")

    # D-MoLE: compute layer importance (optional)
    layer_importance = None
    if ENABLE_DMOLE_SELECT:
        probe_loader = prepare_data(tokenizer, 0, base_seed=seed)
        layer_importance = compute_layer_importance(model, probe_loader, device)

    # Inject adapters
    critical_adapters, funlora_adapters = inject_cascades_v3(
        model, rank=8, layer_importance=layer_importance
    )
    all_adapters = critical_adapters + funlora_adapters

    # Set quantization noise on critical adapters
    for a in critical_adapters:
        a.quant_noise_std.fill_(quant_noise)

    num_tasks = 3
    epochs = 3
    accuracy_matrix = np.zeros((num_tasks, num_tasks))
    start_time = time.time()

    for t in range(num_tasks):
        print(f"\n{'=' * 60}")
        print(f"--- Training CASCADES Task {t} ---")
        print(f"{'=' * 60}")

        dataloader = prepare_data(tokenizer, t, base_seed=seed)

        # Build optimizer: shared subspace + task lambdas + gates + FunLoRA params
        def build_optimizer():
            param_groups = []
            assigned_param_ids = set()
            
            # Recalculate which adapters are currently critical/funlora after potential dynamic D-MoLE swaps
            current_critical = [m.adapter for m in model.modules() if isinstance(m, CASCADES_v6_Linear) and m.is_critical]
            current_funlora  = [m.adapter for m in model.modules() if isinstance(m, CASCADES_v6_Linear) and not m.is_critical]

            liquid_core_params = [p for a in current_critical for p in a.liquid_core.parameters()]
            if liquid_core_params:
                param_groups.append({'params': liquid_core_params, 'lr': 5e-3})
                assigned_param_ids.update(id(p) for p in liquid_core_params)

            if ENABLE_GAINLORA_GATE:
                gate_params = [p for a in current_critical if hasattr(a, 'gate_proj') for p in a.gate_proj.parameters()]
                if gate_params:
                    param_groups.append({'params': gate_params, 'lr': 1e-3})
                    assigned_param_ids.update(id(p) for p in gate_params)

            funlora_params = [p for a in current_funlora for p in a.parameters()]
            if funlora_params:
                param_groups.append({'params': funlora_params, 'lr': 1e-4})
                assigned_param_ids.update(id(p) for p in funlora_params)

            stiefel_bases = []
            for layer in current_critical:
                stiefel_bases.extend([id(layer.U_shared), id(layer.V_shared)])
                
            fallback_params = [
                p for p in model.parameters() 
                if p.requires_grad 
                and id(p) not in stiefel_bases 
                and id(p) not in assigned_param_ids
            ]
            
            if fallback_params:
                param_groups.append({"params": fallback_params, "lr": 5e-4})
                
            return optim.Adam(param_groups), current_critical
            
        optimizer, critical_adapters = build_optimizer()

        # CASCADES v6: Boundary-less architecture. No explicit CoSO null-space precomputation at boundaries required.

        for ep in range(epochs):
            epoch_loss = 0
            num_batches = 0
            for batch in dataloader:
                print(f"    [Ep {ep+1} B {num_batches+1}] Starting batch...")
                input_ids, attention_mask = batch
                input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

                print(f"    [Ep {ep+1} B {num_batches+1}] Forward pass...")
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
                print(f"    [Ep {ep+1} B {num_batches+1}] Backward pass (loss={loss.item():.4f})...")
                loss.backward()

                print(f"    [Ep {ep+1} B {num_batches+1}] Gradient clipping...")
                trainable = [p for p in model.parameters() if p.grad is not None]
                if trainable:
                    torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)

                # === CASCADES v3.1 Full Pipeline (critical adapters only) ===
                print(f"    [Ep {ep+1} B {num_batches+1}] Adapter CASCADES descent step...")
                for a in critical_adapters:
                     a.full_descent_step(lr=0.01)
                     
                     # V9: Bi-Directional Optimizer State Elasticity Hook 
                     param = a.liquid_core.core_pool
                     if getattr(a, 'contracted_this_step', False):
                         if param in optimizer.state:
                             # Basis was globally rotated & sliced. Flush dead momentum.
                             optimizer.state[param] = {}
                         a.contracted_this_step = False
                     elif param in optimizer.state:
                         state = optimizer.state[param]
                         if 'exp_avg' in state and state['exp_avg'].shape != param.shape:
                             # V8 Expansion: Zero-pad the new ranks (+1)
                             state['exp_avg'] = F.pad(state['exp_avg'], (0, 1, 0, 1), "constant", 0.0)
                             state['exp_avg_sq'] = F.pad(state['exp_avg_sq'], (0, 1, 0, 1), "constant", 0.0)

                print(f"    [Ep {ep+1} B {num_batches+1}] Optimizer step...")
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
                
                # Global Batched Operations (Replaces layer-by-layer CUDA bottleneck)
                if ENABLE_COSO_NULLSPACE and num_batches % 25 == 0:
                    batched_null_space_extraction(critical_adapters)

                if ENABLE_SVC and num_batches % 50 == 0:
                    batched_autopoiesis_and_svc(critical_adapters, ENABLE_SVC, ENABLE_PACA, ENABLE_COSO_NULLSPACE)

                print(f"    [Ep {ep+1} B {num_batches}] Batch finished.")

            avg_loss = epoch_loss / max(num_batches, 1)
            print(f"Task {t}, Epoch {ep + 1}/{epochs}, Avg Loss: {avg_loss:.4f}")

        # --- Continuous D-MoLE Migration (Phase-Transition Hardware) ---
        if t < num_tasks - 1: # Perform migration at the end of each task (except the last)
            print(f"\n{'=' * 60}")
            print(f"Phase-Transition: Continuous D-MoLE Migration")
            print(f"{'=' * 60}")
            
            # Recalculate importance to find the new most active non-critical & least active critical
            current_wrapper_modules = {name: m for name, m in model.named_modules() if isinstance(m, CASCADES_v6_Linear)}
            
            # Re-evaluate layer variance to map topological activations dynamically
            importance_scores = compute_layer_importance(model, dataloader, device, threshold=0.15)
            
            promoted, demoted = 0, 0
            if importance_scores:
                # Rank layers by importance (highest score first)
                sorted_layers = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
                
                # Top 10% most active overall -> Should be Critical (Rank-8)
                # Bottom 50% least active overall -> Should be FunLoRA (Rank-1)
                top_k = max(2, len(sorted_layers) // 10)
                bottom_k = max(2, len(sorted_layers) // 2)
                
                most_active_names = [name for name, score in sorted_layers[:top_k]]
                least_active_names = [name for name, score in sorted_layers[-bottom_k:]]
                
                # Demote dormant criticals
                for name in least_active_names:
                    if name in current_wrapper_modules:
                        w = current_wrapper_modules[name]
                        if w.is_critical:
                            if w.demote(): demoted += 1
                            
                # Promote highly active non-criticals
                for name in most_active_names:
                    if name in current_wrapper_modules:
                        w = current_wrapper_modules[name]
                        if not w.is_critical:
                            if w.promote(): promoted += 1
                    
            print(f"Dynamically Promoted (FunLoRA -> ResonantCore): {promoted}")
            print(f"Dynamically Demoted (ResonantCore -> FunLoRA): {demoted}")
            print(f"VRAM Optimization: Stale optimizer momentum buffers will be wiped and reconstructed.")

        # --- Evaluation ---
        print(f"\n--- Evaluation after Task {t} ---")
        for eval_t in range(t + 1):
            eval_dataloader = prepare_data(tokenizer, eval_t, base_seed=seed)
            total_loss = 0
            num_batches = 0
            with torch.no_grad():
                for batch in eval_dataloader:
                    input_ids, attention_mask = batch
                    input_ids = input_ids.to(device)
                    out = model(input_ids=input_ids, labels=input_ids)
                    total_loss += out.loss.item()
                    num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)
            proxy_acc = math.exp(-avg_loss)
            accuracy_matrix[t, eval_t] = proxy_acc
            print(f"  Task {eval_t} proxy accuracy: {proxy_acc * 100:.2f}% (avg_loss: {avg_loss:.4f})")

    end_time = time.time()

    # Final metrics
    final_accs = accuracy_matrix[-1, :]
    avg_acc = np.mean(final_accs)

    bwt_list = []
    for i in range(num_tasks - 1):
        bwt_list.append(accuracy_matrix[-1, i] - accuracy_matrix[i, i])
    bwt = np.mean(bwt_list)

    print(f"\n{'=' * 60}")
    print("=== FINAL CASCADES REASONING METRICS ===")
    print(f"{'=' * 60}")
    print(f"Average Accuracy Proxy: {avg_acc * 100:.2f}%")
    print(f"Backward Transfer (BWT): {bwt * 100:.2f}%")
    print(f"Total Time: {end_time - start_time:.2f}s")
    print(f"\nAccuracy Matrix:")
    for i in range(num_tasks):
        row = " | ".join([
            f"{accuracy_matrix[i, j] * 100:6.2f}%" if accuracy_matrix[i, j] > 0 else "   —   "
            for j in range(num_tasks)
        ])
        print(f"  After T{i}: {row}")

    # Ablation config in output
    print(f"\nAblation: PaCA={ENABLE_PACA} DEAL={ENABLE_DEAL} Gate={ENABLE_GAINLORA_GATE} "
          f"CoSO={ENABLE_COSO_NULLSPACE} CL-LoRA={ENABLE_CLLORA_REASSIGN} SVC={ENABLE_SVC} "
          f"D-MoLE={ENABLE_DMOLE_SELECT} FunLoRA={ENABLE_FUNLORA}")

    df = pd.DataFrame(accuracy_matrix,
                      columns=[f"Eval_T{i}" for i in range(num_tasks)],
                      index=[f"Train_T{i}" for i in range(num_tasks)])
    df.to_csv("cascades_reasoning_results.csv")
    print("\nResults saved to cascades_reasoning_results.csv")

    # --- Phase 12: Weight Persistence ---
    print("\nSaving CASCADES v9 Manifold Weights...")
    save_path = "cascades_v9_weights.pt"
    # Extract only the adapter state dictionary to minimize file size
    adapter_state = {name: param for name, param in model.named_parameters() if 'wrapper' in name or 'adapter' in name}
    torch.save(adapter_state, save_path)
    print(f"Weights saved successfully to {save_path}")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CASCADES v9 Ecosystem Training")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    args = parser.parse_args()
    
    train_cascades_v3(seed=args.seed)
