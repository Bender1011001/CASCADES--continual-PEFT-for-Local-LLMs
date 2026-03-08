"""
CASCADES-HX: Hybrid eXpansion for Qwen3.5 hybrid linear/full attention models.

Qwen3.5-4B uses a 3:1 layout of Gated DeltaNet (linear attention) and standard
Gated Softmax attention. Actual layer structure confirmed from model weight map:

  Linear attention layers (24/32, e.g. layers 0,1,2,4,5,6,...):
    linear_attn.in_proj_a     — Q-like content projection        [TARGET]
    linear_attn.in_proj_b     — K-like content projection        [TARGET]
    linear_attn.in_proj_qkv   — Fused QKV content projection     [TARGET]
    linear_attn.in_proj_z     — Selective gate projection        [TARGET]
    linear_attn.out_proj      — Output projection                [TARGET]
    linear_attn.A_log         — Recurrent decay parameter        [BLACKLIST - not Linear]
    linear_attn.conv1d        — Recurrent conv kernel            [BLACKLIST - Conv1d]
    linear_attn.dt_bias       — Time-step bias                   [BLACKLIST - not Linear]
    linear_attn.norm          — Internal layernorm               [BLACKLIST - LayerNorm]

  Full attention layers (8/32, every 4th: 3,7,11,...):
    self_attn.q_proj          — Query projection                 [TARGET]
    self_attn.k_proj          — Key projection (GQA ratio=4.0)   [TARGET]
    self_attn.v_proj          — Value projection (GQA ratio=4.0) [TARGET]
    self_attn.o_proj          — Output projection                [TARGET]
    self_attn.q_norm          — Query RMSNorm                    [BLACKLIST - LayerNorm]
    self_attn.k_norm          — Key RMSNorm                      [BLACKLIST - LayerNorm]

  MLP (all 32 layers):
    mlp.gate_proj, mlp.up_proj, mlp.down_proj                    [TARGET]

This module provides:
  1. RecurrentSafe_FunLoRA_Adapter: Zero-mean-at-init adapter for SSM layers.
     Standard FunLoRA uses x + sigmoid(x) + tanh(x). sigmoid(0) = 0.5, so at
     initialization the output has a nonzero mean, which can bias-shift the
     DeltaNet recurrent state. SiLU(0) = 0, so x + silu(x) is safe.

  2. lock_abliteration_permanent(): Appends the OBLITERATUS refusal direction
     into frozen_null_basis (permanent, not decayed by EMA). This is the v10-native
     approach — streaming_sketch_U decays at beta_ear=0.999 (~1000 step half-life)
     and would lose the lock during training.

  3. compute_hybrid_layer_importance(): Stratified D-MoLE that normalizes activation
     variance independently within SSM / Softmax / MLP strata. Without stratification,
     the 4.0x variance difference between softmax and SSM layers causes the allocator
     to dump all critical slots into one architecture type.

  4. inject_hybrid_cascades(): Proper Qwen3.5 injection with:
     - Correct target layer names (verified from weight map)
     - GQA ratio 1.0 for linear_attn (no grouping: linear_num_key_heads=16=num_heads)
     - GQA ratio 4.0 for self_attn k_proj/v_proj (4 KV heads, 16 Q heads)
     - RecurrentSafe_FunLoRA for non-critical linear_attn layers
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cascades.adapters import CASCADESAdapter, CASCADESLinear, FunLoRA_Adapter
from cascades.config import AblationConfig, DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# A. Recurrent-Safe FunLoRA — zero mean at initialization for DeltaNet layers
# ---------------------------------------------------------------------------

class RecurrentSafe_FunLoRA_Adapter(nn.Module):
    """Rank-1 adapter that preserves zero-mean state geometry for DeltaNet layers.

    Standard FunLoRA_Adapter uses f(x) = x + sigmoid(x) + tanh(x).
    At initialization with small random weights, sigmoid(bottleneck) ≈ 0.5,
    injecting a constant positive bias into the DeltaNet's recurrent state gate.
    Over many steps this shifts the learned decay rate of the linear attention.

    This adapter uses f(x) = x + silu(x) instead:
      - silu(0) = 0 → zero mean at initialization, no bias injection
      - f'(0) = 1 + 0.5 = 1.5 (vs 2.25 for standard FunLoRA)

    Note on demotion compatibility: Non-critical SSM layers are injected directly
    as RecurrentSafe and never promoted/demoted (they stay non-critical), so the
    2.25 gain factor in CASCADESLinear.demote() never touches these adapters.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.a = nn.Parameter(torch.randn(out_features, 1) * 0.05)
        self.b = nn.Parameter(torch.randn(1, in_features) * 0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck = x @ self.b.to(x.dtype).T
        # silu(0) = 0 — safe for recurrent state gates
        expanded = bottleneck + F.silu(bottleneck)
        return expanded @ self.a.to(x.dtype).T


# ---------------------------------------------------------------------------
# B. Permanent abliteration lock via frozen_null_basis
# ---------------------------------------------------------------------------

def lock_abliteration_permanent(
    critical_adapters: list[CASCADESAdapter],
    refusal_vector: torch.Tensor,
) -> int:
    """Permanently lock the OBLITERATUS refusal direction into the v10 frozen null-basis.

    The frozen_null_basis is a structural orthonormal basis that the EAR gradient
    reassignment projects out of at every training step. Unlike streaming_sketch_U
    (which is an EMA buffer decaying at beta_ear=0.999 — ~1000 step half-life),
    the frozen_null_basis persists for the entire lifetime of the model and is never
    touched by EMA decay.

    This makes the abliteration lock mathematically permanent: the optimizer is
    physically incapable of updating any adapter in the direction of refusal behavior,
    regardless of how many sequential tasks are trained.

    Call this ONCE immediately after inject_hybrid_cascades(), before training starts.

    Args:
        critical_adapters: List of CASCADESAdapter instances from inject_hybrid_cascades().
        refusal_vector: The refusal direction from OBLITERATUS (shape: [d_model] = [2560]).
                        If OBLITERATUS did not save this vector separately, reconstruct it
                        by computing the mean difference of harmful vs. harmless activations
                        at the strong_layers listed in abliteration_metadata.json.

    Returns:
        Number of adapters locked.
    """
    print("\n Engaging CASCADES-HX Permanent Abliteration Lock...")

    with torch.no_grad():
        ref = refusal_vector.float()
        ref = ref / (ref.norm() + 1e-8)
        ref_col = ref.unsqueeze(1)  # (d_model, 1)

        locked_count = 0
        for adapter in critical_adapters:
            if not hasattr(adapter, 'frozen_null_basis'):
                continue

            # Only anchor to adapters whose output dimension matches d_model.
            # (Skip fused projections like in_proj_qkv whose out_dim ≠ 2560.)
            if adapter.U_shared.shape[0] != ref_col.shape[0]:
                continue

            device = adapter.U_shared.device
            dtype = adapter.U_shared.dtype
            ref_col_dev = ref_col.to(device=device, dtype=dtype)

            if adapter.frozen_null_basis.shape[1] == 0:
                # First direction — just store it
                adapter.frozen_null_basis = ref_col_dev.clone()
            else:
                # Append and re-orthogonalize to maintain numerical stability
                combined = torch.cat([adapter.frozen_null_basis, ref_col_dev], dim=1)
                Q, R = torch.linalg.qr(combined)
                # Drop numerically degenerate columns
                significant = R.diag().abs() > 1e-6
                adapter.frozen_null_basis = Q[:, significant]

            # Mark EAR as initialized so _cllora_reassign uses the frozen basis
            adapter.ear_initialized = True
            locked_count += 1

    print(f" Refusal direction permanently locked in {locked_count} adapters.")
    return locked_count


# ---------------------------------------------------------------------------
# C. Stratified D-MoLE for hybrid architectures
# ---------------------------------------------------------------------------

# Layer name patterns for each Qwen3.5 modality (verified from weight map)
_LINEAR_ATTN_PATTERNS = ("linear_attn",)
_FULL_ATTN_PATTERNS   = ("self_attn",)
_MLP_PATTERNS         = ("mlp",)


def _classify_layer(name: str) -> str:
    """Return modality stratum for a Qwen3.5 layer name."""
    if any(p in name for p in _LINEAR_ATTN_PATTERNS):
        return "linear_attn"
    if any(p in name for p in _FULL_ATTN_PATTERNS):
        return "full_attn"
    return "mlp"


def compute_hybrid_layer_importance(
    model: nn.Module,
    dataloader,
    device: str,
    top_p: float = 0.25,
    config: AblationConfig = DEFAULT_CONFIG,
) -> dict[str, bool]:
    """Stratified D-MoLE for Qwen3.5 hybrid architecture.

    Problem with standard D-MoLE on hybrid models: softmax attention layers
    produce activation variance ~4x higher than SSM layers due to the softmax
    normalization step. A single global threshold causes the allocator to assign
    all Stiefel cores to softmax layers, leaving all 24 linear_attn layers as
    non-critical FunLoRA — exactly the "Ghosting Trap."

    Fix: normalize activation variance independently within each architecture
    stratum (linear_attn / full_attn / mlp), then select the top top_p fraction
    within each stratum. This guarantees proportional representation.

    Args:
        model:      The model (before or after injection).
        dataloader: DataLoader returning (input_ids, attention_mask, labels).
        device:     Device string.
        top_p:      Fraction to mark critical within each stratum (0.25 = top 25%).
        config:     Ablation configuration.

    Returns:
        Dictionary mapping layer names to criticality booleans.
    """
    if not config.enable_dmole_select:
        return {}

    strata: dict[str, dict[str, float]] = {
        "linear_attn": {},
        "full_attn": {},
        "mlp": {},
    }
    hooks = []

    for name, module in model.named_modules():
        if not (
            isinstance(module, (CASCADESLinear, nn.Linear))
            or type(module).__name__ == "Linear4bit"
        ):
            continue

        stratum = _classify_layer(name)
        strata[stratum][name] = 0.0

        def make_hook(s, n):
            def hook_fn(mod, inp, out):
                # Handle SSM/tuple outputs (DeltaNet returns (hidden, cache))
                tensor = out[0] if isinstance(out, tuple) else out
                if isinstance(tensor, torch.Tensor):
                    # Variance per position, averaged — more stable than global var()
                    strata[s][n] += tensor.float().var(dim=-1).mean().item()
            return hook_fn

        hooks.append(module.register_forward_hook(make_hook(stratum, name)))

    model.eval()
    batches_done = 0
    with torch.no_grad():
        for batch in dataloader:
            if batches_done >= 3:
                break
            input_ids, attention_mask, labels = batch  # 3-item batch
            model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
            batches_done += 1

    for h in hooks:
        h.remove()

    if batches_done > 0:
        for s in strata:
            for k in strata[s]:
                strata[s][k] /= batches_done

    critical: dict[str, bool] = {}
    for stratum, stats in strata.items():
        if not stats:
            continue
        max_var = max(stats.values()) or 1.0
        norm_stats = {k: v / max_var for k, v in stats.items()}
        threshold = np.percentile(list(norm_stats.values()), 100 * (1 - top_p))
        n_crit = 0
        for k, v in norm_stats.items():
            critical[k] = v >= threshold
            if critical[k]:
                n_crit += 1
        print(
            f"D-MoLE [{stratum.upper():12s}] "
            f"{n_crit}/{len(stats)} critical "
            f"(threshold={threshold:.3f})"
        )

    return critical


# ---------------------------------------------------------------------------
# D. Hybrid injection for Qwen3.5
# ---------------------------------------------------------------------------

# Layers that form the DeltaNet recurrent kernel — must not be adapted.
# Note: A_log and dt_bias are plain Parameters (not nn.Linear), conv1d is
# nn.Conv1d, and norm is LayerNorm — all naturally skipped by the isinstance
# check. This blacklist is an explicit safety net for edge cases and future
# transformers versions that might wrap them differently.
_BLACKLIST = frozenset(["A_log", "conv1d", "dt_bias", "norm"])

# Content projection names for linear_attn (verified from weight map)
_LINEAR_ATTN_TARGETS = frozenset([
    "in_proj_a",   # Q-like content projection
    "in_proj_b",   # K-like content projection
    "in_proj_qkv", # Fused QKV content projection
    "in_proj_z",   # Selective gate projection
    "out_proj",    # Output projection
])

# Content projection names for full self_attn
_FULL_ATTN_TARGETS = frozenset(["q_proj", "k_proj", "v_proj", "o_proj"])

# MLP projection names
_MLP_TARGETS = frozenset(["gate_proj", "up_proj", "down_proj"])

# K/V projections in full attention that receive GQA preconditioning
_FULL_ATTN_KV = frozenset(["k_proj", "v_proj"])


def inject_hybrid_cascades(
    model: nn.Module,
    rank: int = 8,
    layer_importance: Optional[dict[str, bool]] = None,
    config: AblationConfig = DEFAULT_CONFIG,
) -> tuple[list[CASCADESAdapter], list[FunLoRA_Adapter | RecurrentSafe_FunLoRA_Adapter]]:
    """Inject CASCADES-HX adapters into Qwen3.5's hybrid architecture.

    Differences from standard inject_cascades():
      - Targets actual Qwen3.5 linear_attn projection names (in_proj_a/b/qkv/z/out_proj)
      - Sets gqa_ratio=1.0 for all linear_attn projections (no KV grouping in DeltaNet)
      - Sets gqa_ratio=4.0 for self_attn k_proj/v_proj (16Q / 4KV heads)
      - Uses RecurrentSafe_FunLoRA_Adapter for non-critical linear_attn layers
      - Blacklists recurrent dynamics params (explicit safety net)

    Args:
        model:            HuggingFace Qwen3.5 model (base or quantized).
        rank:             Stiefel manifold rank for critical adapters.
        layer_importance: D-MoLE criticality map from compute_hybrid_layer_importance().
                          If None, all matched layers are marked critical.
        config:           Ablation configuration.

    Returns:
        (critical_adapters, funlora_adapters) — pass critical_adapters to
        lock_abliteration_permanent() immediately after this call.
    """
    # GQA ratios from model config
    num_q  = getattr(getattr(model, 'config', None), 'num_attention_heads', 16)
    num_kv = getattr(getattr(model, 'config', None), 'num_key_value_heads', 4)
    full_attn_gqa = num_q / max(num_kv, 1)
    # Linear attention uses no GQA grouping (linear_num_key_heads == num_attention_heads)
    linear_attn_gqa = 1.0

    print(f"CASCADES-HX: Full-attn GQA ratio = {full_attn_gqa:.1f}x | "
          f"Linear-attn GQA ratio = {linear_attn_gqa:.1f}x")

    adapters_critical: list[CASCADESAdapter] = []
    adapters_funlora: list = []

    for name, module in dict(model.named_modules()).items():
        # Blacklist check by path component
        name_parts = name.split(".")
        if any(b in name_parts for b in _BLACKLIST):
            continue

        # Only wrap nn.Linear / Linear4bit
        if not (
            isinstance(module, nn.Linear)
            or type(module).__name__ == "Linear4bit"
        ):
            continue

        leaf = name_parts[-1]
        is_linear_attn = "linear_attn" in name
        is_full_attn   = "self_attn" in name
        is_mlp         = "mlp" in name

        # Check if this leaf is a valid target for its modality
        if is_linear_attn and leaf not in _LINEAR_ATTN_TARGETS:
            continue
        if is_full_attn and leaf not in _FULL_ATTN_TARGETS:
            continue
        if is_mlp and leaf not in _MLP_TARGETS:
            continue
        if not (is_linear_attn or is_full_attn or is_mlp):
            continue

        parent_name = ".".join(name_parts[:-1])
        try:
            parent = model.get_submodule(parent_name)
        except AttributeError:
            continue

        # D-MoLE criticality
        is_critical = True
        if config.enable_dmole_select and config.enable_funlora and layer_importance:
            is_critical = layer_importance.get(name, True)

        # Build the wrapper
        new_module = CASCADESLinear(
            module, rank=rank, is_critical=is_critical, config=config
        ).to(module.weight.device)

        if is_critical:
            # Set GQA ratio: only full_attn K/V get the broadcast inflation correction
            if is_full_attn and leaf in _FULL_ATTN_KV:
                new_module.adapter.gqa_ratio = full_attn_gqa
            else:
                new_module.adapter.gqa_ratio = linear_attn_gqa

            adapters_critical.append(new_module.adapter)

        else:
            # Non-critical: use recurrent-safe adapter for SSM layers
            if is_linear_attn:
                new_module.adapter = RecurrentSafe_FunLoRA_Adapter(
                    module.in_features, module.out_features
                ).to(module.weight.device)
            else:
                new_module.adapter = FunLoRA_Adapter(
                    module.in_features, module.out_features
                ).to(module.weight.device)

            adapters_funlora.append(new_module.adapter)

        setattr(parent, leaf, new_module)

    # Freeze backbone, train only adapters
    for param in model.parameters():
        param.requires_grad = False
    for adapter in adapters_critical + adapters_funlora:
        for param in adapter.parameters():
            param.requires_grad = True

    n_ssm  = sum(1 for a in adapters_funlora if isinstance(a, RecurrentSafe_FunLoRA_Adapter))
    n_std  = len(adapters_funlora) - n_ssm
    print(
        f"CASCADES-HX injection complete:\n"
        f"  {len(adapters_critical)} critical (ResonantCore)\n"
        f"  {n_ssm} non-critical SSM (RecurrentSafe_FunLoRA)\n"
        f"  {n_std} non-critical other (FunLoRA)"
    )
    return adapters_critical, adapters_funlora
