"""
CASCADES model injection and D-MoLE layer selection utilities.

Provides functions to:
  - Compute layer importance via activation variance (D-MoLE)
  - Inject CASCADES adapters into HuggingFace models
  - Batch operations for null-space extraction and autopoiesis/SVC
  - Estimate quantization noise floor for DEAL filter
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cascades.adapters import CASCADESAdapter, CASCADESLinear, FunLoRA_Adapter
from cascades.config import AblationConfig, DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# A. Quantization noise estimation
# ---------------------------------------------------------------------------

def estimate_quant_noise(model: nn.Module) -> float:
    """Estimate quantization noise floor from 4-bit weight statistics.

    Scans all Linear4bit layers and computes the mean standard deviation
    of their dequantized weights. This value parameterizes the DEAL filter's
    noise floor gate (ε_quant).

    Args:
        model: Quantized HuggingFace model.

    Returns:
        Mean weight std across all quantized layers (0.0 if none found).
    """
    import numpy as np

    stds = []
    for name, module in model.named_modules():
        if type(module).__name__ == "Linear4bit" and hasattr(module, 'weight'):
            try:
                w = module.weight.data.float()
                stds.append(w.std().item())
            except Exception:
                pass
    return float(np.mean(stds)) if stds else 0.0


# ---------------------------------------------------------------------------
# B. D-MoLE layer importance scoring
# ---------------------------------------------------------------------------

def compute_layer_importance(
    model: nn.Module,
    dataloader,
    device: str,
    threshold: float = 0.15,
    config: AblationConfig = DEFAULT_CONFIG,
) -> dict[str, bool]:
    """D-MoLE ICML'25: activation-variance-based layer importance scoring.

    For 4-bit models, uses output variance as a proxy since 4-bit weights
    don't support requires_grad. Layers above the threshold are marked critical.

    Args:
        model: The model with injected adapters.
        dataloader: DataLoader for importance probing.
        device: Device string.
        threshold: Normalized importance threshold for criticality.
        config: Ablation configuration.

    Returns:
        Dictionary mapping layer names to criticality booleans.
    """
    if not config.enable_dmole_select:
        return {}

    activation_stats: dict[str, float] = {}
    hooks = []

    for name, module in model.named_modules():
        if (
            isinstance(module, CASCADESLinear)
            or isinstance(module, nn.Linear)
            or type(module).__name__ == "Linear4bit"
        ):
            activation_stats[name] = 0.0

            def make_hook(layer_name):
                def hook_fn(mod, inp, out):
                    if isinstance(out, torch.Tensor):
                        activation_stats[layer_name] += out.float().var().item()
                    elif isinstance(out, tuple) and isinstance(out[0], torch.Tensor):
                        activation_stats[layer_name] += out[0].float().var().item()
                return hook_fn

            hooks.append(module.register_forward_hook(make_hook(name)))

    max_batches = 3
    batches_processed = 0
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            if batches_processed >= max_batches:
                break
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            model(input_ids=input_ids, attention_mask=attention_mask)
            batches_processed += 1

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

    critical = {k: v >= threshold for k, v in importance.items()}
    n_critical = sum(1 for v in critical.values() if v)
    n_total = len(critical)
    print(f"D-MoLE: {n_critical}/{n_total} layers marked critical (threshold={threshold})")
    return critical


# ---------------------------------------------------------------------------
# C. Adapter injection
# ---------------------------------------------------------------------------

def inject_cascades(
    model: nn.Module,
    rank: int = 8,
    target_modules: Optional[list[str]] = None,
    layer_importance: Optional[dict[str, bool]] = None,
    config: AblationConfig = DEFAULT_CONFIG,
) -> tuple[list[CASCADESAdapter], list[FunLoRA_Adapter]]:
    """Inject CASCADES adapters into a HuggingFace model.

    Replaces target linear layers with CASCADESLinear wrappers.
    Critical layers get full CASCADESAdapter; non-critical get FunLoRA_Adapter.

    v10: Auto-detects GQA broadcast ratio from model.config and sets
    per-adapter gqa_ratio for K/V projections, enabling the GQA-aware
    metric preconditioning that resolves the 8B scaling paradox.

    Args:
        model: The HuggingFace model to inject into.
        rank: Stiefel manifold rank for critical adapters.
        target_modules: List of layer name substrings to target.
        layer_importance: D-MoLE criticality map (from compute_layer_importance).
        config: Ablation configuration.

    Returns:
        Tuple of (critical_adapters, funlora_adapters) lists.
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "up_proj", "down_proj"]

    # v10: Auto-detect GQA broadcast ratio from model config
    num_q_heads = getattr(getattr(model, 'config', None), 'num_attention_heads', 1)
    num_kv_heads = getattr(getattr(model, 'config', None), 'num_key_value_heads', num_q_heads)
    gqa_ratio = num_q_heads / max(num_kv_heads, 1)
    if gqa_ratio > 1.0:
        print(f"v10 GQA detected: {num_q_heads}Q / {num_kv_heads}KV = {gqa_ratio:.1f}x broadcast ratio")

    # Substrings that identify K/V projections (receive GQA preconditioning)
    kv_proj_names = ["k_proj", "v_proj"]

    adapters_critical: list[CASCADESAdapter] = []
    adapters_funlora: list[FunLoRA_Adapter] = []

    for name, module in dict(model.named_modules()).items():
        if any(t in name for t in target_modules) and (
            isinstance(module, nn.Linear)
            or type(module).__name__ == "Linear4bit"
        ):
            parts = name.split(".")
            parent_name = ".".join(parts[:-1])
            child_name = parts[-1]
            try:
                parent = model.get_submodule(parent_name)

                is_critical = True
                if config.enable_dmole_select and config.enable_funlora and layer_importance:
                    is_critical = layer_importance.get(name, True)

                new_module = CASCADESLinear(
                    module, rank=rank, is_critical=is_critical, config=config
                )
                new_module = new_module.to(module.weight.device)

                # v10: Set per-adapter GQA ratio for K/V projections
                if is_critical and gqa_ratio > 1.0:
                    is_kv = any(kv in name for kv in kv_proj_names)
                    new_module.adapter.gqa_ratio = gqa_ratio if is_kv else 1.0

                setattr(parent, child_name, new_module)

                if is_critical:
                    adapters_critical.append(new_module.adapter)
                else:
                    adapters_funlora.append(new_module.adapter)
            except AttributeError:
                pass

    # Freeze base model, unfreeze adapters
    for param in model.parameters():
        param.requires_grad = False
    for adapter in adapters_critical + adapters_funlora:
        for param in adapter.parameters():
            param.requires_grad = True

    print(
        f"Injected: {len(adapters_critical)} full CASCADES + "
        f"{len(adapters_funlora)} FunLoRA rank-1"
    )
    return adapters_critical, adapters_funlora


# ---------------------------------------------------------------------------
# D. Batched global operations (replaces per-layer CUDA bottleneck)
# ---------------------------------------------------------------------------

def batched_null_space_extraction(adapters: list[CASCADESAdapter]) -> None:
    """Amortized null-space extraction across all adapters, batched by rank."""
    if not adapters:
        return

    by_rank: dict[int, list[CASCADESAdapter]] = {}
    for a in adapters:
        r = a.U_shared.shape[1]
        if r not in by_rank:
            by_rank[r] = []
        by_rank[r].append(a)

    with torch.no_grad():
        for r, rank_adapters in by_rank.items():
            C_Us = torch.stack([
                a.streaming_sketch_U.T @ a.streaming_sketch_U
                for a in rank_adapters
            ])
            _, V_eigs = torch.linalg.eigh(C_Us)
            for i, a in enumerate(rank_adapters):
                k = a.Q_null_U.shape[1]
                occ_ambient_U = a.streaming_sketch_U @ V_eigs[i, :, -k:]
                if occ_ambient_U.norm() > 1e-8:
                    Q_U, _ = torch.linalg.qr(occ_ambient_U)
                    a.Q_null_U.copy_(Q_U)
                    a.ear_initialized = True


def batched_autopoiesis_and_svc(
    adapters: list[CASCADESAdapter],
    config: AblationConfig = DEFAULT_CONFIG,
) -> None:
    """Batched rank contraction (exhale) and SVC calibration.

    A. Continuous rank contraction: uses relative variance bound to detect
       dead channels and slice them from the manifold.
    B. Singular value calibration: soft-thresholds core singular values.
    """
    if not adapters:
        return

    with torch.no_grad():
        # --- A. CONTINUOUS RANK CONTRACTION (EXHALE) ---
        # IMPORTANT: We ZERO OUT dead dimensions instead of physically slicing
        # parameter tensors. Slicing .data changes tensor shapes mid-training
        # which breaks PyTorch's autograd graph (PermuteBackward0 shape mismatch).
        by_rank: dict[int, list[CASCADESAdapter]] = {}
        for a in adapters:
            r = a.U_shared.shape[1]
            n_dead = getattr(a, '_dead_ranks', 0)
            effective_r = r - n_dead
            if effective_r > 2:
                if r not in by_rank:
                    by_rank[r] = []
                by_rank[r].append(a)

        for r, rank_adapters in by_rank.items():
            C_Ls = torch.stack([
                sum(c @ c.T for c in a.liquid_core.core_pool)
                for a in rank_adapters
            ])
            C_Rs = torch.stack([
                sum(c.T @ c for c in a.liquid_core.core_pool)
                for a in rank_adapters
            ])

            D_Us, O_Us = torch.linalg.eigh(C_Ls)
            D_Vs, O_Vs = torch.linalg.eigh(C_Rs)

            for i, a in enumerate(rank_adapters):
                n_dead = getattr(a, '_dead_ranks', 0)
                check_idx = n_dead  # First live dimension after dead ones
                if check_idx >= r - 1:
                    continue
                if (
                    D_Us[i, check_idx] / (D_Us[i, check_idx:].sum() + 1e-8) < 0.0001
                    and D_Vs[i, check_idx] / (D_Vs[i, check_idx:].sum() + 1e-8) < 0.0001
                ):
                    effective_r = r - n_dead
                    print(
                        f"🫁 Breathing Manifold Triggered! "
                        f"Zeroing dead rank (effective {effective_r} -> {effective_r - 1})"
                    )
                    O_U, O_V = O_Us[i], O_Vs[i]
                    # Counter-rotate into eigenbasis
                    a.U_shared.data = a.U_shared.data @ O_U
                    a.V_shared.data = O_V.T @ a.V_shared.data
                    for k in range(a.liquid_core.num_cores):
                        a.liquid_core.core_pool.data[k] = (
                            O_U.T @ a.liquid_core.core_pool.data[k] @ O_V
                        )
                    a.ema_U.data = a.ema_U.data @ O_U
                    a.ema_V.data = O_V.T @ a.ema_V.data
                    if config.enable_coso_nullspace:
                        a.streaming_sketch_U.data = a.streaming_sketch_U.data @ O_U
                    if config.enable_paca:
                        a.ema_fast_U.data = a.ema_fast_U.data @ O_U
                        a.ema_slow_U.data = a.ema_slow_U.data @ O_U
                        a.ema_fast_V.data = O_V.T @ a.ema_fast_V.data
                        a.ema_slow_V.data = O_V.T @ a.ema_slow_V.data

                    # Zero out the dead dimension instead of slicing
                    dead_idx = check_idx
                    a.U_shared.data[:, dead_idx] = 0.0
                    a.V_shared.data[dead_idx, :] = 0.0
                    for k in range(a.liquid_core.num_cores):
                        a.liquid_core.core_pool.data[k, dead_idx, :] = 0.0
                        a.liquid_core.core_pool.data[k, :, dead_idx] = 0.0
                    a.ema_U.data[:, dead_idx] = 0.0
                    a.ema_V.data[dead_idx, :] = 0.0
                    if config.enable_paca:
                        a.ema_fast_U.data[:, dead_idx] = 0.0
                        a.ema_slow_U.data[:, dead_idx] = 0.0
                        a.ema_fast_V.data[dead_idx, :] = 0.0
                        a.ema_slow_V.data[dead_idx, :] = 0.0
                    if config.enable_coso_nullspace:
                        a.streaming_sketch_U.data[:, dead_idx] = 0.0
                    a._dead_ranks = n_dead + 1
                    a._last_dead_idx = dead_idx  # For surgical optimizer state cleanup
                    a.contracted_this_step = True

        # --- B. SINGULAR VALUE CALIBRATION (SVC) ---
        if config.enable_svc:
            svc_by_rank: dict[int, list[CASCADESAdapter]] = {}
            for a in adapters:
                r = a.U_shared.shape[1]
                if r not in svc_by_rank:
                    svc_by_rank[r] = []
                svc_by_rank[r].append(a)

            for r, rank_adapters in svc_by_rank.items():
                cores, core_refs = [], []
                for a in rank_adapters:
                    for k in range(a.liquid_core.num_cores):
                        cores.append(a.liquid_core.core_pool[k].data)
                        core_refs.append((a, k))
                if cores:
                    U_s, S_s, V_h = torch.linalg.svd(
                        torch.stack(cores), full_matrices=False
                    )
                    svc_lambda = core_refs[0][0].svc_lambda
                    S_s = S_s / (1 + svc_lambda * S_s)
                    reconstructed = U_s @ torch.diag_embed(S_s) @ V_h
                    for idx, (a, k) in enumerate(core_refs):
                        a.liquid_core.core_pool.data[k] = reconstructed[idx]
