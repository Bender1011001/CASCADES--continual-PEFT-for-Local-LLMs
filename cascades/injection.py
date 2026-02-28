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
        by_rank: dict[int, list[CASCADESAdapter]] = {}
        for a in adapters:
            r = a.U_shared.shape[1]
            if r > 2:
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
                if (
                    D_Us[i, 0] / (D_Us[i].sum() + 1e-8) < 0.0001
                    and D_Vs[i, 0] / (D_Vs[i].sum() + 1e-8) < 0.0001
                ):
                    current_r = a.U_shared.shape[1]
                    print(
                        f"🫁 Breathing Manifold Triggered! "
                        f"Slicing dead rank {current_r} -> {current_r - 1}"
                    )
                    O_U, O_V = O_Us[i], O_Vs[i]
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

                    # Slice dead dimension
                    a.U_shared.data = a.U_shared.data[:, 1:]
                    a.V_shared.data = a.V_shared.data[1:, :]
                    a.liquid_core.core_pool.data = (
                        a.liquid_core.core_pool.data[:, 1:, 1:]
                    )
                    a.ema_U.data = a.ema_U.data[:, 1:]
                    a.ema_V.data = a.ema_V.data[1:, :]
                    if config.enable_paca:
                        a.ema_fast_U.data = a.ema_fast_U.data[:, 1:]
                        a.ema_slow_U.data = a.ema_slow_U.data[:, 1:]
                        a.ema_fast_V.data = a.ema_fast_V.data[1:, :]
                        a.ema_slow_V.data = a.ema_slow_V.data[1:, :]
                    if config.enable_coso_nullspace:
                        a.streaming_sketch_U.data = a.streaming_sketch_U.data[:, 1:]
                        a.ear_initialized = False
                        a.Q_null_U = torch.zeros(
                            a.out_features,
                            max(1, a.U_shared.shape[1] // 2),
                            device=a.U_shared.device,
                        )
                    # Rebuild gate_proj to match new rank dimension
                    if hasattr(a, 'gate_proj'):
                        new_r = a.U_shared.shape[1]
                        old_gate = a.gate_proj
                        a.gate_proj = nn.Linear(new_r * 2, 1, bias=True).to(
                            a.U_shared.device
                        )
                        nn.init.xavier_uniform_(a.gate_proj.weight)
                        a.gate_proj.bias.data.copy_(old_gate.bias.data)
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
