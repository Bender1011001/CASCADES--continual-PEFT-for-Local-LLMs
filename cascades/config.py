"""
CASCADES ablation configuration — replaces global ENABLE_* flags.

Using a frozen dataclass instead of module-level globals enables:
  - Clean dependency injection into adapter constructors
  - Per-experiment overrides without monkeypatching
  - Reproducible ablation studies with serializable configs
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AblationConfig:
    """Immutable configuration for CASCADES ablation flags.

    All flags default to True (full system enabled).
    Set individual flags to False to ablate specific components.
    """

    # Intersection B: CaLoRA causal attribution
    enable_paca: bool = True
    # Intersection B: DEAL heat kernel filter (quantization-aware)
    enable_deal: bool = True
    # Intersection B: GainLoRA learned interference gate
    enable_gainlora_gate: bool = True
    # Intersection C: CoSO Frequent Directions (streaming EAR)
    enable_coso_nullspace: bool = True
    # Intersection C: CL-LoRA gradient reassignment
    enable_cllora_reassign: bool = True
    # Intersection C: Singular Value Calibration
    enable_svc: bool = True
    # Intersection D: D-MoLE layer importance selection
    enable_dmole_select: bool = True
    # Intersection D: FunLoRA rank-1 for non-critical layers
    enable_funlora: bool = True

    # --- v10 advancements ---
    # GQA-aware metric preconditioning: H_q / H_kv ratio (auto-detected)
    gqa_ratio: float = 1.0
    # Tikhonov damping for smooth EAR (prevents noise amplification)
    ear_gamma: float = 1e-4
    # Use smooth Tikhonov EAR instead of hard 1% cutoff
    enable_soft_ear: bool = True
    # Use power-iteration on EAR sketch for rank expansion (vs stochastic)
    enable_principal_expansion: bool = True
    # Adapter-level CFG lambda for decoding (1.0 = no boost)
    cfg_lambda: float = 1.5
    # Ambient trace dedup: compare W = U Λ V^T in ambient space during sleep
    # (wires through to SleepConfig.enable_cross_adapter_dedup)
    enable_ambient_dedup: bool = True


# Convenience instance with all components enabled (production default)
DEFAULT_CONFIG = AblationConfig()

# Common ablation presets
MINIMAL_CONFIG = AblationConfig(
    enable_paca=False,
    enable_deal=False,
    enable_gainlora_gate=False,
    enable_coso_nullspace=False,
    enable_cllora_reassign=False,
    enable_svc=False,
    enable_dmole_select=False,
    enable_funlora=False,
)
