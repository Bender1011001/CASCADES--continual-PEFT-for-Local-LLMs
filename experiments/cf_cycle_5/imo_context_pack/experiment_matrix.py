"""
CASCADES Research Loop — Experiment Matrix.

Defines all 24 experiments across 6 cycles for systematic validation
of CASCADES v10 components, ablation studies, and baseline comparisons.

Each experiment maps to an AblationConfig that gets passed to
train_cascades() or train_lora_baseline().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from cascades.config import AblationConfig


# ---------------------------------------------------------------------------
# Reusable config presets
# ---------------------------------------------------------------------------

# v9 baseline: all v9 components ON, all v10 advancements OFF
V9_CONFIG = AblationConfig(
    enable_paca=True,
    enable_deal=True,
    enable_gainlora_gate=True,
    enable_coso_nullspace=True,
    enable_cllora_reassign=True,
    enable_svc=True,
    enable_dmole_select=True,
    enable_funlora=True,
    # v10 flags OFF
    gqa_ratio=1.0,
    ear_gamma=1e-4,
    enable_soft_ear=False,
    enable_principal_expansion=False,
    cfg_lambda=1.0,
    enable_ambient_dedup=False,
)

# Full v10: everything ON (same as DEFAULT_CONFIG)
V10_CONFIG = AblationConfig()

# Plain LoRA: all CASCADES components disabled
LORA_CONFIG = AblationConfig(
    enable_paca=False,
    enable_deal=False,
    enable_gainlora_gate=False,
    enable_coso_nullspace=False,
    enable_cllora_reassign=False,
    enable_svc=False,
    enable_dmole_select=False,
    enable_funlora=False,
    enable_soft_ear=False,
    enable_principal_expansion=False,
    cfg_lambda=1.0,
    gqa_ratio=1.0,
    enable_ambient_dedup=False,
)

# Default 3-task file list
DEFAULT_TASK_FILES = [
    "data/task0_gsm8k_cot.jsonl",
    "data/task1_arc_cot.jsonl",
    "data/task2_csqa_cot.jsonl",
]

# Extended 5-task file list (for scaling experiment 5.1)
FIVE_TASK_FILES = [
    "data/task0_gsm8k_cot.jsonl",
    "data/task1_arc_cot.jsonl",
    "data/task2_csqa_cot.jsonl",
    "data/task0_logic_cot.jsonl",
    "data/task1_decomp_cot.jsonl",
]


# ---------------------------------------------------------------------------
# ExperimentConfig dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment in the research loop.

    Attributes:
        id: Unique experiment identifier (e.g. "1.1", "2.3").
        name: Human-readable experiment name.
        cycle: Cycle number (1-5).
        description: What this experiment tests.
        ablation_config: AblationConfig instance controlling CASCADES flags.
        training_overrides: Dict of kwargs passed to train_cascades() beyond
            the ablation config (e.g. rank, epochs, enable_sleep, task_files).
        eval_overrides: Dict of kwargs for generative evaluation overrides
            (e.g. max_new_tokens, do_sample, temperature, few_shot).
        is_baseline: True if this is a plain LoRA baseline (uses
            train_lora_baseline instead of train_cascades).
    """
    id: str
    name: str
    cycle: int
    description: str
    ablation_config: AblationConfig = field(default_factory=lambda: V9_CONFIG)
    training_overrides: dict = field(default_factory=dict)
    eval_overrides: dict = field(default_factory=dict)
    is_baseline: bool = False


# ---------------------------------------------------------------------------
# Full experiment matrix — 18 experiments across 5 cycles
# ---------------------------------------------------------------------------

EXPERIMENTS: list[ExperimentConfig] = [
    # ===================================================================
    # Cycle 1: Baselines
    # ===================================================================
    ExperimentConfig(
        id="1.1",
        name="Plain LoRA rank-8",
        cycle=1,
        description=(
            "Establish the catastrophic forgetting floor. Standard PEFT "
            "LoRA(r=8, alpha=16, dropout=0.05) with NO CASCADES components. "
            "Expected: ACC 25-40%, BWT -5% to -15%."
        ),
        ablation_config=LORA_CONFIG,
        training_overrides={"rank": 8, "epochs": 2},
        is_baseline=True,
    ),
    ExperimentConfig(
        id="1.2",
        name="CASCADES v9 reproduction",
        cycle=1,
        description=(
            "Confirm v9 results are reproducible on local GPU. "
            "DEFAULT_CONFIG with v10 flags explicitly disabled. "
            "Success: ACC within ±2% of 35.91%, BWT within ±1% of -1.46%."
        ),
        ablation_config=V9_CONFIG,
        training_overrides={"epochs": 2, "enable_sleep": True},
    ),

    # ===================================================================
    # Cycle 2: v10 Patch Validation (each adds ONE patch to v9 base)
    # ===================================================================
    ExperimentConfig(
        id="2.1",
        name="v9 + frozen null-space",
        cycle=2,
        description=(
            "Test BWT improvement from gradient projection out of prior-task "
            "subspace. enable_coso_nullspace=True with freeze_current_subspace() "
            "active at task boundaries. Expected: BWT ≥ +1% over v9."
        ),
        ablation_config=AblationConfig(
            enable_paca=True, enable_deal=True, enable_gainlora_gate=True,
            enable_coso_nullspace=True, enable_cllora_reassign=True,
            enable_svc=True, enable_dmole_select=True, enable_funlora=True,
            # v10: only frozen null-space active (via enable_coso_nullspace)
            gqa_ratio=1.0, ear_gamma=1e-4,
            enable_soft_ear=False, enable_principal_expansion=False,
            cfg_lambda=1.0, enable_ambient_dedup=False,
        ),
        training_overrides={"epochs": 2, "enable_sleep": True},
    ),
    ExperimentConfig(
        id="2.2",
        name="v9 + soft-EAR",
        cycle=2,
        description=(
            "Test Tikhonov-regularized EAR vs hard 1% cutoff. "
            "enable_soft_ear=True, ear_gamma=1e-4. "
            "Expected: smoother loss curves, ≤ +1% ACC improvement."
        ),
        ablation_config=AblationConfig(
            enable_paca=True, enable_deal=True, enable_gainlora_gate=True,
            enable_coso_nullspace=True, enable_cllora_reassign=True,
            enable_svc=True, enable_dmole_select=True, enable_funlora=True,
            gqa_ratio=1.0, ear_gamma=1e-4,
            enable_soft_ear=True,  # v10 patch
            enable_principal_expansion=False, cfg_lambda=1.0,
            enable_ambient_dedup=False,
        ),
        training_overrides={"epochs": 2, "enable_sleep": True},
    ),
    ExperimentConfig(
        id="2.3",
        name="v9 + GQA preconditioning",
        cycle=2,
        description=(
            "Confirm no-op on 4B model. Qwen3-4B has 8 KV groups "
            "(gqa_ratio=4.0). Expected: Δ ACC ≈ 0%."
        ),
        ablation_config=AblationConfig(
            enable_paca=True, enable_deal=True, enable_gainlora_gate=True,
            enable_coso_nullspace=True, enable_cllora_reassign=True,
            enable_svc=True, enable_dmole_select=True, enable_funlora=True,
            gqa_ratio=4.0,  # v10 patch — auto-detected from model
            ear_gamma=1e-4,
            enable_soft_ear=False, enable_principal_expansion=False,
            cfg_lambda=1.0, enable_ambient_dedup=False,
        ),
        training_overrides={"epochs": 2, "enable_sleep": True},
    ),
    ExperimentConfig(
        id="2.4",
        name="v9 + principal expansion",
        cycle=2,
        description=(
            "Test power-iteration-based rank expansion vs stochastic "
            "mini-batch init. Expected: more stable expansion, ≤ +1% ACC."
        ),
        ablation_config=AblationConfig(
            enable_paca=True, enable_deal=True, enable_gainlora_gate=True,
            enable_coso_nullspace=True, enable_cllora_reassign=True,
            enable_svc=True, enable_dmole_select=True, enable_funlora=True,
            gqa_ratio=1.0, ear_gamma=1e-4,
            enable_soft_ear=False,
            enable_principal_expansion=True,  # v10 patch
            cfg_lambda=1.0, enable_ambient_dedup=False,
        ),
        training_overrides={"epochs": 2, "enable_sleep": True},
    ),
    ExperimentConfig(
        id="2.5",
        name="v9 + CFG decoding",
        cycle=2,
        description=(
            "Test classifier-free guidance boost at eval time. "
            "cfg_lambda=1.5 applied only during generative evaluation. "
            "Expected: containment ↑ 5-15%, proxy ACC unchanged."
        ),
        ablation_config=AblationConfig(
            enable_paca=True, enable_deal=True, enable_gainlora_gate=True,
            enable_coso_nullspace=True, enable_cllora_reassign=True,
            enable_svc=True, enable_dmole_select=True, enable_funlora=True,
            gqa_ratio=1.0, ear_gamma=1e-4,
            enable_soft_ear=False, enable_principal_expansion=False,
            cfg_lambda=1.5,  # v10 patch — eval-time only
            enable_ambient_dedup=False,
        ),
        training_overrides={"epochs": 2, "enable_sleep": True},
    ),
    ExperimentConfig(
        id="2.6",
        name="v9 + ambient trace dedup",
        cycle=2,
        description=(
            "Test ambient-space cross-adapter deduplication during sleep. "
            "enable_ambient_dedup=True. Compares W = U Λ V^T across layers. "
            "Expected: fewer redundant adapters, slight BWT gain."
        ),
        ablation_config=AblationConfig(
            enable_paca=True, enable_deal=True, enable_gainlora_gate=True,
            enable_coso_nullspace=True, enable_cllora_reassign=True,
            enable_svc=True, enable_dmole_select=True, enable_funlora=True,
            gqa_ratio=1.0, ear_gamma=1e-4,
            enable_soft_ear=False, enable_principal_expansion=False,
            cfg_lambda=1.0,
            enable_ambient_dedup=True,  # v10 patch
        ),
        training_overrides={"epochs": 2, "enable_sleep": True},
    ),
    ExperimentConfig(
        id="2.7",
        name="Full v10 — all patches",
        cycle=2,
        description=(
            "Measure cumulative effect of all 5 v10 patches. "
            "All v10 flags enabled. Expected: ≥ 37% ACC, ≥ +2% BWT."
        ),
        ablation_config=V10_CONFIG,
        training_overrides={"epochs": 2, "enable_sleep": True},
    ),

    # ===================================================================
    # Cycle 3: Component Ablations (each removes ONE component from v9)
    # ===================================================================
    ExperimentConfig(
        id="3.1",
        name="v9 minus EAR",
        cycle=3,
        description=(
            "Remove gradient redirection (EAR/null-space). "
            "enable_coso_nullspace=False, enable_cllora_reassign=False. "
            "Expected: BWT regression — EAR is primary anti-forgetting."
        ),
        ablation_config=AblationConfig(
            enable_paca=True, enable_deal=True, enable_gainlora_gate=True,
            enable_coso_nullspace=False,   # ABLATED
            enable_cllora_reassign=False,  # ABLATED
            enable_svc=True, enable_dmole_select=True, enable_funlora=True,
            gqa_ratio=1.0, ear_gamma=1e-4,
            enable_soft_ear=False, enable_principal_expansion=False,
            cfg_lambda=1.0, enable_ambient_dedup=False,
        ),
        training_overrides={"epochs": 2, "enable_sleep": True},
    ),
    ExperimentConfig(
        id="3.2",
        name="v9 minus Sleep",
        cycle=3,
        description=(
            "Remove bio-inspired sleep consolidation. All flags on, "
            "enable_sleep=False. Expected: ACC stable, possible BWT regression."
        ),
        ablation_config=V9_CONFIG,
        training_overrides={"epochs": 2, "enable_sleep": False},  # ABLATED
    ),
    ExperimentConfig(
        id="3.3",
        name="v9 minus PaCA",
        cycle=3,
        description=(
            "Remove causal conflict detection (PaCA). enable_paca=False. "
            "Expected: ACC ↓ 2-5% — PaCA gates information flow."
        ),
        ablation_config=AblationConfig(
            enable_paca=False,  # ABLATED
            enable_deal=True, enable_gainlora_gate=True,
            enable_coso_nullspace=True, enable_cllora_reassign=True,
            enable_svc=True, enable_dmole_select=True, enable_funlora=True,
            gqa_ratio=1.0, ear_gamma=1e-4,
            enable_soft_ear=False, enable_principal_expansion=False,
            cfg_lambda=1.0, enable_ambient_dedup=False,
        ),
        training_overrides={"epochs": 2, "enable_sleep": True},
    ),
    ExperimentConfig(
        id="3.4",
        name="v9 minus Breathing",
        cycle=3,
        description=(
            "Remove autopoietic rank dynamics (Breathing Manifolds / SVC). "
            "enable_svc=False. Expected: minor ACC ↓."
        ),
        ablation_config=AblationConfig(
            enable_paca=True, enable_deal=True, enable_gainlora_gate=True,
            enable_coso_nullspace=True, enable_cllora_reassign=True,
            enable_svc=False,  # ABLATED
            enable_dmole_select=True, enable_funlora=True,
            gqa_ratio=1.0, ear_gamma=1e-4,
            enable_soft_ear=False, enable_principal_expansion=False,
            cfg_lambda=1.0, enable_ambient_dedup=False,
        ),
        training_overrides={"epochs": 2, "enable_sleep": True},
    ),
    ExperimentConfig(
        id="3.5",
        name="v9 minus DEAL",
        cycle=3,
        description=(
            "Remove quantization noise filter (DEAL). enable_deal=False. "
            "Expected: training instability on NF4."
        ),
        ablation_config=AblationConfig(
            enable_paca=True,
            enable_deal=False,  # ABLATED
            enable_gainlora_gate=True,
            enable_coso_nullspace=True, enable_cllora_reassign=True,
            enable_svc=True, enable_dmole_select=True, enable_funlora=True,
            gqa_ratio=1.0, ear_gamma=1e-4,
            enable_soft_ear=False, enable_principal_expansion=False,
            cfg_lambda=1.0, enable_ambient_dedup=False,
        ),
        training_overrides={"epochs": 2, "enable_sleep": True},
    ),

    # ===================================================================
    # Cycle 4: Generative Gap Investigation (v9 training, eval overrides)
    # ===================================================================
    ExperimentConfig(
        id="4.1",
        name="Few-shot eval prompts",
        cycle=4,
        description=(
            "Inject 2 task-specific examples into the eval prompt. "
            "Tests whether format guidance improves containment. "
            "Expected: containment ↑ to ~70%, possible EM > 0%."
        ),
        ablation_config=V9_CONFIG,
        training_overrides={"epochs": 2, "enable_sleep": True},
        eval_overrides={"few_shot": 2},
    ),
    ExperimentConfig(
        id="4.2",
        name="Longer generation",
        cycle=4,
        description=(
            "Double max_new_tokens to 1024. Tests whether truncation "
            "causes the EM gap. Expected: more complete answers."
        ),
        ablation_config=V9_CONFIG,
        training_overrides={"epochs": 2, "enable_sleep": True},
        eval_overrides={"max_new_tokens": 1024},
    ),
    ExperimentConfig(
        id="4.3",
        name="Greedy decoding",
        cycle=4,
        description=(
            "Use do_sample=False, temperature=0. Tests whether sampling "
            "randomness causes format inconsistency. "
            "Expected: more consistent output format."
        ),
        ablation_config=V9_CONFIG,
        training_overrides={"epochs": 2, "enable_sleep": True},
        eval_overrides={"do_sample": False, "temperature": 0.0},
    ),

    # ===================================================================
    # Cycle 5: Scaling (optional)
    # ===================================================================
    ExperimentConfig(
        id="5.1",
        name="5-task run",
        cycle=5,
        description=(
            "Extend to 5 tasks to test null-space saturation. "
            "Add logic_cot + decomp_cot as tasks 3-4. "
            "Expected: BWT degrades slightly, null-space ~60-80% capacity."
        ),
        ablation_config=V9_CONFIG,
        training_overrides={
            "epochs": 2,
            "enable_sleep": True,
            "task_files": FIVE_TASK_FILES,
        },
    ),
    ExperimentConfig(
        id="5.2",
        name="Rank sensitivity",
        cycle=5,
        description=(
            "Three sub-runs with rank=4, rank=8, rank=16. "
            "Tests ACC/BWT/VRAM tradeoff across ranks. "
            "Expected: rank-8 is sweet spot; rank-16 may OOM."
        ),
        ablation_config=V9_CONFIG,
        training_overrides={
            "epochs": 2,
            "enable_sleep": True,
            "rank_sweep": [4, 8, 16],  # Runner handles sub-runs
        },
    ),
    # ===================================================================
    # Cycle 6: v10 Leave-One-Out Ablation
    #   (each disables ONE v10 patch from the full v10 stack)
    # ===================================================================
    ExperimentConfig(
        id="6.1",
        name="v10 minus GQA scaling",
        cycle=6,
        description=(
            "Leave-one-out: disable GQA preconditioning from full v10. "
            "gqa_ratio=1.0. Tests marginal contribution of GQA fix."
        ),
        ablation_config=AblationConfig(
            gqa_ratio=1.0,  # ABLATED (v10 default is 4.0 for Qwen3-4B)
            enable_soft_ear=True, enable_principal_expansion=True,
            cfg_lambda=1.5, enable_ambient_dedup=True,
        ),
        training_overrides={"epochs": 2, "enable_sleep": True},
    ),
    ExperimentConfig(
        id="6.2",
        name="v10 minus soft-EAR",
        cycle=6,
        description=(
            "Leave-one-out: disable Tikhonov soft-EAR from full v10. "
            "enable_soft_ear=False. Falls back to hard 1%% cutoff. "
            "Tests marginal contribution of smooth EAR regularization."
        ),
        ablation_config=AblationConfig(
            gqa_ratio=4.0,
            enable_soft_ear=False,  # ABLATED
            enable_principal_expansion=True,
            cfg_lambda=1.5, enable_ambient_dedup=True,
        ),
        training_overrides={"epochs": 2, "enable_sleep": True},
    ),
    ExperimentConfig(
        id="6.3",
        name="v10 minus principal expansion",
        cycle=6,
        description=(
            "Leave-one-out: disable power-iteration rank expansion from "
            "full v10. enable_principal_expansion=False. Falls back to "
            "stochastic mini-batch init. Tests marginal contribution."
        ),
        ablation_config=AblationConfig(
            gqa_ratio=4.0, enable_soft_ear=True,
            enable_principal_expansion=False,  # ABLATED
            cfg_lambda=1.5, enable_ambient_dedup=True,
        ),
        training_overrides={"epochs": 2, "enable_sleep": True},
    ),
    ExperimentConfig(
        id="6.4",
        name="v10 minus ambient dedup",
        cycle=6,
        description=(
            "Leave-one-out: disable ambient trace deduplication from "
            "full v10. enable_ambient_dedup=False. Sleep still runs "
            "but skips cross-adapter dedup. Tests marginal contribution."
        ),
        ablation_config=AblationConfig(
            gqa_ratio=4.0, enable_soft_ear=True,
            enable_principal_expansion=True, cfg_lambda=1.5,
            enable_ambient_dedup=False,  # ABLATED
        ),
        training_overrides={"epochs": 2, "enable_sleep": True},
    ),
    ExperimentConfig(
        id="6.5",
        name="v10 minus CFG decoding",
        cycle=6,
        description=(
            "Leave-one-out: disable classifier-free guidance from full v10. "
            "cfg_lambda=1.0. Tests marginal contribution of CFG boost. "
            "Expected: containment drops but proxy ACC unchanged."
        ),
        ablation_config=AblationConfig(
            gqa_ratio=4.0, enable_soft_ear=True,
            enable_principal_expansion=True,
            cfg_lambda=1.0,  # ABLATED
            enable_ambient_dedup=True,
        ),
        training_overrides={"epochs": 2, "enable_sleep": True},
    ),
]


# ---------------------------------------------------------------------------
# Accessor functions
# ---------------------------------------------------------------------------

def get_all_experiments() -> list[ExperimentConfig]:
    """Return the complete experiment list."""
    return list(EXPERIMENTS)


def get_cycle(n: int) -> list[ExperimentConfig]:
    """Return all experiments belonging to cycle *n* (1-6)."""
    return [e for e in EXPERIMENTS if e.cycle == n]


def get_experiment(exp_id: str) -> Optional[ExperimentConfig]:
    """Look up a single experiment by its ID string (e.g. '2.3').

    Returns None if not found.
    """
    for e in EXPERIMENTS:
        if e.id == exp_id:
            return e
    return None


# ---------------------------------------------------------------------------
# CLI quick-view
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"CASCADES Research Loop — {len(EXPERIMENTS)} experiments\n")
    for cycle_num in range(1, 7):
        cycle_exps = get_cycle(cycle_num)
        cycle_names = {
            1: "Baselines", 2: "v10 Patches", 3: "Ablations",
            4: "Generative Gap", 5: "Scaling", 6: "v10 Leave-One-Out",
        }
        if not cycle_exps:
            continue
        print(f"Cycle {cycle_num}: {cycle_names.get(cycle_num, 'Unknown')} ({len(cycle_exps)} experiments)")
        for exp in cycle_exps:
            baseline_tag = " [BASELINE]" if exp.is_baseline else ""
            print(f"  {exp.id:>4s}  {exp.name:<30s}{baseline_tag}")
        print()
