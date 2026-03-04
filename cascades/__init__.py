"""CASCADES: Parameter-Efficient Continual Adaptation via Shared Dynamic Subspaces."""

__version__ = "2.0.0"

# Configuration (no torch dependency — always available)
from cascades.config import AblationConfig, DEFAULT_CONFIG, MINIMAL_CONFIG  # noqa: F401

# Heavy imports require torch; make them optional so lightweight tools
# (experiment_matrix, research_analyzer, dry-run mode) can import
# cascades.config without a full PyTorch installation.
try:
    # Core adapters (v10 architecture)
    from cascades.adapters import (  # noqa: F401
        CASCADESAdapter,
        CASCADESLinear,
        FunLoRA_Activation,
        FunLoRA_Adapter,
        ResonantCore,
    )

    # v10 math operations
    from cascades.math_ops import (  # noqa: F401
        gqa_precondition_gradient,
        soft_ear,
    )

    # Data loading
    from cascades.data import (  # noqa: F401
        NUM_TASKS,
        TASK_FILES,
        diagnose_per_example_loss,
        prepare_data,
    )

    # Evaluation
    from cascades.eval import (  # noqa: F401
        answers_match,
        build_inference_prompt,
        evaluate_accuracy,
        evaluate_generative,
        extract_answer_from_cot,
        normalize_answer,
    )

    # Injection and utilities
    from cascades.injection import (  # noqa: F401
        batched_autopoiesis_and_svc,
        batched_null_space_extraction,
        compute_layer_importance,
        estimate_quant_noise,
        inject_cascades,
    )

    # Sleep consolidation
    from cascades.sleep import SleepConsolidation, SleepConfig  # noqa: F401

    # VRAM monitoring
    from cascades.vram_monitor import (  # noqa: F401
        check_oom_risk,
        clear_cache,
        get_vram_stats,
        log_vram,
        reset_peak_stats,
    )

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
