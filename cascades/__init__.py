"""CASCADES: Parameter-Efficient Continual Adaptation via Shared Dynamic Subspaces."""

__version__ = "1.0.0"

# Configuration
from cascades.config import AblationConfig, DEFAULT_CONFIG, MINIMAL_CONFIG  # noqa: F401

# Core adapters (v9 architecture)
from cascades.adapters import (  # noqa: F401
    CASCADESAdapter,
    CASCADESLinear,
    FunLoRA_Activation,
    FunLoRA_Adapter,
    ResonantCore,
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
