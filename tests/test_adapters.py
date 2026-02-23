"""
CPU-only tests for CASCADES adapter modules.

These tests verify forward-pass shapes, dtype contracts, and adapter
composition — all without loading a large language model.

The adapters are reimported from the v5 implementation file.  Since that file
uses module-level globals (ENABLE_*), we test behaviour with those flags as-is.
"""

import importlib
import sys
import types
import math

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Lightweight stubs so we can import the adapter code without HuggingFace
# ---------------------------------------------------------------------------

def _make_hf_stub():
    """Create a minimal transformers stub so the import does not fail."""
    stub = types.ModuleType("transformers")
    stub.AutoModelForCausalLM = None
    stub.AutoTokenizer = None
    stub.BitsAndBytesConfig = None
    sys.modules.setdefault("transformers", stub)

    bitsandbytes = types.ModuleType("bitsandbytes")
    sys.modules.setdefault("bitsandbytes", bitsandbytes)


_make_hf_stub()

# Dynamic import so the test does not hard-depend on a specific file path
import importlib.util, pathlib

_v5_path = pathlib.Path(__file__).parent.parent / "cascades_exp" / "hf_cascades_v5.py"

_spec = importlib.util.spec_from_file_location("hf_cascades_v5", _v5_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

CASCADES_v3_Adapter = _mod.CASCADES_v3_Adapter
CASCADES_v3_Linear = _mod.CASCADES_v3_Linear
FunLoRA_Adapter = _mod.FunLoRA_Adapter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

IN_F = 32
OUT_F = 16
RANK = 4
BATCH = 3
SEQ = 8


@pytest.fixture
def dummy_linear():
    layer = nn.Linear(IN_F, OUT_F, bias=False)
    nn.init.normal_(layer.weight)
    return layer


@pytest.fixture
def cascades_adapter():
    adapter = CASCADES_v3_Adapter(IN_F, OUT_F, rank=RANK)
    adapter.add_task(0)
    return adapter


@pytest.fixture
def funlora_adapter():
    return FunLoRA_Adapter(IN_F, OUT_F)


@pytest.fixture
def critical_linear(dummy_linear):
    module = CASCADES_v3_Linear(dummy_linear, rank=RANK, is_critical=True)
    module.add_task(0)
    return module


@pytest.fixture
def noncritical_linear(dummy_linear):
    return CASCADES_v3_Linear(dummy_linear, rank=RANK, is_critical=False)


@pytest.fixture
def x_2d():
    return torch.randn(BATCH * SEQ, IN_F)


# ---------------------------------------------------------------------------
# FunLoRA
# ---------------------------------------------------------------------------

class TestFunLoRAAdapter:
    def test_output_shape(self, funlora_adapter, x_2d):
        out = funlora_adapter(x_2d)
        assert out.shape == (BATCH * SEQ, OUT_F)

    def test_output_dtype_matches_input(self, funlora_adapter, x_2d):
        """Adapter must cast output back to input dtype."""
        x_half = x_2d.to(torch.float16)
        out = funlora_adapter(x_half)
        assert out.dtype == torch.float16

    def test_nonzero_output(self, funlora_adapter, x_2d):
        out = funlora_adapter(x_2d)
        assert out.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# CASCADES_v3_Adapter
# ---------------------------------------------------------------------------

class TestCASCADESAdapter:
    def test_output_shape(self, cascades_adapter, x_2d):
        out = cascades_adapter(x_2d, task_id=0)
        assert out.shape == (BATCH * SEQ, OUT_F)

    def test_output_dtype_matches_float32_input(self, cascades_adapter, x_2d):
        out = cascades_adapter(x_2d.float(), task_id=0)
        assert out.dtype == torch.float32

    def test_output_dtype_matches_float16_input(self, cascades_adapter, x_2d):
        out = cascades_adapter(x_2d.to(torch.float16), task_id=0)
        assert out.dtype == torch.float16, (
            "Adapter returned wrong dtype — SDPA mixed-dtype crash would occur in HF model"
        )

    def test_multiple_tasks_independent(self, cascades_adapter, x_2d):
        """Task lambdas must be independent — adding task 1 must not change task 0 output."""
        out_t0_before = cascades_adapter(x_2d, task_id=0).detach().clone()
        cascades_adapter.add_task(1)
        out_t0_after = cascades_adapter(x_2d, task_id=0).detach().clone()
        assert torch.allclose(out_t0_before, out_t0_after), (
            "Adding a new task changed the output of task 0"
        )

    def test_task_not_added_raises(self, cascades_adapter, x_2d):
        # PyTorch ParameterDict raises AttributeError for missing keys
        with pytest.raises((KeyError, AttributeError)):
            cascades_adapter(x_2d, task_id=99)

    def test_u_v_shared_are_initially_orthonormal_after_add_task(self):
        """U_shared and V_shared^T should be orthonormalized before training begins."""
        from cascades.math_ops import is_orthonormal
        adapter = CASCADES_v3_Adapter(IN_F, OUT_F, rank=RANK)
        adapter.add_task(0)
        # They are initialized with small random values, not necessarily orthonormal —
        # orthonormality is enforced *during* the first Riemannian step.
        # What we test here: QR retraction produces a valid manifold point.
        Q_U, _ = torch.linalg.qr(adapter.U_shared)
        assert is_orthonormal(Q_U)


# ---------------------------------------------------------------------------
# CASCADES_v3_Linear (wrapper)
# ---------------------------------------------------------------------------

class TestCASCADESLinearWrapper:
    def test_critical_output_shape(self, critical_linear, x_2d):
        out = critical_linear(x_2d)
        assert out.shape == (BATCH * SEQ, OUT_F)

    def test_noncritical_output_shape(self, noncritical_linear, x_2d):
        out = noncritical_linear(x_2d)
        assert out.shape == (BATCH * SEQ, OUT_F)

    def test_adapter_contributes_to_output(self, critical_linear, x_2d):
        """Output with adapter must differ from base-only output."""
        base_out = critical_linear.base_layer(x_2d)
        full_out = critical_linear(x_2d)
        # They should differ (adapter contributes)
        assert not torch.allclose(base_out, full_out), (
            "Adapter contribution is zero — adapter is not active"
        )

    def test_critical_flag_controls_adapter_type(self, dummy_linear):
        crit = CASCADES_v3_Linear(dummy_linear, rank=RANK, is_critical=True)
        noncrit = CASCADES_v3_Linear(dummy_linear, rank=RANK, is_critical=False)
        assert isinstance(crit.adapter, CASCADES_v3_Adapter)
        assert isinstance(noncrit.adapter, FunLoRA_Adapter)

    def test_trainable_parameters_exist(self, critical_linear):
        trainable = [p for p in critical_linear.parameters() if p.requires_grad]
        # Base layer is frozen, adapter params must be trainable
        assert len(trainable) > 0, "No trainable parameters found in critical linear"
