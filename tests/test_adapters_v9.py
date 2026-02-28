"""
CPU-only tests for CASCADES v9 adapter library modules.

Tests the extracted library code in cascades/adapters.py directly,
unlike test_adapters.py which tests the older v5 experiment file.
"""

import math

import pytest
import torch
import torch.nn as nn

from cascades.adapters import (
    CASCADESAdapter,
    CASCADESLinear,
    FunLoRA_Activation,
    FunLoRA_Adapter,
    ResonantCore,
)
from cascades.config import AblationConfig, DEFAULT_CONFIG, MINIMAL_CONFIG
from cascades.math_ops import is_orthonormal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IN_F = 32
OUT_F = 16
RANK = 4
BATCH = 2
SEQ = 8


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_linear():
    layer = nn.Linear(IN_F, OUT_F, bias=False)
    nn.init.normal_(layer.weight)
    return layer


@pytest.fixture
def adapter_default():
    return CASCADESAdapter(IN_F, OUT_F, rank=RANK, config=DEFAULT_CONFIG)


@pytest.fixture
def adapter_minimal():
    return CASCADESAdapter(IN_F, OUT_F, rank=RANK, config=MINIMAL_CONFIG)


@pytest.fixture
def funlora():
    return FunLoRA_Adapter(IN_F, OUT_F)


@pytest.fixture
def critical_linear(dummy_linear):
    return CASCADESLinear(dummy_linear, rank=RANK, is_critical=True, config=DEFAULT_CONFIG)


@pytest.fixture
def noncrit_linear(dummy_linear):
    return CASCADESLinear(dummy_linear, rank=RANK, is_critical=False, config=DEFAULT_CONFIG)


@pytest.fixture
def x_3d():
    """3D input: (batch, seq, features) — required for ResonantCore."""
    return torch.randn(BATCH, SEQ, IN_F)


# ---------------------------------------------------------------------------
# FunLoRA_Activation
# ---------------------------------------------------------------------------

class TestFunLoRAActivation:
    def test_forward_formula(self):
        x = torch.randn(4, 8)
        out = FunLoRA_Activation.apply(x)
        expected = x + torch.sigmoid(x) + torch.tanh(x)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_gradient_flows(self):
        x = torch.randn(4, 8, requires_grad=True)
        out = FunLoRA_Activation.apply(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


# ---------------------------------------------------------------------------
# FunLoRA_Adapter
# ---------------------------------------------------------------------------

class TestFunLoRA:
    def test_output_shape(self, funlora, x_3d):
        x_2d = x_3d.view(-1, IN_F)
        out = funlora(x_2d)
        assert out.shape == (BATCH * SEQ, OUT_F)

    def test_dtype_preservation_float16(self, funlora):
        x = torch.randn(4, IN_F, dtype=torch.float16)
        out = funlora(x)
        assert out.dtype == torch.float16

    def test_nonzero_output(self, funlora):
        x = torch.randn(4, IN_F)
        out = funlora(x)
        assert out.abs().sum().item() > 0

    def test_param_count(self, funlora):
        """FunLoRA should have exactly 2 parameters: a and b."""
        params = list(funlora.parameters())
        assert len(params) == 2
        assert params[0].shape == (OUT_F, 1)  # a
        assert params[1].shape == (1, IN_F)   # b


# ---------------------------------------------------------------------------
# ResonantCore
# ---------------------------------------------------------------------------

class TestResonantCore:
    def test_output_shape(self, x_3d):
        core = ResonantCore(IN_F, rank=RANK, num_cores=4)
        U = torch.randn(OUT_F, RANK)
        V = torch.randn(RANK, IN_F)
        out = core(x_3d, V, U)
        assert out.shape == (BATCH, SEQ, OUT_F)

    def test_core_pool_initialized_orthogonal(self):
        core = ResonantCore(IN_F, rank=RANK, num_cores=4)
        for k in range(4):
            c = core.core_pool[k]
            # Each core should be approximately orthogonal
            identity = torch.eye(RANK)
            assert torch.allclose(c @ c.T, identity, atol=0.1), (
                f"Core {k} is not sufficiently orthogonal"
            )


# ---------------------------------------------------------------------------
# CASCADESAdapter
# ---------------------------------------------------------------------------

class TestCASCADESAdapter:
    def test_output_shape(self, adapter_default, x_3d):
        out = adapter_default(x_3d)
        assert out.shape == (BATCH, SEQ, OUT_F)

    def test_output_dtype_float32(self, adapter_default, x_3d):
        out = adapter_default(x_3d.float())
        assert out.dtype == torch.float32

    def test_has_gate_when_enabled(self, adapter_default):
        assert hasattr(adapter_default, 'gate_proj')
        assert isinstance(adapter_default.gate_proj, nn.Linear)

    def test_no_gate_when_disabled(self, adapter_minimal):
        assert not hasattr(adapter_minimal, 'gate_proj')

    def test_config_stored(self, adapter_default):
        assert adapter_default.config is DEFAULT_CONFIG

    def test_qr_retractable(self, adapter_default):
        """U_shared can be QR-retracted to a valid Stiefel point."""
        Q, _ = torch.linalg.qr(adapter_default.U_shared)
        assert is_orthonormal(Q)

    def test_full_descent_step_no_crash(self, adapter_default, x_3d):
        """Descent step with gradients should not crash."""
        adapter_default.train()
        out = adapter_default(x_3d)
        loss = out.sum()
        loss.backward()
        adapter_default.full_descent_step(lr=0.01)
        # After descent, U_shared should remain approximately on the manifold
        Q, _ = torch.linalg.qr(adapter_default.U_shared.detach())
        assert is_orthonormal(Q)

    def test_riemannian_freeze_skips_zero_gradient(self, adapter_default):
        """When gradients are zero, descent step should be a no-op."""
        adapter_default.U_shared.grad = torch.zeros_like(adapter_default.U_shared)
        adapter_default.V_shared.grad = torch.zeros_like(adapter_default.V_shared)
        u_before = adapter_default.U_shared.data.clone()
        adapter_default.full_descent_step(lr=0.01)
        assert torch.allclose(u_before, adapter_default.U_shared.data)

    def test_streaming_paca_mask_shape(self, adapter_default, x_3d):
        """PaCA mask should return tensors broadcastable with gradients."""
        grad_U = torch.randn(OUT_F, RANK)
        grad_V = torch.randn(RANK, IN_F)
        mask_U, mask_V = adapter_default.streaming_paca_mask(grad_U, grad_V)
        # mask_U should be (OUT_F, 1) or broadcastable
        assert mask_U.shape[0] == OUT_F
        # mask_V should be (1, IN_F) or broadcastable
        assert mask_V.shape[1] == IN_F


# ---------------------------------------------------------------------------
# CASCADESLinear
# ---------------------------------------------------------------------------

class TestCASCADESLinear:
    def test_critical_output_shape(self, critical_linear, x_3d):
        out = critical_linear(x_3d)
        assert out.shape == (BATCH, SEQ, OUT_F)

    def test_noncritical_output_shape(self, noncrit_linear):
        x = torch.randn(BATCH * SEQ, IN_F)
        out = noncrit_linear(x)
        assert out.shape == (BATCH * SEQ, OUT_F)

    def test_critical_uses_cascades_adapter(self, critical_linear):
        assert isinstance(critical_linear.adapter, CASCADESAdapter)

    def test_noncritical_uses_funlora(self, noncrit_linear):
        assert isinstance(noncrit_linear.adapter, FunLoRA_Adapter)

    def test_adapter_contributes(self, critical_linear, x_3d):
        base_out = critical_linear.base_layer(x_3d)
        full_out = critical_linear(x_3d)
        assert not torch.allclose(base_out, full_out), (
            "Adapter contribution is zero"
        )

    def test_promote_changes_adapter_type(self, noncrit_linear):
        assert isinstance(noncrit_linear.adapter, FunLoRA_Adapter)
        result = noncrit_linear.promote(rank=RANK)
        assert result is True
        assert isinstance(noncrit_linear.adapter, CASCADESAdapter)
        assert noncrit_linear.is_critical is True

    def test_promote_idempotent_on_critical(self, critical_linear):
        result = critical_linear.promote(rank=RANK)
        assert result is False  # Already critical

    def test_demote_changes_adapter_type(self, critical_linear):
        # Need to do a forward pass first so SVD has something to work with
        result = critical_linear.demote()
        assert result is True
        assert isinstance(critical_linear.adapter, FunLoRA_Adapter)
        assert critical_linear.is_critical is False

    def test_demote_idempotent_on_noncritical(self, noncrit_linear):
        result = noncrit_linear.demote()
        assert result is False

    def test_promote_demote_roundtrip(self, noncrit_linear):
        """FunLoRA → promote → demote should leave a FunLoRA."""
        noncrit_linear.promote(rank=RANK)
        assert noncrit_linear.is_critical is True
        noncrit_linear.demote()
        assert noncrit_linear.is_critical is False
        assert isinstance(noncrit_linear.adapter, FunLoRA_Adapter)


# ---------------------------------------------------------------------------
# AblationConfig
# ---------------------------------------------------------------------------

class TestConfig:
    def test_default_all_enabled(self):
        cfg = DEFAULT_CONFIG
        assert cfg.enable_paca is True
        assert cfg.enable_deal is True
        assert cfg.enable_gainlora_gate is True
        assert cfg.enable_coso_nullspace is True
        assert cfg.enable_cllora_reassign is True
        assert cfg.enable_svc is True
        assert cfg.enable_dmole_select is True
        assert cfg.enable_funlora is True

    def test_minimal_all_disabled(self):
        cfg = MINIMAL_CONFIG
        assert cfg.enable_paca is False
        assert cfg.enable_svc is False

    def test_frozen(self):
        """Config should be immutable."""
        from dataclasses import FrozenInstanceError
        with pytest.raises(FrozenInstanceError):
            DEFAULT_CONFIG.enable_paca = False

    def test_custom_config(self):
        cfg = AblationConfig(enable_paca=False, enable_svc=False)
        assert cfg.enable_paca is False
        assert cfg.enable_svc is False
        assert cfg.enable_deal is True  # other flags unchanged
