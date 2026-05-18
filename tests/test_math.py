"""
Unit tests for CASCADES core mathematical operations.

Each test maps to a specific claim in the paper:
  - test_qr_retraction_orthonormality   → §3.1 "QR retraction maintains St(n,r)"
  - test_riemannian_gradient_tangent    → §3.1 "G_R is in the tangent space"
  - test_stella_step_stays_on_manifold  → §3.1 end-to-end Riemannian step
  - test_ear_norm_preservation          → §3.3 Proposition 1
  - test_ear_grad_in_free_subspace      → §3.3 "g_EAR ⊥ occupied directions"
  - test_ear_zero_free_component        → §3.3 edge case handling
  - test_svc_shrinks_singular_values    → §3.1 "SVC prevents spectral accumulation"
  - test_svc_output_shape               → sanity check
  - test_deal_noise_floor_gate          → §3.4 Stage 1
  - test_deal_decay_applied             → §3.4 Stage 2
  - test_funlora_effective_rank         → §Appendix "FunLoRA effective rank ≥ 3"
"""

import math
import torch
import pytest

from cascades.math_ops import (
    riemannian_gradient,
    qr_retraction,
    stella_riemannian_step,
    is_orthonormal,
    energy_accounted_reassignment,
    deal_heat_kernel_filter,
    svc_calibration,
    SBA_FORMULA_VARIANT,
    sba_reference_attention,
    sba_optimized_attention,
    sba_value_attention,
    sba_attention_diagnostics,
    sba_jacobian_frobenius_norm,
    softmax_jacobian_frobenius_norm,
)


# ---------------------------------------------------------------------------
# Stiefel / Riemannian geometry
# ---------------------------------------------------------------------------

class TestQRRetraction:
    def test_output_is_orthonormal(self, small_matrix):
        Q = qr_retraction(small_matrix)
        assert is_orthonormal(Q, tol=1e-5), "QR retraction did not produce orthonormal columns"

    def test_output_shape_preserved(self, small_matrix):
        Q = qr_retraction(small_matrix)
        assert Q.shape == small_matrix.shape

    def test_already_orthonormal_is_stable(self, orthonormal_matrix):
        Q2 = qr_retraction(orthonormal_matrix)
        assert is_orthonormal(Q2, tol=1e-5)


class TestRiemannianGradient:
    def test_tangent_space_condition(self, orthonormal_matrix):
        """G_R must satisfy: U^T G_R + G_R^T U = 0  (skew-symmetry of U^T G_R)."""
        grad = torch.randn_like(orthonormal_matrix)
        g_r = riemannian_gradient(orthonormal_matrix, grad)
        skew = orthonormal_matrix.T @ g_r
        # Skew-symmetric: A + A^T = 0
        error = (skew + skew.T).norm().item()
        assert error < 1e-5, f"Riemannian gradient not in tangent space: error={error:.2e}"

    def test_output_shape(self, orthonormal_matrix):
        grad = torch.randn_like(orthonormal_matrix)
        g_r = riemannian_gradient(orthonormal_matrix, grad)
        assert g_r.shape == orthonormal_matrix.shape


class TestStellaStep:
    def test_stays_on_manifold(self, orthonormal_matrix):
        """After one full Riemannian step, parameter must remain on St(n, r)."""
        grad = torch.randn_like(orthonormal_matrix)
        param = orthonormal_matrix.clone()
        stella_riemannian_step(param, grad, lr=0.01)
        assert is_orthonormal(param, tol=1e-5), "Parameter left Stiefel manifold after step"

    def test_modifies_in_place(self, orthonormal_matrix):
        grad = torch.randn_like(orthonormal_matrix)
        original_data_ptr = orthonormal_matrix.data_ptr()
        stella_riemannian_step(orthonormal_matrix, grad, lr=0.01)
        assert orthonormal_matrix.data_ptr() == original_data_ptr, "Step should be in-place"

    def test_multiple_steps_stay_on_manifold(self, orthonormal_matrix):
        param = orthonormal_matrix.clone()
        for _ in range(20):
            grad = torch.randn_like(param)
            stella_riemannian_step(param, grad, lr=0.005)
        assert is_orthonormal(param, tol=1e-4), "Manifold drift after 20 steps"


# ---------------------------------------------------------------------------
# Energy-Accounted Reassignment (EAR) — §3.3 Proposition 1
# ---------------------------------------------------------------------------

class TestEAR:
    def test_norm_preservation_mixed_gradient(self, occupied_basis, gradient_mixed):
        """‖g_EAR‖₂ = ‖g‖₂ when gradient has a free component."""
        g_ear = energy_accounted_reassignment(gradient_mixed, occupied_basis)
        original_norm = gradient_mixed.norm().item()
        ear_norm = g_ear.norm().item()
        assert abs(original_norm - ear_norm) < 1e-4, (
            f"EAR norm not preserved: original={original_norm:.4f}, EAR={ear_norm:.4f}"
        )

    def test_result_in_free_subspace(self, occupied_basis, gradient_mixed):
        """g_EAR must be orthogonal to all columns of the occupied basis."""
        g_ear = energy_accounted_reassignment(gradient_mixed, occupied_basis)
        Q, _ = torch.linalg.qr(occupied_basis)
        # Project g_EAR onto occupied subspace — should be near zero
        occupied_component = Q @ (Q.T @ g_ear)
        residual_norm = occupied_component.norm().item()
        assert residual_norm < 1e-4, (
            f"g_EAR has occupied component: {residual_norm:.2e}"
        )

    def test_zero_free_component_edge_case(self, occupied_basis, gradient_in_occupied):
        """When gradient is entirely in occupied space, return zeros (no energy to reassign)."""
        g_ear = energy_accounted_reassignment(gradient_in_occupied, occupied_basis)
        # The free component is ~0, so output should also be ~0
        assert g_ear.norm().item() < 1e-3, (
            f"Expected ~0 for fully-occupied gradient, got ‖g_EAR‖={g_ear.norm().item():.4f}"
        )

    def test_output_shape(self, occupied_basis, gradient_mixed):
        g_ear = energy_accounted_reassignment(gradient_mixed, occupied_basis)
        assert g_ear.shape == gradient_mixed.shape


# ---------------------------------------------------------------------------
# Quantization-aware DEAL heat-kernel filter — §3.4
# ---------------------------------------------------------------------------

class TestDEALFilter:
    def test_noise_floor_gate_zeros_small_grad(self):
        """Gradients smaller than ε_quant must be zeroed out (Stage 1)."""
        tiny_grad = torch.randn(4, 4) * 1e-6  # much smaller than any ε_quant
        out = deal_heat_kernel_filter(tiny_grad, quant_noise_std=1.0)
        assert out.norm().item() == 0.0, "Stage-1 gate did not zero out noise-level gradient"

    def test_valid_grad_passes_through_scaled(self):
        """Gradients above ε_quant should be attenuated by exp(-λt), not zeroed."""
        grad = torch.ones(4, 4)  # norm >> any ε_quant
        out = deal_heat_kernel_filter(grad, quant_noise_std=0.0, t=0.05, lambda_decay=0.01)
        expected_scale = math.exp(-0.01 * 0.05)
        assert out.norm().item() > 0.0, "Valid gradient was incorrectly zeroed"
        assert abs(out.mean().item() - expected_scale) < 1e-5

    def test_1d_input_returned_unchanged(self):
        """Filter is only applied to 2-D tensors (per paper spec)."""
        grad_1d = torch.randn(8)
        out = deal_heat_kernel_filter(grad_1d)
        assert torch.allclose(out, grad_1d)

    def test_output_shape(self):
        grad = torch.randn(8, 16)
        out = deal_heat_kernel_filter(grad)
        assert out.shape == grad.shape


# ---------------------------------------------------------------------------
# SVC — Singular Value Calibration — §3.1
# ---------------------------------------------------------------------------

class TestSVC:
    def test_shrinks_singular_values(self, square_matrix):
        """After SVC, every singular value should be ≤ the original value."""
        _, S_before, _ = torch.linalg.svd(square_matrix, full_matrices=False)
        calibrated = svc_calibration(square_matrix, svc_lambda=0.01)
        _, S_after, _ = torch.linalg.svd(calibrated, full_matrices=False)
        assert (S_after <= S_before + 1e-5).all(), "SVC did not shrink singular values"

    def test_output_shape(self, square_matrix):
        out = svc_calibration(square_matrix)
        assert out.shape == square_matrix.shape

    def test_identity_input_stays_small(self):
        """SVC on identity: S_i = 1 → S_i_cal = 1/(1+λ) < 1."""
        I = torch.eye(4)
        out = svc_calibration(I, svc_lambda=0.5)
        _, S, _ = torch.linalg.svd(out, full_matrices=False)
        expected = 1.0 / (1.0 + 0.5)
        assert (S - expected).abs().max().item() < 1e-5

    def test_large_lambda_suppresses_singular_values(self, square_matrix):
        """Very large λ should bring singular values close to 0."""
        out = svc_calibration(square_matrix, svc_lambda=1e6)
        _, S, _ = torch.linalg.svd(out, full_matrices=False)
        assert S.max().item() < 1e-3


# ---------------------------------------------------------------------------
# FunLoRA effective rank — §Appendix
# ---------------------------------------------------------------------------

class TestFunLoRAEffectiveRank:
    """Verify that the rank-1 base + nonlinear expansion yields effective rank ≥ 3."""

    def _funlora_forward(self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ab = x @ b.T @ a.T  # rank-1 base
        return ab + torch.sigmoid(ab) + torch.tanh(ab)

    def _numerical_rank(self, J: torch.Tensor, tol: float = 1e-2) -> int:
        """Count singular values above *tol* in the Jacobian."""
        _, S, _ = torch.linalg.svd(J, full_matrices=False)
        return int((S > tol).sum().item())

    def test_effective_rank_at_least_3(self):
        """The three-function expansion must achieve effective rank ≥ 3."""
        in_f, out_f = 16, 8
        a = torch.randn(out_f, 1) * 0.1
        b = torch.randn(1, in_f) * 0.1

        # Build Jacobian by evaluating at multiple input points
        X = torch.randn(32, in_f)
        outputs = self._funlora_forward(X, a, b)  # (32, out_f)

        # Treat output matrix directly — its rank reflects functional diversity
        rank = self._numerical_rank(outputs, tol=1e-3)
        assert rank >= 3, (
            f"FunLoRA effective rank={rank} < 3 — nonlinear expansion is not working"
        )

    def test_linear_rank1_has_rank_1(self):
        """Sanity: the raw rank-1 product ab should have rank exactly 1."""
        in_f, out_f = 16, 8
        a = torch.randn(out_f, 1) * 0.1
        b = torch.randn(1, in_f) * 0.1
        X = torch.randn(32, in_f)
        ab = X @ b.T @ a.T
        rank = self._numerical_rank(ab, tol=1e-3)
        assert rank == 1, f"Rank-1 product has rank={rank} (expected 1)"


# ---------------------------------------------------------------------------
# Symmetric Bipolar Attention v1 validation invariants
# ---------------------------------------------------------------------------

class TestSymmetricBipolarAttention:
    def test_formula_variant_is_pinned(self):
        assert SBA_FORMULA_VARIANT == "sba-v1-valid-position-sinh-cosh-shared-denominator"

    def test_reference_formula_hand_checked_values(self):
        logits = torch.tensor([[[0.0, math.log(2.0), -math.log(2.0)]]], dtype=torch.float64)
        weights = sba_reference_attention(logits)
        expected = torch.tensor([[[0.0, 0.21428571428571427, -0.21428571428571427]]], dtype=torch.float64)
        assert torch.allclose(weights, expected, atol=1e-12, rtol=0.0)

    def test_optimized_matches_reference_over_shapes_and_masks(self):
        generator = torch.Generator().manual_seed(20260518)
        shapes = [(2, 3), (2, 4, 5), (2, 2, 3, 7)]
        for shape in shapes:
            logits = torch.randn(*shape, generator=generator, dtype=torch.float64) * 3.0
            valid_mask = torch.rand(*shape, generator=generator) > 0.25
            valid_mask[..., 0] = True
            reference = sba_reference_attention(logits, valid_mask)
            optimized = sba_optimized_attention(logits, valid_mask)
            assert torch.allclose(optimized, reference, atol=1e-10, rtol=1e-10)

    def test_query_key_mask_broadcasts_across_batch_and_heads(self):
        generator = torch.Generator().manual_seed(20260518)
        logits = torch.randn(2, 3, 4, 5, generator=generator, dtype=torch.float64)
        valid_mask = torch.tensor(
            [
                [True, True, False, True, False],
                [True, False, True, True, False],
                [False, True, True, False, True],
                [True, True, True, False, True],
            ]
        )
        expanded_mask = valid_mask.reshape(1, 1, 4, 5).expand_as(logits)

        broadcast_weights = sba_optimized_attention(logits, valid_mask)
        expanded_weights = sba_optimized_attention(logits, expanded_mask)

        assert torch.allclose(broadcast_weights, expanded_weights, atol=0.0, rtol=0.0)
        assert torch.all(broadcast_weights.masked_select(~expanded_mask) == 0.0)

    def test_batch_query_key_mask_broadcasts_across_heads(self):
        generator = torch.Generator().manual_seed(20260519)
        logits = torch.randn(2, 3, 4, 5, generator=generator, dtype=torch.float64)
        valid_mask = torch.tensor(
            [
                [
                    [True, True, False, True, False],
                    [True, False, True, True, False],
                    [False, True, True, False, True],
                    [True, True, True, False, True],
                ],
                [
                    [True, False, True, False, True],
                    [False, True, True, True, False],
                    [True, True, False, True, True],
                    [True, False, False, True, True],
                ],
            ]
        )
        expanded_mask = valid_mask.reshape(2, 1, 4, 5).expand_as(logits)

        broadcast_weights = sba_optimized_attention(logits, valid_mask)
        expanded_weights = sba_optimized_attention(logits, expanded_mask)

        assert torch.allclose(broadcast_weights, expanded_weights, atol=0.0, rtol=0.0)
        assert torch.all(broadcast_weights.masked_select(~expanded_mask) == 0.0)

    def test_key_and_batch_key_masks_broadcast_over_attention_axes(self):
        generator = torch.Generator().manual_seed(20260520)
        logits = torch.randn(2, 3, 4, 5, generator=generator, dtype=torch.float64)
        key_mask = torch.tensor([True, False, True, False, True])
        batch_key_mask = torch.tensor(
            [
                [True, False, True, True, False],
                [False, True, True, False, True],
            ]
        )

        expanded_key_mask = key_mask.reshape(1, 1, 1, 5).expand_as(logits)
        expanded_batch_key_mask = batch_key_mask.reshape(2, 1, 1, 5).expand_as(logits)

        key_weights = sba_optimized_attention(logits, key_mask)
        expanded_key_weights = sba_optimized_attention(logits, expanded_key_mask)
        batch_key_weights = sba_optimized_attention(logits, batch_key_mask)
        expanded_batch_key_weights = sba_optimized_attention(logits, expanded_batch_key_mask)

        assert torch.allclose(key_weights, expanded_key_weights, atol=0.0, rtol=0.0)
        assert torch.all(key_weights.masked_select(~expanded_key_mask) == 0.0)
        assert torch.allclose(batch_key_weights, expanded_batch_key_weights, atol=0.0, rtol=0.0)
        assert torch.all(batch_key_weights.masked_select(~expanded_batch_key_mask) == 0.0)

    def test_ambiguous_2d_mask_with_equal_batch_and_query_is_rejected(self):
        logits = torch.randn(2, 3, 2, 5, dtype=torch.float64)
        valid_mask = torch.tensor(
            [
                [True, False, True, True, False],
                [False, True, True, False, True],
            ]
        )

        with pytest.raises(ValueError, match=r"ambiguous.*\(Q, K\).*\(B, K\)"):
            sba_optimized_attention(logits, valid_mask)

    def test_safe_masking_and_all_masked_rows(self):
        logits = torch.tensor([[[50.0, -50.0, 0.0], [1.0, 2.0, 3.0]]], dtype=torch.float64)
        valid_mask = torch.tensor([[[True, False, True], [False, False, False]]])
        values = torch.tensor([[[1.0, 0.0], [1000.0, 1000.0], [0.0, 1.0]]], dtype=torch.float64)
        weights, attended = sba_value_attention(logits, values, valid_mask)
        assert weights[0, 0, 1].item() == 0.0
        assert torch.all(weights[0, 1] == 0.0)
        assert torch.all(attended[0, 1] == 0.0)
        assert torch.isfinite(weights).all().item()
        assert torch.isfinite(attended).all().item()
        assert abs(attended[0, 0, 1].item()) <= 1.0

    def test_signed_l1_bound_and_finite_gradients(self):
        logits = torch.tensor(
            [[[15.0, -12.0, 0.5, -0.25], [0.0, 0.0, 0.0, 0.0]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        valid_mask = torch.tensor([[[True, True, True, False], [True, True, True, True]]])
        weights = sba_optimized_attention(logits, valid_mask)
        diagnostics = sba_attention_diagnostics(weights, valid_mask)
        assert diagnostics["finite"] is True
        assert diagnostics["max_masked_leakage"] == 0.0
        assert diagnostics["max_signed_l1"] <= 1.0 + 1e-12
        loss = (weights.square().sum() + weights.sum())
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all().item()

    def test_jacobian_claim_is_measured_not_assumed(self):
        logits = torch.tensor([[0.25, -0.75, 1.25]], dtype=torch.float64)
        valid_mask = torch.tensor([[True, True, True]])
        sba_norm = sba_jacobian_frobenius_norm(logits, valid_mask)
        softmax_norm = softmax_jacobian_frobenius_norm(logits, valid_mask)
        assert math.isfinite(sba_norm)
        assert math.isfinite(softmax_norm)
        assert sba_norm >= 0.0
        assert softmax_norm >= 0.0
