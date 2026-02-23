"""
Shared pytest fixtures for the CASCADES test suite.

All tests run on CPU (no GPU required) for reproducibility in CI.
"""

import pytest
import torch


@pytest.fixture(autouse=True)
def seed_everything():
    """Fix the global RNG seed before every test for reproducibility."""
    torch.manual_seed(0)
    yield


@pytest.fixture
def small_matrix():
    """A tall thin matrix (n=16, r=4) with random entries — not yet orthonormal."""
    return torch.randn(16, 4)


@pytest.fixture
def orthonormal_matrix(small_matrix):
    """QR-orthonormalized version of small_matrix — lives on the Stiefel manifold."""
    Q, _ = torch.linalg.qr(small_matrix)
    return Q


@pytest.fixture
def square_matrix():
    """Small square matrix suitable for SVC calibration tests."""
    return torch.randn(4, 4)


@pytest.fixture
def occupied_basis():
    """Orthonormal basis for a 2-D occupied subspace embedded in ℝ^8."""
    B = torch.randn(8, 2)
    Q, _ = torch.linalg.qr(B)
    return Q


@pytest.fixture
def gradient_in_occupied(occupied_basis):
    """A gradient that lies entirely inside the occupied subspace."""
    coeffs = torch.randn(occupied_basis.shape[1])
    return occupied_basis @ coeffs


@pytest.fixture
def gradient_mixed(occupied_basis):
    """A gradient with components both inside and outside the occupied subspace."""
    return torch.randn(occupied_basis.shape[0])
