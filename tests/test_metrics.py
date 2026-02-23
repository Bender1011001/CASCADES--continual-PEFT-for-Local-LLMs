"""
Unit tests for ACC / BWT / proxy-accuracy metric functions.

Paper claims tested:
  - ACC is the mean of the final row of the accuracy matrix
  - BWT > 0 ↔ backward knowledge transfer (improvement on old tasks)
  - BWT < 0 ↔ catastrophic forgetting
  - BWT = 0 ↔ no change (perfect isolation)
  - proxy_accuracy_from_loss is monotonically decreasing in loss
"""

import math
import numpy as np
import pytest

from cascades.metrics import (
    average_accuracy,
    backward_transfer,
    proxy_accuracy_from_loss,
    full_report,
)


class TestAverageAccuracy:
    def test_perfect_retention(self):
        """If the model aces every task after training, ACC = 1.0."""
        A = np.eye(3)  # diagonal — model is perfect after each task, no change
        A[-1, :] = 1.0
        assert average_accuracy(A) == pytest.approx(1.0)

    def test_complete_forgetting(self):
        """If the model forgets everything except the last task, ACC = 1/T."""
        T = 4
        A = np.zeros((T, T))
        A[-1, -1] = 1.0
        assert average_accuracy(A) == pytest.approx(1.0 / T)

    def test_uses_final_row_only(self):
        """ACC must only look at A[-1, :], not earlier rows."""
        A = np.array([
            [0.9, 0.0, 0.0],
            [0.5, 0.8, 0.0],
            [0.3, 0.4, 0.7],  # final row
        ])
        expected = (0.3 + 0.4 + 0.7) / 3
        assert average_accuracy(A) == pytest.approx(expected)

    def test_single_task(self):
        A = np.array([[0.85]])
        assert average_accuracy(A) == pytest.approx(0.85)


class TestBackwardTransfer:
    def test_positive_bwt_means_improvement(self):
        """BWT > 0 when final performance exceeds per-task peak."""
        A = np.array([
            [0.7, 0.0],
            [0.9, 0.8],  # task-0 improved from 0.7 → 0.9
        ])
        bwt = backward_transfer(A)
        assert bwt > 0, f"Expected positive BWT, got {bwt:.4f}"
        assert bwt == pytest.approx(0.9 - 0.7)

    def test_negative_bwt_means_forgetting(self):
        """BWT < 0 when later training degrades earlier task performance."""
        A = np.array([
            [0.9, 0.0],
            [0.3, 0.8],  # task-0 degraded from 0.9 → 0.3
        ])
        bwt = backward_transfer(A)
        assert bwt < 0, f"Expected negative BWT, got {bwt:.4f}"
        assert bwt == pytest.approx(0.3 - 0.9)

    def test_zero_bwt_means_isolation(self):
        """BWT = 0 when later tasks have no effect on earlier ones."""
        A = np.array([
            [0.8, 0.0, 0.0],
            [0.8, 0.7, 0.0],
            [0.8, 0.7, 0.6],
        ])
        bwt = backward_transfer(A)
        assert bwt == pytest.approx(0.0, abs=1e-6)

    def test_single_task_raises(self):
        A = np.array([[0.8]])
        with pytest.raises(ValueError, match="at least 2 tasks"):
            backward_transfer(A)

    def test_three_task_computation(self):
        """Verify manual calculation for a 3-task matrix."""
        A = np.array([
            [0.9, 0.0, 0.0],
            [0.6, 0.8, 0.0],
            [0.5, 0.7, 0.75],
        ])
        # BWT = mean([A[2,0]-A[0,0], A[2,1]-A[1,1]])
        #      = mean([0.5-0.9, 0.7-0.8])
        #      = mean([-0.4, -0.1]) = -0.25
        expected = (-0.4 + -0.1) / 2
        assert backward_transfer(A) == pytest.approx(expected)

    def test_cascades_v3_vs_lora_baseline(self):
        """Regression test: CASCADES v3.1 must have better BWT than LoRA baseline."""
        # From observed experimental results (Table 1 in paper)
        lora_bwt = -0.284   # -28.4%
        cascades_bwt = -0.034  # -3.41%
        assert cascades_bwt > lora_bwt, (
            "CASCADES BWT should be better (less negative) than LoRA baseline"
        )


class TestProxyAccuracy:
    def test_zero_loss_gives_one(self):
        assert proxy_accuracy_from_loss(0.0) == pytest.approx(1.0)

    def test_high_loss_gives_small_value(self):
        assert proxy_accuracy_from_loss(10.0) == pytest.approx(math.exp(-10.0))

    def test_monotone_decreasing(self):
        """Higher loss must give lower proxy accuracy."""
        losses = [0.5, 1.0, 2.0, 5.0]
        proxies = [proxy_accuracy_from_loss(l) for l in losses]
        for i in range(len(proxies) - 1):
            assert proxies[i] > proxies[i + 1], (
                f"proxy_accuracy is not monotone decreasing: {proxies}"
            )

    def test_always_positive(self):
        for loss in [0.0, 0.1, 1.0, 100.0]:
            assert proxy_accuracy_from_loss(loss) > 0.0


class TestFullReport:
    def test_returns_string(self):
        A = np.array([[0.8, 0.0], [0.5, 0.7]])
        report = full_report(A, "TestMethod")
        assert isinstance(report, str)
        assert "TestMethod" in report
        assert "ACC" in report
        assert "BWT" in report

    def test_single_task_no_bwt_line(self):
        A = np.array([[0.9]])
        report = full_report(A, "SingleTask")
        assert "BWT" not in report
