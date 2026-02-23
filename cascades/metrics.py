"""
Standard continual-learning evaluation metrics.

All functions operate on the accuracy matrix A where:
    A[t, i] = accuracy on task i measured right after finishing training on task t.

These definitions follow Lopez-Paz & Ranzato (2017) and are the standard
in the continual-learning literature.
"""

import numpy as np


def average_accuracy(accuracy_matrix: np.ndarray) -> float:
    """Mean accuracy across all tasks measured after training on the final task.

    ACC = (1/T) Σ_i A[T-1, i]

    Args:
        accuracy_matrix: T×T float array; A[t, i] = acc on task i after task t.

    Returns:
        Scalar in [0, 1].
    """
    return float(np.mean(accuracy_matrix[-1, :]))


def backward_transfer(accuracy_matrix: np.ndarray) -> float:
    """Average change in performance on previous tasks after subsequent training.

    BWT = (1 / (T-1)) Σ_{i=1}^{T-1} (A[T-1, i] - A[i, i])

    Positive BWT → model improves on old tasks as it learns new ones.
    Negative BWT → forgetting (catastrophic interference).

    Args:
        accuracy_matrix: T×T float array.

    Returns:
        Scalar (unbounded, but typically in (-1, 1)).

    Raises:
        ValueError: If fewer than 2 tasks (BWT undefined).
    """
    T = accuracy_matrix.shape[0]
    if T < 2:
        raise ValueError("BWT requires at least 2 tasks.")
    diffs = [accuracy_matrix[-1, i] - accuracy_matrix[i, i] for i in range(T - 1)]
    return float(np.mean(diffs))


def forward_transfer(accuracy_matrix: np.ndarray, random_baseline: np.ndarray) -> float:
    """Average influence of learning task i on future tasks j > i.

    FWT = (1 / (T-1)) Σ_{i=1}^{T-1} (A[i-1, i] - b_i)

    where b_i is random-chance accuracy on task i.

    Args:
        accuracy_matrix: T×T float array.
        random_baseline: 1-D array of length T with random-chance accuracy per task.

    Returns:
        Scalar (positive → positive transfer, negative → interference).
    """
    T = accuracy_matrix.shape[0]
    if T < 2:
        raise ValueError("FWT requires at least 2 tasks.")
    diffs = [accuracy_matrix[i - 1, i] - random_baseline[i] for i in range(1, T)]
    return float(np.mean(diffs))


def proxy_accuracy_from_loss(avg_loss: float) -> float:
    """Convert average cross-entropy loss to a proxy accuracy via exp(-loss).

    This is used when true token-level accuracy is not computed (i.e., in the
    QLoRA regime where decoding is expensive). Note: this saturates near 1.0
    for low-loss models and is NOT equivalent to classification accuracy.

    Args:
        avg_loss: Mean cross-entropy loss.

    Returns:
        Proxy accuracy in (0, 1].
    """
    return float(np.exp(-avg_loss))


def full_report(accuracy_matrix: np.ndarray, method_name: str = "Method") -> str:
    """Pretty-print all metrics for a completed accuracy matrix.

    Args:
        accuracy_matrix: T×T float array.
        method_name:     Label for the report header.

    Returns:
        Multi-line string suitable for logging.
    """
    T = accuracy_matrix.shape[0]
    lines = [
        f"=== {method_name} ===",
        f"  Tasks: {T}",
        f"  ACC   (avg final accuracy): {average_accuracy(accuracy_matrix)*100:.2f}%",
    ]
    if T >= 2:
        lines.append(f"  BWT   (backward transfer):  {backward_transfer(accuracy_matrix)*100:.2f}%")

    lines.append("")
    lines.append("  Accuracy matrix (rows=after-task, cols=eval-task):")
    header = "       " + "  ".join(f"T{i}" for i in range(T))
    lines.append(header)
    for t in range(T):
        row = "  ".join(
            f"{accuracy_matrix[t, i]*100:5.1f}%" if accuracy_matrix[t, i] > 0 else "  — %"
            for i in range(T)
        )
        lines.append(f"  T{t} |  {row}")

    return "\n".join(lines)
