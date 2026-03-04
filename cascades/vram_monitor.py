"""
VRAM monitoring utility for CASCADES training.

Logs peak GPU memory usage at key checkpoints to diagnose OOM issues
and track memory consumption across training phases.
"""

from __future__ import annotations

import functools
import time
from typing import Optional

import torch


def get_vram_stats(device: str = "cuda") -> dict[str, float]:
    """Get current VRAM statistics in MB.

    Returns:
        Dict with keys: allocated_mb, reserved_mb, peak_mb, free_mb, total_mb.
    """
    if not torch.cuda.is_available():
        return {
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "peak_mb": 0.0,
            "free_mb": 0.0,
            "total_mb": 0.0,
        }

    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    free = total - reserved

    return {
        "allocated_mb": allocated,
        "reserved_mb": reserved,
        "peak_mb": peak,
        "free_mb": free,
        "total_mb": total,
    }


def log_vram(checkpoint: str, device: str = "cuda") -> dict[str, float]:
    """Log VRAM usage at a named checkpoint.

    Args:
        checkpoint: Human-readable name for this checkpoint
            (e.g., "after_model_load", "after_first_forward").
        device: CUDA device string.

    Returns:
        VRAM stats dict.
    """
    stats = get_vram_stats(device)
    print(
        f"  [VRAM] {checkpoint}: "
        f"alloc={stats['allocated_mb']:.0f}MB "
        f"peak={stats['peak_mb']:.0f}MB "
        f"reserved={stats['reserved_mb']:.0f}MB "
        f"free={stats['free_mb']:.0f}MB "
        f"(total={stats['total_mb']:.0f}MB)"
    )
    return stats


def clear_cache(checkpoint: Optional[str] = None, device: str = "cuda") -> None:
    """Clear CUDA cache and optionally log the effect.

    Args:
        checkpoint: If provided, logs VRAM before and after clearing.
        device: CUDA device string.
    """
    if not torch.cuda.is_available():
        return

    if checkpoint:
        before = torch.cuda.memory_allocated(device) / (1024 ** 2)

    torch.cuda.empty_cache()

    if checkpoint:
        after = torch.cuda.memory_allocated(device) / (1024 ** 2)
        freed = before - after
        print(
            f"  [VRAM] cache_clear@{checkpoint}: "
            f"freed={freed:.0f}MB, now={after:.0f}MB"
        )


def reset_peak_stats(device: str = "cuda") -> None:
    """Reset peak memory tracking for a new measurement window."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def check_oom_risk(
    threshold_mb: float = 7500.0,
    device: str = "cuda",
) -> bool:
    """Check if current allocation is dangerously close to OOM.

    Args:
        threshold_mb: Warning threshold in MB (default: 7500 for 8GB GPU).
        device: CUDA device string.

    Returns:
        True if OOM risk detected.
    """
    if not torch.cuda.is_available():
        return False

    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    if peak > threshold_mb:
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        print(
            f"  ⚠️  [VRAM WARNING] Peak {peak:.0f}MB exceeds "
            f"threshold {threshold_mb:.0f}MB "
            f"({peak / total * 100:.1f}% of {total:.0f}MB total)"
        )
        return True
    return False
