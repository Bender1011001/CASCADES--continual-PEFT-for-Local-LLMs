"""
CASCADES Sleep Cycle — Bio-Inspired Scheduled Consolidation

Implements a full "sleep cycle" inspired by neuroscience research on
memory consolidation during sleep. Rather than only distilling when
D-MoLE demotes a layer (reactive nap), this module runs periodic
offline consolidation passes that:

1. **SVD Core Consolidation** — Identifies redundant dimensions across
   all K-cores within each adapter and compresses them.
2. **Cross-Adapter Redundancy Detection** — Finds duplicate structure
   across adapters at different layers and merges shared components.
3. **Stiefel Re-Orthogonalization** — Corrects accumulated numerical
   drift in U_shared/V_shared (QR retraction isn't exact over 1000s of steps).
4. **Synaptic Homeostasis (SHY)** — Globally scales core magnitudes
   to prevent runaway growth, inspired by the Synaptic Homeostasis
   Hypothesis (Tononi & Cirelli, 2006).

Integration:
    Called between tasks in the training loop, after D-MoLE migration
    and before evaluation. Can also be called periodically every N steps.

Reference Architecture:
    - Tononi & Cirelli (2006) — Synaptic Homeostasis Hypothesis
    - Rasch & Born (2013) — Active system consolidation during sleep
    - CASCADES v9 — Dormant Core Distillation (reactive demotion sleep)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass


@dataclass
class SleepConfig:
    """Configuration for the sleep consolidation cycle.

    Attributes:
        enable_svd_consolidation: Compress redundant singular dimensions
            within each adapter's K-core pool.
        enable_cross_adapter_dedup: Detect and merge duplicate structure
            across adapters at different model layers.
        enable_reorthogonalization: Correct numerical drift in Stiefel
            manifold bases (U_shared, V_shared).
        enable_synaptic_homeostasis: Globally rescale core magnitudes to
            prevent runaway growth (SHY hypothesis).
        shy_target_norm: Target Frobenius norm for synaptic homeostasis.
            Cores are scaled toward this value.
        svd_energy_threshold: Fraction of singular energy to retain during
            consolidation (0.95 = keep 95% of energy, prune the rest).
        dedup_cosine_threshold: Cosine similarity threshold above which
            two adapter cores are considered duplicates and merged.
        reorth_drift_threshold: Maximum allowed deviation from
            orthonormality (||U^T U - I||_F) before re-orthogonalization.
        verbose: Print detailed sleep cycle diagnostics.
    """
    enable_svd_consolidation: bool = True
    enable_cross_adapter_dedup: bool = True
    enable_reorthogonalization: bool = True
    enable_synaptic_homeostasis: bool = True
    shy_target_norm: float = 1.0
    svd_energy_threshold: float = 0.95
    dedup_cosine_threshold: float = 0.95
    reorth_drift_threshold: float = 1e-4
    verbose: bool = True


class SleepConsolidation:
    """Full bio-inspired sleep consolidation cycle for CASCADES adapters.

    This is the "deep sleep" phase that runs between tasks or periodically
    during training. Unlike the reactive Dormant Core Distillation (which
    fires only on D-MoLE demotion), this runs proactively to maintain
    manifold health and compress learned representations.
    """

    def __init__(self, config: SleepConfig | None = None):
        self.config = config or SleepConfig()
        self.cycle_count = 0
        self.stats_history: list[dict] = []

    def run(self, adapters: list, task_id: int = -1) -> dict:
        """Execute a full sleep consolidation cycle.

        Args:
            adapters: List of CASCADES_v6_Adapter (or equivalent) nn.Modules
                that have U_shared, V_shared, and liquid_core attributes.
            task_id: The task that just completed (for logging).

        Returns:
            Dictionary of consolidation statistics.
        """
        self.cycle_count += 1
        stats = {
            "cycle": self.cycle_count,
            "task_id": task_id,
            "num_adapters": len(adapters),
            "svd_dimensions_pruned": 0,
            "duplicates_merged": 0,
            "reorth_corrections": 0,
            "shy_rescaled": 0,
        }

        if not adapters:
            return stats

        if self.config.verbose:
            print(f"\n{'🛌' * 3} SLEEP CYCLE {self.cycle_count} "
                  f"(after Task {task_id}) {'🛌' * 3}")

        # Phase 1: SVD Core Consolidation
        if self.config.enable_svd_consolidation:
            stats["svd_dimensions_pruned"] = self._svd_consolidation(adapters)

        # Phase 2: Cross-Adapter Redundancy Detection
        if self.config.enable_cross_adapter_dedup:
            stats["duplicates_merged"] = self._cross_adapter_dedup(adapters)

        # Phase 3: Stiefel Re-Orthogonalization
        if self.config.enable_reorthogonalization:
            stats["reorth_corrections"] = self._reorthogonalize(adapters)

        # Phase 4: Synaptic Homeostasis Scaling
        if self.config.enable_synaptic_homeostasis:
            stats["shy_rescaled"] = self._synaptic_homeostasis(adapters)

        self.stats_history.append(stats)

        if self.config.verbose:
            print(f"  Sleep cycle complete: "
                  f"{stats['svd_dimensions_pruned']} dims pruned, "
                  f"{stats['duplicates_merged']} duplicates merged, "
                  f"{stats['reorth_corrections']} bases corrected, "
                  f"{stats['shy_rescaled']} cores rescaled")
            print(f"{'🛌' * 3} SLEEP CYCLE {self.cycle_count} DONE {'🛌' * 3}\n")

        return stats

    @torch.no_grad()
    def _svd_consolidation(self, adapters: list) -> int:
        """Phase 1: Compress redundant singular dimensions within each
        adapter's K-core pool.

        For each adapter, compute SVD of the mean core. If the bottom
        singular values carry < (1-threshold) of total energy, rotate
        the core pool to align with principal directions and zero out
        the weak dimensions. This is like the brain pruning weak synapses
        during slow-wave sleep.
        """
        total_pruned = 0
        threshold = self.config.svd_energy_threshold

        for adapter in adapters:
            if not hasattr(adapter, "liquid_core"):
                continue

            core_pool = adapter.liquid_core.core_pool  # (K, r, r)
            if core_pool.ndim != 3 or core_pool.shape[1] < 2:
                continue

            # Compute structural consensus (mean core)
            mean_core = core_pool.mean(dim=0)  # (r, r)
            U_c, S_c, Vh_c = torch.linalg.svd(mean_core, full_matrices=False)

            # Compute cumulative energy
            total_energy = S_c.sum()
            if total_energy < 1e-10:
                continue

            cumulative = torch.cumsum(S_c, dim=0) / total_energy
            # Find the rank at which we hit the energy threshold
            keep_rank = int((cumulative >= threshold).float().argmax().item()) + 1
            keep_rank = max(keep_rank, 2)  # Never go below rank 2

            pruned = core_pool.shape[1] - keep_rank
            if pruned <= 0:
                continue

            # Soft-decay the weak dimensions instead of hard pruning
            # (gentler than amputation — like weakening synapses, not cutting)
            decay_factor = 0.1  # Reduce weak dimensions by 90%
            for k in range(core_pool.shape[0]):
                # Rotate core to SVD-aligned coordinates
                aligned = U_c.T @ core_pool[k] @ Vh_c.T
                # Decay the weak dimensions
                aligned[keep_rank:, :] *= decay_factor
                aligned[:, keep_rank:] *= decay_factor
                # Rotate back
                core_pool.data[k] = U_c @ aligned @ Vh_c

            total_pruned += pruned

            if self.config.verbose:
                energy_kept = cumulative[keep_rank - 1].item() * 100
                print(f"  💤 SVD consolidation: pruned {pruned} weak dims "
                      f"(keeping {keep_rank}/{core_pool.shape[1]}, "
                      f"{energy_kept:.1f}% energy retained)")

        return total_pruned

    @torch.no_grad()
    def _cross_adapter_dedup(self, adapters: list) -> int:
        """Phase 2: Detect duplicate structure across adapters at
        different model layers.

        If two adapters have very similar mean cores (high cosine
        similarity), merge them by averaging and distributing. This is
        like the brain consolidating overlapping episodic memories into
        shared semantic knowledge during REM sleep.
        """
        merged = 0
        threshold = self.config.dedup_cosine_threshold

        # Collect mean cores as flattened vectors
        adapter_cores = []
        for i, adapter in enumerate(adapters):
            if not hasattr(adapter, "liquid_core"):
                continue
            mean_core = adapter.liquid_core.core_pool.mean(dim=0)  # (r, r)
            flat = mean_core.flatten()
            norm = flat.norm()
            if norm > 1e-8:
                adapter_cores.append((i, adapter, flat / norm, mean_core))

        if len(adapter_cores) < 2:
            return 0

        # Pairwise cosine similarity (only check neighbors to avoid O(n²))
        seen_merged = set()
        for idx in range(len(adapter_cores) - 1):
            i, adapter_a, flat_a, core_a = adapter_cores[idx]
            j, adapter_b, flat_b, core_b = adapter_cores[idx + 1]

            if i in seen_merged or j in seen_merged:
                continue

            # Only compare adapters with same rank
            if core_a.shape != core_b.shape:
                continue

            cosine_sim = (flat_a * flat_b).sum().item()

            if cosine_sim > threshold:
                # Merge: average the cores and distribute to both
                avg_core = (core_a + core_b) / 2
                for k in range(adapter_a.liquid_core.core_pool.shape[0]):
                    frac_a = adapter_a.liquid_core.core_pool[k]
                    adapter_a.liquid_core.core_pool.data[k] = (
                        0.7 * frac_a + 0.3 * avg_core
                    )
                for k in range(adapter_b.liquid_core.core_pool.shape[0]):
                    frac_b = adapter_b.liquid_core.core_pool[k]
                    adapter_b.liquid_core.core_pool.data[k] = (
                        0.7 * frac_b + 0.3 * avg_core
                    )
                seen_merged.update({i, j})
                merged += 1

                if self.config.verbose:
                    print(f"  🔗 Cross-adapter merge: adapters {i}↔{j} "
                          f"(cosine={cosine_sim:.4f})")

        return merged

    @torch.no_grad()
    def _reorthogonalize(self, adapters: list) -> int:
        """Phase 3: Correct accumulated numerical drift in U_shared and
        V_shared Stiefel manifold bases.

        After thousands of QR retraction steps, floating-point errors
        accumulate and U^T U drifts from identity. This is like the brain
        recalibrating its reference frames during deep sleep.

        Note on shapes:
            U_shared: (d_out, r) — columns are orthonormal → U^T U ≈ I_r
            V_shared: (r, d_in) — rows are orthonormal   → V V^T ≈ I_r
        """
        corrected = 0
        threshold = self.config.reorth_drift_threshold

        for adapter in adapters:
            if not hasattr(adapter, "U_shared"):
                continue

            U = adapter.U_shared  # (d_out, r) — orthonormal columns
            V = adapter.V_shared  # (r, d_in) — orthonormal rows

            r = U.shape[1]
            I_r = torch.eye(r, device=U.device, dtype=U.dtype)

            # Measure orthonormality deviation
            drift_U = (U.T @ U - I_r).norm().item()       # columns check
            drift_V = (V @ V.T - I_r).norm().item()        # rows check

            if drift_U > threshold or drift_V > threshold:
                # Re-orthogonalize U via QR (column-orthonormal)
                if drift_U > threshold:
                    Q_U, R_U = torch.linalg.qr(U)
                    # U_old ≈ Q_U @ R_U → cores absorb R_U: S_new = R_U @ S_old
                    R_U_inv = torch.linalg.pinv(R_U)  # (r, r)
                    for k in range(adapter.liquid_core.core_pool.shape[0]):
                        adapter.liquid_core.core_pool.data[k] = (
                            R_U @ adapter.liquid_core.core_pool.data[k]
                        )
                    if hasattr(adapter, "ema_U"):
                        adapter.ema_U.data = adapter.ema_U.data @ R_U_inv
                    if hasattr(adapter, "streaming_sketch_U"):
                        adapter.streaming_sketch_U.data = (
                            adapter.streaming_sketch_U.data @ R_U_inv
                        )
                    adapter.U_shared.data.copy_(Q_U)

                # Re-orthogonalize V via QR on V^T (row-orthonormal)
                if drift_V > threshold:
                    # V is (r, d_in). QR works on columns, so decompose V^T
                    Q_Vt, R_Vt = torch.linalg.qr(V.T)  # V^T = Q_Vt @ R_Vt
                    # V_new = Q_Vt^T = row-orthonormal (r, d_in)
                    # V_old = R_Vt^T @ Q_Vt^T → cores absorb R_Vt^T on right
                    R_Vt_inv = torch.linalg.pinv(R_Vt)  # (r, r)
                    for k in range(adapter.liquid_core.core_pool.shape[0]):
                        adapter.liquid_core.core_pool.data[k] = (
                            adapter.liquid_core.core_pool.data[k] @ R_Vt.T
                        )
                    if hasattr(adapter, "ema_V"):
                        adapter.ema_V.data = R_Vt_inv.T @ adapter.ema_V.data
                    adapter.V_shared.data.copy_(Q_Vt.T)

                corrected += 1

                if self.config.verbose:
                    print(f"  🔧 Re-orthogonalized: drift_U={drift_U:.6f}, "
                          f"drift_V={drift_V:.6f}")

        return corrected

    @torch.no_grad()
    def _synaptic_homeostasis(self, adapters: list) -> int:
        """Phase 4: Globally rescale core magnitudes toward a target
        norm (Synaptic Homeostasis Hypothesis).

        During waking, synaptic strengths tend to increase (LTP dominates).
        During sleep, the brain uniformly downscales synapses, keeping
        relative differences but reducing absolute magnitudes. This
        prevents runaway growth and improves signal-to-noise ratio.

        We implement this as a soft exponential moving average toward
        the target norm — not a hard clamp.
        """
        rescaled = 0
        target = self.config.shy_target_norm
        shy_rate = 0.3  # How aggressively to scale toward target

        for adapter in adapters:
            if not hasattr(adapter, "liquid_core"):
                continue

            core_pool = adapter.liquid_core.core_pool  # (K, r, r)
            for k in range(core_pool.shape[0]):
                core = core_pool[k]
                current_norm = core.norm().item()

                if current_norm < 1e-10:
                    continue

                # Soft scaling: blend toward target norm
                # scale = (1 - shy_rate) + shy_rate * (target / current_norm)
                ratio = target / current_norm
                # Clamp the ratio to prevent extreme rescaling
                ratio = max(0.5, min(2.0, ratio))
                scale = (1 - shy_rate) + shy_rate * ratio

                if abs(scale - 1.0) > 0.01:  # Only rescale if meaningful
                    core_pool.data[k] *= scale
                    rescaled += 1

        if self.config.verbose and rescaled > 0:
            print(f"  ⚖️  Synaptic homeostasis: rescaled {rescaled} cores "
                  f"toward target norm {target:.2f}")

        return rescaled

    def summary(self) -> str:
        """Return a human-readable summary of all sleep cycles."""
        if not self.stats_history:
            return "No sleep cycles completed."

        lines = ["Sleep Cycle History:"]
        for s in self.stats_history:
            lines.append(
                f"  Cycle {s['cycle']} (Task {s['task_id']}): "
                f"pruned={s['svd_dimensions_pruned']}, "
                f"merged={s['duplicates_merged']}, "
                f"reorth={s['reorth_corrections']}, "
                f"rescaled={s['shy_rescaled']}"
            )
        return "\n".join(lines)
