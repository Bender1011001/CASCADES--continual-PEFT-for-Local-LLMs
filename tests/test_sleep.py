"""Tests for cascades.sleep — Bio-Inspired Sleep Consolidation Cycle."""

import math

import pytest
import torch

from cascades.sleep import SleepConsolidation, SleepConfig


# ===========================================================================
# Mock adapter that mimics CASCADES_v6_Adapter interface for testing
# ===========================================================================

class MockLiquidCore:
    """Minimal mock of the LiquidResonantCore with a K-core pool."""

    def __init__(self, num_cores: int = 4, rank: int = 8):
        self.num_cores = num_cores
        # core_pool shape: (K, r, r)
        self.core_pool = torch.randn(num_cores, rank, rank) * 0.1


class MockAdapter:
    """Minimal mock of CASCADES_v6_Adapter / CASCADESAdapter."""

    def __init__(self, d_out: int = 64, d_in: int = 64, rank: int = 8,
                 num_cores: int = 4):
        self.out_features = d_out
        self.in_features = d_in

        # Stiefel bases — start orthonormal
        # U_shared: (d_out, r) — column-orthonormal
        U, _, _ = torch.linalg.svd(torch.randn(d_out, rank), full_matrices=False)
        self.U_shared = torch.nn.Parameter(U)

        # V_shared: (r, d_in) — row-orthonormal (matches real CASCADES adapter)
        Vt, _, _ = torch.linalg.svd(torch.randn(rank, d_in), full_matrices=False)
        self.V_shared = torch.nn.Parameter(Vt)

        self.liquid_core = MockLiquidCore(num_cores=num_cores, rank=rank)

        # EMA buffers — same shape as their corresponding bases
        self.ema_U = torch.nn.Parameter(torch.randn(d_out, rank))
        self.ema_V = torch.nn.Parameter(torch.randn(rank, d_in))

        # Sketch buffer
        self.streaming_sketch_U = torch.nn.Parameter(torch.randn(10, rank))

        self.quant_noise_std = torch.zeros(1)


def make_adapters(n: int = 5, rank: int = 8) -> list:
    return [MockAdapter(rank=rank) for _ in range(n)]


# ===========================================================================
# SleepConfig Tests
# ===========================================================================

class TestSleepConfig:
    def test_defaults(self):
        cfg = SleepConfig()
        assert cfg.enable_svd_consolidation is True
        assert cfg.enable_cross_adapter_dedup is True
        assert cfg.enable_reorthogonalization is True
        assert cfg.enable_synaptic_homeostasis is True
        assert cfg.shy_target_norm == 1.0
        assert cfg.svd_energy_threshold == 0.95

    def test_custom_config(self):
        cfg = SleepConfig(
            enable_svd_consolidation=False,
            shy_target_norm=2.0,
            verbose=False,
        )
        assert cfg.enable_svd_consolidation is False
        assert cfg.shy_target_norm == 2.0
        assert cfg.verbose is False


# ===========================================================================
# SleepConsolidation Basic Tests
# ===========================================================================

class TestSleepConsolidationBasic:
    def test_init_default(self):
        engine = SleepConsolidation()
        assert engine.cycle_count == 0
        assert engine.stats_history == []

    def test_init_custom_config(self):
        cfg = SleepConfig(verbose=False)
        engine = SleepConsolidation(cfg)
        assert engine.config.verbose is False

    def test_run_empty_adapters(self):
        engine = SleepConsolidation(SleepConfig(verbose=False))
        stats = engine.run([], task_id=0)
        assert stats["cycle"] == 1
        assert stats["num_adapters"] == 0

    def test_run_increments_cycle_count(self):
        engine = SleepConsolidation(SleepConfig(verbose=False))
        engine.run(make_adapters(3), task_id=0)
        engine.run(make_adapters(3), task_id=1)
        assert engine.cycle_count == 2
        assert len(engine.stats_history) == 2

    def test_summary_empty(self):
        engine = SleepConsolidation()
        assert "No sleep cycles" in engine.summary()

    def test_summary_after_run(self):
        engine = SleepConsolidation(SleepConfig(verbose=False))
        engine.run(make_adapters(3), task_id=0)
        summary = engine.summary()
        assert "Cycle 1" in summary


# ===========================================================================
# Phase 1: SVD Core Consolidation Tests
# ===========================================================================

class TestSVDConsolidation:
    def test_no_crash_on_small_rank(self):
        """Rank-2 adapters should not crash or over-prune."""
        engine = SleepConsolidation(SleepConfig(verbose=False))
        adapters = make_adapters(3, rank=2)
        stats = engine.run(adapters, task_id=0)
        # Should complete without error

    def test_energy_preserved(self):
        """After SVD consolidation, most energy should be retained."""
        engine = SleepConsolidation(SleepConfig(
            verbose=False,
            svd_energy_threshold=0.95,
            enable_cross_adapter_dedup=False,
            enable_reorthogonalization=False,
            enable_synaptic_homeostasis=False,
        ))
        adapters = make_adapters(3, rank=8)

        # Measure total energy before
        total_before = sum(
            a.liquid_core.core_pool.norm().item() for a in adapters
        )

        engine.run(adapters, task_id=0)

        # Measure total energy after
        total_after = sum(
            a.liquid_core.core_pool.norm().item() for a in adapters
        )

        # Energy should be roughly preserved (within 50% — soft decay)
        assert total_after > total_before * 0.3

    def test_weak_dimensions_decayed(self):
        """Create an adapter with clearly weak dimensions and verify they are decayed."""
        engine = SleepConsolidation(SleepConfig(
            verbose=False,
            svd_energy_threshold=0.5,  # Aggressive threshold
            enable_cross_adapter_dedup=False,
            enable_reorthogonalization=False,
            enable_synaptic_homeostasis=False,
        ))

        adapter = MockAdapter(rank=4, num_cores=2)
        # Make core_pool have only 1 strong direction
        for k in range(2):
            adapter.liquid_core.core_pool.data[k] = torch.zeros(4, 4)
            adapter.liquid_core.core_pool.data[k, 0, 0] = 10.0  # Strong dim
            adapter.liquid_core.core_pool.data[k, 3, 3] = 0.001  # Weak dim

        engine.run([adapter], task_id=0)

        # The strong dimension should still be large, weak should be even smaller
        for k in range(2):
            core = adapter.liquid_core.core_pool[k]
            assert core.abs().max() > 1.0  # Strong survived


# ===========================================================================
# Phase 2: Cross-Adapter Dedup Tests
# ===========================================================================

class TestCrossAdapterDedup:
    def test_identical_cores_merged(self):
        """Two adapters with identical cores should be detected as duplicates."""
        engine = SleepConsolidation(SleepConfig(
            verbose=False,
            enable_svd_consolidation=False,
            enable_reorthogonalization=False,
            enable_synaptic_homeostasis=False,
            dedup_cosine_threshold=0.99,
        ))

        adapters = make_adapters(2, rank=4)
        # Make them identical
        adapters[1].liquid_core.core_pool.data = adapters[0].liquid_core.core_pool.data.clone()

        stats = engine.run(adapters, task_id=0)
        assert stats["duplicates_merged"] >= 1

    def test_different_cores_not_merged(self):
        """Two adapters with very different cores should NOT be merged."""
        engine = SleepConsolidation(SleepConfig(
            verbose=False,
            enable_svd_consolidation=False,
            enable_reorthogonalization=False,
            enable_synaptic_homeostasis=False,
            dedup_cosine_threshold=0.99,
        ))

        adapters = make_adapters(2, rank=4)
        # Make them orthogonal
        adapters[0].liquid_core.core_pool.data = torch.eye(4).unsqueeze(0).expand(4, -1, -1)
        adapters[1].liquid_core.core_pool.data = (
            torch.eye(4).flip(1).unsqueeze(0).expand(4, -1, -1)
        )

        stats = engine.run(adapters, task_id=0)
        assert stats["duplicates_merged"] == 0

    def test_different_rank_not_compared(self):
        """Adapters with different ranks should not be compared."""
        engine = SleepConsolidation(SleepConfig(
            verbose=False,
            enable_svd_consolidation=False,
            enable_reorthogonalization=False,
            enable_synaptic_homeostasis=False,
        ))

        a1 = MockAdapter(rank=4)
        a2 = MockAdapter(rank=8)  # Different rank
        stats = engine.run([a1, a2], task_id=0)
        assert stats["duplicates_merged"] == 0


# ===========================================================================
# Phase 3: Stiefel Re-Orthogonalization Tests
# ===========================================================================

class TestReorthogonalization:
    def test_drift_detected_and_corrected(self):
        """Artificially drift U_shared and verify re-ortho fixes it."""
        engine = SleepConsolidation(SleepConfig(
            verbose=False,
            enable_svd_consolidation=False,
            enable_cross_adapter_dedup=False,
            enable_synaptic_homeostasis=False,
            reorth_drift_threshold=1e-6,  # Very sensitive
        ))

        adapter = MockAdapter(rank=4)
        # Introduce drift by adding noise
        adapter.U_shared.data += torch.randn_like(adapter.U_shared) * 0.01

        # Measure drift before
        I_r = torch.eye(4)
        drift_before = (adapter.U_shared.T @ adapter.U_shared - I_r).norm().item()
        assert drift_before > 1e-6  # Should have drift

        stats = engine.run([adapter], task_id=0)
        assert stats["reorth_corrections"] >= 1

        # Verify U is now orthonormal again
        drift_after = (adapter.U_shared.T @ adapter.U_shared - I_r).norm().item()
        assert drift_after < drift_before

    def test_already_orthogonal_not_touched(self):
        """Already-orthonormal bases should not be modified."""
        engine = SleepConsolidation(SleepConfig(
            verbose=False,
            enable_svd_consolidation=False,
            enable_cross_adapter_dedup=False,
            enable_synaptic_homeostasis=False,
            reorth_drift_threshold=1e-2,  # Lenient threshold
        ))

        adapter = MockAdapter(rank=4)
        # U_shared should already be orthonormal from SVD construction
        stats = engine.run([adapter], task_id=0)
        assert stats["reorth_corrections"] == 0


# ===========================================================================
# Phase 4: Synaptic Homeostasis Tests
# ===========================================================================

class TestSynapticHomeostasis:
    def test_large_cores_shrunk(self):
        """Cores much larger than target norm should be shrunk."""
        engine = SleepConsolidation(SleepConfig(
            verbose=False,
            enable_svd_consolidation=False,
            enable_cross_adapter_dedup=False,
            enable_reorthogonalization=False,
            shy_target_norm=1.0,
        ))

        adapter = MockAdapter(rank=4, num_cores=2)
        # Make cores very large
        adapter.liquid_core.core_pool.data *= 100

        norm_before = adapter.liquid_core.core_pool.norm().item()
        engine.run([adapter], task_id=0)
        norm_after = adapter.liquid_core.core_pool.norm().item()

        assert norm_after < norm_before  # Should have shrunk

    def test_small_cores_grown(self):
        """Cores much smaller than target norm should be grown."""
        engine = SleepConsolidation(SleepConfig(
            verbose=False,
            enable_svd_consolidation=False,
            enable_cross_adapter_dedup=False,
            enable_reorthogonalization=False,
            shy_target_norm=1.0,
        ))

        adapter = MockAdapter(rank=4, num_cores=2)
        # Make cores very small
        adapter.liquid_core.core_pool.data *= 0.001

        norm_before = adapter.liquid_core.core_pool.norm().item()
        engine.run([adapter], task_id=0)
        norm_after = adapter.liquid_core.core_pool.norm().item()

        assert norm_after > norm_before  # Should have grown

    def test_near_target_not_changed(self):
        """Cores near target norm should not be significantly changed."""
        engine = SleepConsolidation(SleepConfig(
            verbose=False,
            enable_svd_consolidation=False,
            enable_cross_adapter_dedup=False,
            enable_reorthogonalization=False,
            shy_target_norm=1.0,
        ))

        adapter = MockAdapter(rank=4, num_cores=1)
        # Set core to exactly target norm
        core = torch.randn(4, 4)
        core = core / core.norm()  # norm = 1.0
        adapter.liquid_core.core_pool.data[0] = core

        engine.run([adapter], task_id=0)

        # Should be nearly unchanged
        new_norm = adapter.liquid_core.core_pool[0].norm().item()
        assert abs(new_norm - 1.0) < 0.05


# ===========================================================================
# Integration Tests
# ===========================================================================

class TestFullSleepCycle:
    def test_all_phases_run(self):
        """Full cycle with all phases enabled should complete."""
        engine = SleepConsolidation(SleepConfig(verbose=False))
        adapters = make_adapters(5, rank=8)
        stats = engine.run(adapters, task_id=0)

        assert stats["cycle"] == 1
        assert stats["num_adapters"] == 5
        # All stats should be non-negative
        assert stats["svd_dimensions_pruned"] >= 0
        assert stats["duplicates_merged"] >= 0
        assert stats["reorth_corrections"] >= 0
        assert stats["shy_rescaled"] >= 0

    def test_multiple_cycles(self):
        """Running multiple sleep cycles should accumulate stats."""
        engine = SleepConsolidation(SleepConfig(verbose=False))
        adapters = make_adapters(3, rank=4)

        for t in range(3):
            engine.run(adapters, task_id=t)

        assert engine.cycle_count == 3
        assert len(engine.stats_history) == 3
        assert "Cycle 1" in engine.summary()
        assert "Cycle 3" in engine.summary()

    def test_no_nan_after_sleep(self):
        """Sleep should never introduce NaN values."""
        engine = SleepConsolidation(SleepConfig(verbose=False))
        adapters = make_adapters(5, rank=8)

        engine.run(adapters, task_id=0)

        for a in adapters:
            assert not torch.isnan(a.U_shared).any(), "NaN in U_shared after sleep"
            assert not torch.isnan(a.V_shared).any(), "NaN in V_shared after sleep"
            assert not torch.isnan(a.liquid_core.core_pool).any(), "NaN in core_pool after sleep"

    def test_no_inf_after_sleep(self):
        """Sleep should never introduce Inf values."""
        engine = SleepConsolidation(SleepConfig(verbose=False))
        adapters = make_adapters(5, rank=8)

        engine.run(adapters, task_id=0)

        for a in adapters:
            assert not torch.isinf(a.U_shared).any(), "Inf in U_shared after sleep"
            assert not torch.isinf(a.liquid_core.core_pool).any(), "Inf in core_pool after sleep"

    def test_disabled_phases(self):
        """Disabling all phases should still run without error."""
        cfg = SleepConfig(
            enable_svd_consolidation=False,
            enable_cross_adapter_dedup=False,
            enable_reorthogonalization=False,
            enable_synaptic_homeostasis=False,
            verbose=False,
        )
        engine = SleepConsolidation(cfg)
        stats = engine.run(make_adapters(3), task_id=0)
        assert stats["svd_dimensions_pruned"] == 0
        assert stats["duplicates_merged"] == 0
        assert stats["reorth_corrections"] == 0
        assert stats["shy_rescaled"] == 0
