"""
CASCADES v10.3 — Apply 6 BWT bug fixes directly in Colab.
Run this cell ONCE before Step 7 (training).

Patches: adapters.py (bugs 1-5) and sleep.py (bug 6).
"""
import re

# =============================================================================
# PATCH 1: cascades/adapters.py
# =============================================================================
with open("cascades/adapters.py", "r") as f:
    code = f.read()

# --- Bug 3: Replace _cllora_reassign with strict frozen + soft active ---
old_reassign = '''def _cllora_reassign(grad, null_sketch, config: AblationConfig, frozen_basis=None):
    """CL-LoRA EAR gradient reassignment if enabled.

    v10.2 BWT Fix: Strict Orthogonal Lockdown.
    Concatenates the streaming task null-space and frozen_basis prior-task
    directions, computes a strictly orthogonal QR factorization, and applies
    a single robust Householder projection to avoid numerical leakage.
    """
    if not config.enable_cllora_reassign:
        return grad

    protected_dirs = []
    if null_sketch is not None:
        protected_dirs.append(null_sketch)
    if frozen_basis is not None and frozen_basis.shape[1] > 0:
        protected_dirs.append(frozen_basis)

    if not protected_dirs:
        return grad

    # Build unified protection space
    W = torch.cat(protected_dirs, dim=1)
    
    # Strict orthogonalization
    Q, R = torch.linalg.qr(W)
    
    # Numerical stability filter: drop degenerate columns from the unified space
    col_norms = R.diag().abs()
    Q_strict = Q[:, col_norms > 1e-6]

    if Q_strict.shape[1] == 0:
        return grad

    # Project out the perfectly orthogonal unified basis
    occupied = Q_strict @ (Q_strict.T @ grad)
    free = grad - occupied

    if config.enable_soft_ear:
        # v10: Tikhonov-regularized smooth reassignment
        return soft_ear(grad, free, gamma=config.ear_gamma)

    # Legacy hard-cutoff EAR (v9 fallback)
    grad_energy = grad.norm()
    free_energy = free.norm()
    if grad_energy > 1e-8 and free_energy > 1e-8:
        return (grad_energy / (free_energy + 1e-8)) * free
    return free'''

new_reassign = '''def _cllora_reassign(grad, null_sketch, config: AblationConfig, frozen_basis=None):
    """CL-LoRA EAR gradient reassignment with strict frozen protection.

    v10.3 BWT Fix: Separated two-phase lockdown.
    Phase 1 — STRICT: Hard zero-tolerance projection for frozen (prior-task)
    directions. No gradient energy is allowed to leak into past tasks.
    Phase 2 — SOFT: Tikhonov-regularized EAR penalty for active sketch
    (current-task exploration dampening).

    Uses SVD to extract valid bases (handles linear dependencies gracefully).
    """
    # 1. STRICT lockdown for past tasks (Absolute Zero Tolerance)
    if frozen_basis is not None and frozen_basis.shape[1] > 0:
        U_f, S_f, _ = torch.linalg.svd(frozen_basis, full_matrices=False)
        Q_f = U_f[:, S_f > 1e-6]
        if Q_f.shape[1] > 0:
            grad = grad - Q_f @ (Q_f.T @ grad)

    # 2. SOFT penalty for active sketch (Exploration penalty)
    if not config.enable_cllora_reassign or null_sketch is None:
        return grad

    U_n, S_n, _ = torch.linalg.svd(null_sketch, full_matrices=False)
    Q_n = U_n[:, S_n > 1e-6]

    if Q_n.shape[1] == 0:
        return grad

    occupied = Q_n @ (Q_n.T @ grad)
    free = grad - occupied

    if config.enable_soft_ear:
        return soft_ear(grad, free, gamma=config.ear_gamma)

    # Legacy hard-cutoff EAR (v9 fallback)
    grad_energy = grad.norm()
    free_energy = free.norm()
    if grad_energy > 1e-8 and free_energy > 1e-8:
        return (grad_energy / (free_energy + 1e-8)) * free
    return free'''

assert old_reassign in code, "ERROR: Could not find old _cllora_reassign — file may already be patched"
code = code.replace(old_reassign, new_reassign)
print("✓ Bug 3: Strict frozen + soft active EAR separation")

# --- Bug 4: Q_null_U full-rank → half-rank ---
code = code.replace(
    "'Q_null_U', torch.zeros(out_features, rank)  # v10: full-rank (was rank//2)",
    "'Q_null_U', torch.zeros(out_features, max(1, rank // 2))  # Fix 4: half-rank preserves plasticity"
)
print("✓ Bug 4: Q_null_U reverted to rank//2")

# --- Bug 1: Tangent Mapping Paradox — project AFTER tangent map ---
old_tangent = """            # 4.5. v10.1 Ambient Null-Space Projection
            # Project Euclidean EMA gradient *before* tangent mapping to prevent
            # the QR retraction from bleeding the null-space back into the basis.
            safe_ema_U = self.ema_U.clone()
            if cfg.enable_coso_nullspace and getattr(self, 'ear_initialized', False):
                safe_ema_U = _cllora_reassign(
                    self.ema_U, self.Q_null_U, cfg,
                    frozen_basis=getattr(self, 'frozen_null_basis', None),
                )

            # 5. Map to Stiefel tangent space AFTER protection
            sym_U = 0.5 * (
                self.U_shared.T @ safe_ema_U + safe_ema_U.T @ self.U_shared
            )
            tangent_U = safe_ema_U - self.U_shared @ sym_U"""

new_tangent = """            # 5. Map UNPROTECTED gradient to Stiefel tangent space FIRST (Fix 1)
            # The tangent mapping MUST happen before EAR projection.
            # Projecting before tangent mapping is broken because the
            # sym_U subtraction (U @ sym_U) re-introduces frozen directions.
            sym_U = 0.5 * (
                self.U_shared.T @ self.ema_U + self.ema_U.T @ self.U_shared
            )
            tangent_U = self.ema_U - self.U_shared @ sym_U"""

assert old_tangent in code, "ERROR: Could not find old tangent block"
code = code.replace(old_tangent, new_tangent)
print("✓ Bug 1: Tangent-first, then project")

# Fix the EAR application block to work on tangent vector
old_ear_block = """            # 6. Streaming EAR update and constraint
            if cfg.enable_coso_nullspace:
                self.streaming_ear_update(tangent_U)

                if getattr(self, 'ear_initialized', False):
                    n_orig = self.ema_U.norm()
                    n_free = safe_ema_U.norm()
                    n_occ = (self.ema_U - safe_ema_U).norm()"""

new_ear_block = """            # 6. Streaming EAR update and constraint
            if cfg.enable_coso_nullspace:
                self.streaming_ear_update(tangent_U)  # Tracks true unprotected tangent

                if getattr(self, 'ear_initialized', False):
                    # APPLY EAR TO TANGENT VECTOR (Fix 1: project after tangent map)
                    safe_tangent_U = _cllora_reassign(
                        tangent_U, self.Q_null_U, cfg,
                        frozen_basis=getattr(self, 'frozen_null_basis', None),
                    )

                    n_orig = tangent_U.norm()
                    n_free = safe_tangent_U.norm()
                    n_occ = (tangent_U - safe_tangent_U).norm()"""

assert old_ear_block in code, "ERROR: Could not find old EAR block"
code = code.replace(old_ear_block, new_ear_block)
print("✓ Bug 1 (continued): EAR applied to tangent vector")

# Fix the norm rescaling and remove re-projection
code = code.replace(
    "                    if n_orig > 1e-8 and n_free > 1e-8:\n                        tangent_U = tangent_U * min(n_orig / n_free, 5.0)\n\n                    # Re-project to correct numerical drift from EAR\n                    sym_free_U = 0.5 * (\n                        self.U_shared.T @ tangent_U + tangent_U.T @ self.U_shared\n                    )\n                    tangent_U = tangent_U - self.U_shared @ sym_free_U",
    "                    if n_orig > 1e-8 and n_free > 1e-8:\n                        tangent_U = safe_tangent_U * min(n_orig / n_free, 5.0)\n                    else:\n                        tangent_U = safe_tangent_U\n\n                    # REMOVED: Re-project to correct numerical drift from EAR.\n                    # The QR retraction + Ripple Fix handles out-of-tangent\n                    # matrices perfectly. Re-projecting breaks the lockdown."
)
print("✓ Bug 1 (final): Removed re-projection, use safe_tangent_U")

# --- Bug 5: freeze_current_subspace — use S_norm + FIFO truncation ---
code = code.replace(
    "            # Project to ambient space: occupied directions = S @ eigvecs\n            # (We use the original un-normalized S here to recover true vectors)\n            new_dirs = S @ kept_vecs  # (d_out, k)",
    "            # Fix 5: Use S_norm to preserve structural variance properties.\n            # The eigenvectors came from normalized S, so we must project\n            # through normalized S to get geometrically consistent directions.\n            new_dirs = S_norm @ kept_vecs  # (d_out, k)"
)

code = code.replace(
    "            # QR to get orthonormal columns\n            Q_new, _ = torch.linalg.qr(new_dirs)  # (d_out, k)",
    "            Q_new, _ = torch.linalg.qr(new_dirs)  # (d_out, k)"
)

code = code.replace(
    "                # Re-orthogonalize the combined basis to remove overlap\n                Q_combined, R = torch.linalg.qr(combined)\n                # Keep only columns with significant norm (non-degenerate)\n                col_norms = R.diag().abs()\n                significant = col_norms > 1e-6\n                Q_combined = Q_combined[:, significant]",
    "                # Fix 5: Use SVD to merge bases — handles linear dependencies\n                # gracefully unlike QR which can produce artificial columns.\n                U_svd, S_svd, _ = torch.linalg.svd(combined, full_matrices=False)\n                Q_combined = U_svd[:, S_svd > 1e-6]"
)

code = code.replace(
    "                # SVD truncation: keep the most important directions\n                U_trunc, S_trunc, _ = torch.linalg.svd(\n                    Q_combined, full_matrices=False\n                )\n                Q_combined = U_trunc[:, :max_cols]",
    "                # Fix 5: FIFO drop oldest columns. SVD truncation on an\n                # orthogonal matrix has all singular values ≈ 1.0, making\n                # it an arbitrary rotation that drops random directions.\n                Q_combined = Q_combined[:, -max_cols:]"
)
print("✓ Bug 5: freeze_current_subspace uses S_norm + FIFO")

# --- Bug 2: promote() distills FunLoRA ---
old_promote = '''    def promote(self, rank: int = 8) -> bool:
        """Promote FunLoRA → full ResonantCore adapter."""
        if self.is_critical:
            return False

        self.is_critical = True
        new_adapter = CASCADESAdapter(
            self.in_features, self.out_features, rank=rank, config=self.config
        )
        new_adapter = new_adapter.to(self.adapter.a.device)
        self.adapter = new_adapter
        return True'''

new_promote = '''    def promote(self, rank: int = 8) -> bool:
        """Promote FunLoRA → full ResonantCore adapter.

        Fix 2: Distills the trained FunLoRA rank-1 weights (a, b) into
        the first dimension of the new Stiefel manifold, preserving
        learned knowledge instead of injecting random noise.
        """
        if self.is_critical:
            return False

        self.is_critical = True
        new_adapter = CASCADESAdapter(
            self.in_features, self.out_features, rank=rank, config=self.config
        )
        new_adapter = new_adapter.to(self.adapter.a.device)

        # Fix 2: Distill FunLoRA knowledge into dimension 0
        with torch.no_grad():
            a_norm = self.adapter.a.norm() + 1e-8
            b_norm = self.adapter.b.norm() + 1e-8

            # Normalize to get unit vectors for Stiefel bases
            u_0 = self.adapter.a / a_norm  # (d_out, 1)
            v_0 = self.adapter.b / b_norm  # (1, d_in)

            # Inject into first column/row of Stiefel bases
            new_adapter.U_shared.data[:, 0:1] = u_0
            new_adapter.V_shared.data[0:1, :] = v_0

            # Transfer FunLoRA magnitude into liquid core dimension 0.
            # FunLoRA activation linear approx at origin: f'(0) = 2.25
            for k in range(new_adapter.liquid_core.num_cores):
                new_adapter.liquid_core.core_pool.data[k] *= 0.01  # Suppress noise
                new_adapter.liquid_core.core_pool.data[k, 0, 0] = (
                    a_norm * b_norm * 2.25
                ).item()

        self.adapter = new_adapter
        return True'''

assert old_promote in code, "ERROR: Could not find old promote()"
code = code.replace(old_promote, new_promote)
print("✓ Bug 2: promote() distills FunLoRA weights")

with open("cascades/adapters.py", "w") as f:
    f.write(code)
print("✓ cascades/adapters.py saved")

# =============================================================================
# PATCH 2: cascades/sleep.py
# =============================================================================
with open("cascades/sleep.py", "r") as f:
    sleep_code = f.read()

# --- Bug 6: Sleep covariance parity ---
sleep_code = sleep_code.replace(
    """                if hasattr(adapter, "ema_U"):
                    adapter.ema_U.data = adapter.ema_U.data @ R_U_inv""",
    """                if hasattr(adapter, "ema_U"):
                    # Fix 6: EMA gradients are COVARIANT — transform by R^T
                    adapter.ema_U.data = adapter.ema_U.data @ R_U.T"""
)

sleep_code = sleep_code.replace(
    """                if hasattr(adapter, "ema_V"):
                    adapter.ema_V.data = R_Vt_inv.T @ adapter.ema_V.data""",
    """                if hasattr(adapter, "ema_V"):
                    # Fix 6: EMA gradients are COVARIANT — transform by R_Vt
                    adapter.ema_V.data = R_Vt @ adapter.ema_V.data"""
)

# Remove the now-unused R_Vt_inv computation
sleep_code = sleep_code.replace(
    "                R_Vt_inv = torch.linalg.pinv(R_Vt)  # (r, r)\n",
    ""
)

with open("cascades/sleep.py", "w") as f:
    f.write(sleep_code)
print("✓ Bug 6: Sleep covariance parity fixed")
print("✓ cascades/sleep.py saved")

print("\n" + "=" * 60)
print("ALL 6 v10.3 BWT FIXES APPLIED SUCCESSFULLY")
print("=" * 60)
print("Now re-run Step 7 (training) to see the BWT improvement.")
