"""
Monotone fidelity schedule for TDCF.

Implements Section 4.5:
  - Convert raw pilot sensitivity statistics into monotone schedules
    K(e) and q(e) using isotonic regression.
  - The schedule is non-decreasing: fidelity only increases over training.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression


class FidelityScheduler:
    """
    Fits monotone schedules K(e) and q(e) from pilot-phase sensitivity
    statistics, then provides fidelity parameters for any training epoch.
    """

    def __init__(self, num_bands: int, num_patches: int,
                 eta_f: float = 0.9, eta_s: float = 0.8):
        """
        Args:
            num_bands:   Total number of frequency bands B.
            num_patches: Total number of spatial patches P.
            eta_f:       Frequency retention threshold (Eq. 13).
            eta_s:       Spatial retention threshold (Eq. 14).
        """
        self.num_bands = num_bands
        self.num_patches = num_patches
        self.eta_f = eta_f
        self.eta_s = eta_s

        self._K_schedule = None     # fitted monotone band cutoff schedule
        self._q_schedule = None     # fitted monotone patch quota schedule
        self._pilot_epochs = 0
        self._total_epochs = 0

    def fit_from_pilot(self, sensitivity_estimator, total_epochs: int):
        """
        Fit monotone schedules from pilot sensitivity statistics.

        Uses isotonic regression to enforce monotonicity (Section 4.5).

        Args:
            sensitivity_estimator: A SensitivityEstimator with completed
                                    pilot epochs.
            total_epochs:          Total number of training epochs.
        """
        pilot_epochs = len(sensitivity_estimator.band_sensitivity_history)
        self._pilot_epochs = pilot_epochs
        self._total_epochs = total_epochs

        # ------------------------------------------------------------------
        # Raw cutoffs from pilot
        # ------------------------------------------------------------------
        raw_K = np.array([
            sensitivity_estimator.compute_band_cutoff(e, self.eta_f)
            for e in range(pilot_epochs)
        ], dtype=np.float64)

        raw_q = np.array([
            sensitivity_estimator.compute_patch_quota(e, self.eta_s)
            for e in range(pilot_epochs)
        ], dtype=np.float64)

        # ------------------------------------------------------------------
        # Isotonic regression to fit monotone non-decreasing curves
        # ------------------------------------------------------------------
        iso = IsotonicRegression(increasing=True)

        pilot_x = np.arange(pilot_epochs, dtype=np.float64)

        if pilot_epochs >= 2:
            K_fitted = iso.fit_transform(pilot_x, raw_K)
            q_fitted = iso.fit_transform(pilot_x, raw_q)
        else:
            K_fitted = raw_K.copy()
            q_fitted = raw_q.copy()

        # ------------------------------------------------------------------
        # Extrapolate to all training epochs
        # ------------------------------------------------------------------
        all_epochs = np.arange(total_epochs, dtype=np.float64)

        # Linear extrapolation from pilot trend, clamped to valid range
        if pilot_epochs >= 2:
            # Rate of increase per epoch
            K_rate = (K_fitted[-1] - K_fitted[0]) / max(pilot_epochs - 1, 1)
            q_rate = (q_fitted[-1] - q_fitted[0]) / max(pilot_epochs - 1, 1)
        else:
            K_rate = 0.0
            q_rate = 0.0

        self._K_schedule = np.zeros(total_epochs)
        self._q_schedule = np.zeros(total_epochs)

        for e in range(total_epochs):
            if e < pilot_epochs:
                self._K_schedule[e] = K_fitted[e]
                self._q_schedule[e] = q_fitted[e]
            else:
                # Extrapolate with clamping
                K_val = K_fitted[-1] + K_rate * (e - pilot_epochs + 1)
                q_val = q_fitted[-1] + q_rate * (e - pilot_epochs + 1)
                self._K_schedule[e] = K_val
                self._q_schedule[e] = q_val

        # Clamp to valid ranges
        self._K_schedule = np.clip(self._K_schedule, 1, self.num_bands)
        self._q_schedule = np.clip(self._q_schedule, 1, self.num_patches)

        # Enforce monotonicity on the full schedule
        for e in range(1, total_epochs):
            self._K_schedule[e] = max(self._K_schedule[e],
                                       self._K_schedule[e-1])
            self._q_schedule[e] = max(self._q_schedule[e],
                                       self._q_schedule[e-1])

        # Round to integers
        self._K_schedule = np.round(self._K_schedule).astype(int)
        self._q_schedule = np.round(self._q_schedule).astype(int)

    def get_fidelity(self, epoch: int) -> tuple:
        """
        Get the fidelity parameters for a given epoch (Eq. 15).

        Returns:
            (K_e, q_e): band cutoff and patch quota for this epoch.
        """
        if self._K_schedule is None:
            raise RuntimeError("Schedule not fitted yet. Call fit_from_pilot.")
        epoch = min(epoch, self._total_epochs - 1)
        return int(self._K_schedule[epoch]), int(self._q_schedule[epoch])

    def get_full_schedule(self) -> tuple:
        """Return the complete K(e) and q(e) arrays."""
        return self._K_schedule.copy(), self._q_schedule.copy()

    def summary(self) -> str:
        """Human-readable schedule summary."""
        if self._K_schedule is None:
            return "Schedule not fitted."
        lines = [
            f"TDCF Fidelity Schedule ({self._total_epochs} epochs, "
            f"fitted from {self._pilot_epochs} pilot epochs)",
            f"  η_f = {self.eta_f}, η_s = {self.eta_s}",
            f"  K(e): {self._K_schedule[0]} → {self._K_schedule[-1]} "
            f"(of {self.num_bands} bands)",
            f"  q(e): {self._q_schedule[0]} → {self._q_schedule[-1]} "
            f"(of {self.num_patches} patches)",
        ]
        return "\n".join(lines)


class BudgetScheduler:
    """
    Budget-constrained fidelity scheduler for TDCF.

    Instead of coverage thresholds (η_f, η_s) which drift toward full I/O
    on datasets with uniform sensitivity distributions, this scheduler
    directly controls the total band-slot budget as a fraction of full I/O.

    The greedy allocator then distributes that budget optimally by
    marginal utility:  Δ_utility(p, k) = g̃(p) × s(k)

    Schedule:
        B(e) = β + (1 − β) × (e / T)^γ

    where β is the starting budget ratio and γ controls the ramp shape.
    γ = 1.0 → linear ramp
    γ > 1.0 → hold low longer, ramp late (curriculum: coarse→fine)
    γ < 1.0 → ramp early, plateau high
    """

    def __init__(self, num_bands: int, num_patches: int,
                 beta: float = 0.5, gamma: float = 1.0,
                 k_low: int = 1):
        """
        Args:
            num_bands:   Total number of frequency bands B.
            num_patches: Total number of spatial patches P.
            beta:        Starting budget ratio (0 < β < 1).
                         β = 0.5 means start at 50% I/O.
            gamma:       Ramp exponent.  γ=1 is linear, γ=2 is "hold then ramp".
            k_low:       Minimum bands per patch (0 = skip background entirely).
        """
        self.num_bands = num_bands
        self.num_patches = num_patches
        self.beta = beta
        self.gamma = gamma
        self.k_low = k_low
        self._total_epochs = None
        self._pilot_epochs = 0

        # Full budget = P × B band-slots
        self.full_budget = num_patches * num_bands
        # Minimum budget = P × k_low
        self.min_budget = num_patches * k_low

    def fit_from_pilot(self, sensitivity_estimator, total_epochs: int):
        """
        Store pilot statistics.  The BudgetScheduler doesn't need isotonic
        regression — it uses a parametric curve.  But we still read the
        pilot to expose band/patch sensitivity to the greedy allocator.
        """
        self._pilot_epochs = len(sensitivity_estimator.band_sensitivity_history)
        self._total_epochs = total_epochs

    def get_budget(self, epoch: int) -> int:
        """
        Total band-slot budget for this epoch.

        B(e) = B_min + (B_full - B_min) × ratio(e)
        ratio(e) = β + (1 − β) × (e / T)^γ

        Returns:
            Integer number of total band-slots (max = P × num_bands).
        """
        T = max(self._total_epochs - 1, 1)
        t = min(epoch, T)
        ratio = self.beta + (1.0 - self.beta) * (t / T) ** self.gamma
        budget = int(round(self.full_budget * ratio))
        return max(self.min_budget, min(budget, self.full_budget))

    def get_io_ratio(self, epoch: int) -> float:
        """Fraction of full I/O used at this epoch."""
        return self.get_budget(epoch) / self.full_budget

    def summary(self) -> str:
        if self._total_epochs is None:
            return "BudgetScheduler not fitted."
        b_start = self.get_budget(0)
        b_end = self.get_budget(self._total_epochs - 1)
        avg_io = np.mean([self.get_io_ratio(e) for e in range(self._total_epochs)])
        lines = [
            f"TDCF Budget Schedule ({self._total_epochs} epochs, "
            f"fitted from {self._pilot_epochs} pilot epochs)",
            f"  β = {self.beta:.2f}, γ = {self.gamma:.1f}, k_low = {self.k_low}",
            f"  Budget: {b_start} → {b_end} band-slots "
            f"(of {self.full_budget} full)",
            f"  I/O ratio: {self.get_io_ratio(0):.1%} → {self.get_io_ratio(self._total_epochs - 1):.1%}",
            f"  Average I/O: {avg_io:.1%}",
        ]
        return "\n".join(lines)
