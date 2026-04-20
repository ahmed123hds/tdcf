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
