"""
Gradient-based sensitivity estimation for TDCF.

Implements:
  - Instantaneous input sensitivity spectrum  Φ_t(k)     (Eq. 7)
  - Epoch-aggregated sensitivity              Φ̄_e(k)     (Eq. 8)
  - Normalized distribution                   p_e(k)      (Eq. 9)
  - Band sensitivity                          s_e(b)      (Eq. 10)
  - Spatial residual sensitivity              g_e(p)      (Eq. 11)

All accumulators are kept on GPU for zero-copy performance.
"""

import torch
import torch.nn as nn
import numpy as np
from .transforms import dct2d, idct2d, build_frequency_bands


class SensitivityEstimator:
    """
    Estimates gradient sensitivity of DCT coefficients and spatial patches
    during the pilot training phase.  All GPU-resident.
    """

    def __init__(self, H: int, W: int, num_bands: int,
                 patch_size: int = 7, device: torch.device = None,
                 eps: float = 1e-8):
        self.H = H
        self.W = W
        self.M = H * W
        self.num_bands = num_bands
        self.patch_size = patch_size
        self.device = device or torch.device('cpu')
        self.eps = eps

        # Frequency band structure
        self.bands = build_frequency_bands(H, W, num_bands)

        # Patch grid
        self.nph = H // patch_size
        self.npw = W // patch_size
        self.P = self.nph * self.npw

        # GPU accumulators
        self.reset_epoch()

        # Epoch histories (kept as numpy for schedule fitting)
        self.band_sensitivity_history = []
        self.patch_sensitivity_history = []
        self.coeff_sensitivity_history = []

    def reset_epoch(self):
        """Reset GPU accumulators for a new epoch."""
        self._phi_accum = torch.zeros(self.M, device=self.device)
        self._patch_accum = torch.zeros(self.P, device=self.device)
        self._step_count = 0

    # ------------------------------------------------------------------
    # Eq. 7: Instantaneous input sensitivity spectrum
    # ------------------------------------------------------------------

    def measure_coefficient_sensitivity(
        self, images: torch.Tensor, labels: torch.Tensor,
        model: nn.Module, criterion: nn.Module
    ) -> torch.Tensor:
        """
        Compute Φ_t(k) = E_batch[ |∂L/∂c_k| ].
        All computation stays on GPU.
        """
        coeffs = dct2d(images)
        return self.measure_coefficient_sensitivity_from_coeffs(
            coeffs, labels, model, criterion)

    def measure_coefficient_sensitivity_from_coeffs(
        self, coeffs: torch.Tensor, labels: torch.Tensor,
        model: nn.Module, criterion: nn.Module
    ) -> torch.Tensor:
        """
        Compute Φ_t(k) directly from DCT coefficients.

        This avoids an extra DCT pass during the pilot phase, where the
        training loop already materializes coefficient batches.

        For multi-channel inputs, per-channel coefficient gradients are
        aggregated into a single frequency-location sensitivity map so the
        band structure remains defined over spatial frequencies rather than
        channel-frequency tuples.
        """
        coeffs_leaf = coeffs.detach().clone().requires_grad_(True)
        x_recon = idct2d(coeffs_leaf)

        logits = model(x_recon)
        loss = criterion(logits, labels)
        grad_coeffs = torch.autograd.grad(
            loss, coeffs_leaf, retain_graph=False, create_graph=False
        )[0]

        # Φ_t(k) = batch-mean of channel-aggregated |∂L/∂c_k|
        phi_t = grad_coeffs.abs().sum(dim=1).mean(dim=0).reshape(-1)

        self._phi_accum += phi_t.detach()
        self._step_count += 1

        return phi_t.detach()

    # ------------------------------------------------------------------
    # Eq. 11: Spatial residual sensitivity
    # ------------------------------------------------------------------

    def measure_patch_sensitivity(
        self, images: torch.Tensor, x_base: torch.Tensor,
        labels: torch.Tensor, model: nn.Module, criterion: nn.Module
    ) -> torch.Tensor:
        """
        Compute g_e(p) = E[ |∂L/∂r_p|₁ ] for spatial residual patches.
        Works for both grayscale (C=1, MNIST) and RGB (C=3, CIFAR-100).
        Gradient magnitude is summed over all channels per patch.
        """
        residual = (images - x_base).detach().clone().requires_grad_(True)
        x_input = x_base.detach() + residual

        logits = model(x_input)
        loss = criterion(logits, labels)
        grad_r = torch.autograd.grad(
            loss, residual, retain_graph=False, create_graph=False
        )[0]                                              # (B, C, H, W)
        B = grad_r.shape[0]
        ps = self.patch_size

        # Sum gradient magnitude over channels → (B, H, W)
        grad_r_agg = grad_r.abs().sum(dim=1)             # (B, H, W)

        # Reshape into patches → L1 norm per patch
        grad_patches = (
            grad_r_agg[:, :self.nph*ps, :self.npw*ps]
            .reshape(B, self.nph, ps, self.npw, ps)
            .permute(0, 1, 3, 2, 4)
            .reshape(B, self.P, ps * ps)
        )
        g_t = grad_patches.abs().sum(dim=-1).mean(dim=0)  # (P,) on GPU

        self._patch_accum += g_t.detach()
        return g_t.detach()

    # ------------------------------------------------------------------
    # Eq. 8-10: Epoch-level aggregation
    # ------------------------------------------------------------------

    def finalize_epoch(self):
        """
        After all batches, compute Φ̄_e, p_e, s_e, g̃_e.
        Moves results to CPU/numpy for schedule fitting.
        """
        if self._step_count == 0:
            return

        # Eq. 8
        phi_bar = (self._phi_accum / self._step_count).cpu()

        # Eq. 9
        p_e = phi_bar / (phi_bar.sum() + self.eps)

        # Eq. 10
        s_e = torch.zeros(self.num_bands)
        for b, band_idx in enumerate(self.bands):
            s_e[b] = p_e[band_idx].sum()

        # Patch sensitivity (normalized)
        g_bar = (self._patch_accum / self._step_count).cpu()
        g_tilde = g_bar / (g_bar.sum() + self.eps)

        self.coeff_sensitivity_history.append(phi_bar.numpy().copy())
        self.band_sensitivity_history.append(s_e.numpy().copy())
        self.patch_sensitivity_history.append(g_tilde.numpy().copy())

        self.reset_epoch()

    # ------------------------------------------------------------------
    # Eq. 12-13: Band cutoff
    # ------------------------------------------------------------------

    def compute_band_cutoff(self, epoch: int, eta_f: float = 0.9) -> int:
        """K_e = min{m : S_e(m) ≥ η_f}."""
        s_e = self.band_sensitivity_history[epoch]
        cumsum = np.cumsum(s_e)
        K_e = int(np.searchsorted(cumsum, eta_f)) + 1
        return min(K_e, self.num_bands)

    # ------------------------------------------------------------------
    # Eq. 14: Patch quota
    # ------------------------------------------------------------------

    def compute_patch_quota(self, epoch: int, eta_s: float = 0.8) -> int:
        """Smallest q_e covering ≥ η_s of patch sensitivity mass."""
        g_tilde = self.patch_sensitivity_history[epoch]
        sorted_g = np.sort(g_tilde)[::-1]
        cumsum = np.cumsum(sorted_g)
        q_e = int(np.searchsorted(cumsum, eta_s)) + 1
        return min(q_e, self.P)


# ---------------------------------------------------------------------------
# Block-DCT sensitivity estimator
# ---------------------------------------------------------------------------

class BlockSensitivityEstimator:
    """
    Gradient sensitivity estimator for block-DCT coefficients.

    Measures per-patch importance (for q selection) and per-band importance
    within blocks (for K_high selection).  Compatible with FidelityScheduler.
    """

    def __init__(self, block_size: int, nph: int, npw: int,
                 num_bands: int = 16, device: torch.device = None,
                 eps: float = 1e-8):
        self.block_size = block_size
        self.nph = nph
        self.npw = npw
        self.P = nph * npw
        self.num_bands = num_bands
        self.device = device or torch.device('cpu')
        self.eps = eps

        # Band structure within each block (zig-zag order)
        self.bands = build_frequency_bands(block_size, block_size, num_bands)

        # GPU accumulators
        self.reset_epoch()

        # Histories (numpy, for schedule fitting)
        self.band_sensitivity_history = []
        self.patch_sensitivity_history = []
        self.coeff_sensitivity_history = []

    def reset_epoch(self):
        self._patch_accum = torch.zeros(self.P, device=self.device)
        self._band_accum = torch.zeros(self.num_bands, device=self.device)
        self._step_count = 0

    def measure_sensitivity(self, coeffs, labels, model, criterion):
        """
        Measure gradient sensitivity from block-DCT coefficients.

        Args:
            coeffs: (B, P, C, bs, bs) block-DCT coefficients on GPU.
            labels: (B,) labels on GPU.
            model: the classification model.
            criterion: loss function.
        """
        from .transforms import block_idct2d

        coeffs_leaf = coeffs.detach().clone().requires_grad_(True)
        x_recon = block_idct2d(coeffs_leaf, self.nph, self.npw)
        logits = model(x_recon)
        loss = criterion(logits, labels)
        grad = torch.autograd.grad(
            loss, coeffs_leaf, retain_graph=False, create_graph=False
        )[0]   # (B, P, C, bs, bs)

        # Per-patch sensitivity: aggregate |grad| over (C, bs, bs), mean batch
        patch_sens = grad.abs().sum(dim=(2, 3, 4)).mean(dim=0)   # (P,)
        self._patch_accum += patch_sens.detach()

        # Per-band sensitivity: aggregate over (batch, patches, channels)
        # grad_flat: (bs*bs,) mean over batch, patches, channels
        grad_flat = grad.abs().mean(dim=(0, 1)).sum(dim=0).reshape(-1)
        for b, band_idx in enumerate(self.bands):
            self._band_accum[b] += grad_flat[band_idx.to(self.device)].sum()

        self._step_count += 1

    def finalize_epoch(self):
        if self._step_count == 0:
            return

        # Normalized per-band sensitivity
        band_bar = (self._band_accum / self._step_count).cpu()
        s_e = band_bar / (band_bar.sum() + self.eps)

        # Normalized per-patch sensitivity
        patch_bar = (self._patch_accum / self._step_count).cpu()
        g_tilde = patch_bar / (patch_bar.sum() + self.eps)

        self.band_sensitivity_history.append(s_e.numpy().copy())
        self.patch_sensitivity_history.append(g_tilde.numpy().copy())

        self.reset_epoch()

    # --- FidelityScheduler-compatible interface ---

    def compute_band_cutoff(self, epoch: int, eta_f: float = 0.9) -> int:
        """K_e = min{m : cumsum(s_e) >= eta_f}."""
        s_e = self.band_sensitivity_history[epoch]
        cumsum = np.cumsum(s_e)
        K_e = int(np.searchsorted(cumsum, eta_f)) + 1
        return min(K_e, self.num_bands)

    def compute_patch_quota(self, epoch: int, eta_s: float = 0.8) -> int:
        """Smallest q_e covering >= eta_s of patch sensitivity mass."""
        g_tilde = self.patch_sensitivity_history[epoch]
        sorted_g = np.sort(g_tilde)[::-1]
        cumsum = np.cumsum(sorted_g)
        q_e = int(np.searchsorted(cumsum, eta_s)) + 1
        return min(q_e, self.P)
