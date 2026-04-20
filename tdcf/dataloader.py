"""
TDCF-aware data loader for batched grayscale or RGB inputs.

Instead of computing DCT per-item in __getitem__ (slow), this uses
pre-computed DCT coefficients on GPU and applies masks at batch level
via a custom collate/serving function.

Serves x^{K(e),q(e)}_e per Eq. 15 of the paper.
"""

import torch
import numpy as np
from torch.utils.data import Dataset

from .transforms import idct2d, build_nested_masks


class PrecomputedTDCFDataset(Dataset):
    """
    Thin wrapper around pre-computed DCT coefficients.
    Returns raw coefficients; the fidelity truncation happens
    in the GPU-batched serving step (TDCFServer).
    """

    def __init__(self, dct_coeffs: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            dct_coeffs: (N, C, H, W) DCT coefficients (GPU tensor).
            labels:     (N,) labels (GPU tensor).
        """
        self.coeffs = dct_coeffs
        self.labels = labels

    def __len__(self):
        return self.coeffs.shape[0]

    def __getitem__(self, idx):
        return self.coeffs[idx], self.labels[idx]


class TDCFServer:
    """
    GPU-batched fidelity serving.  Given a batch of DCT coefficients,
    applies the current epoch's (K, q) schedule to produce served images.

    All operations are batched on GPU — no per-item overhead.
    """

    def __init__(self, H: int = 28, W: int = 28, num_bands: int = 8,
                 patch_size: int = 7, device: torch.device = None):
        self.H = H
        self.W = W
        self.num_bands = num_bands
        self.patch_size = patch_size
        self.device = device or torch.device('cuda')

        # Build and cache all nested masks on GPU
        self.masks = build_nested_masks(H, W, num_bands)
        self.masks_gpu = [m.to(self.device) for m in self.masks]

        # Patch grid
        self.nph = H // patch_size
        self.npw = W // patch_size
        self.P = self.nph * self.npw

        # Current fidelity state
        self.current_K = num_bands
        self.current_q = self.P
        self._patch_keep_mask = None  # boolean mask of patches to keep

    def set_fidelity(self, K: int, q: int,
                     patch_sensitivity: np.ndarray = None):
        """
        Update fidelity for the current epoch.

        Args:
            K: Band cutoff (1..num_bands).
            q: Patch quota  (1..P).
            patch_sensitivity: (P,) array for selecting top-q patches.
        """
        self.current_K = min(K, self.num_bands)
        self.current_q = min(q, self.P)

        if patch_sensitivity is not None and q < self.P:
            top_idx = np.argsort(patch_sensitivity)[::-1][:q]
            keep_mask = torch.zeros(self.P, device=self.device,
                                    dtype=torch.bool)
            keep_mask[torch.from_numpy(top_idx.copy()).to(self.device)] = True
            self._patch_keep_mask = keep_mask
        else:
            self._patch_keep_mask = None

    @torch.no_grad()
    def serve(self, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Apply fidelity truncation to a batch of DCT coefficients.

        Args:
            coeffs: (B, C, H, W) DCT coefficients on GPU.
        Returns:
            (B, C, H, W) fidelity-served spatial images on GPU.
        """
        # Fast path: either all bands are kept or all residual patches are
        # restored, both of which recover the full-fidelity image.
        if self.current_K >= self.num_bands or self.current_q >= self.P:
            return idct2d(coeffs)

        B = coeffs.shape[0]
        C_ch = coeffs.shape[1]
        ps = self.patch_size

        # --- Frequency truncation ---
        level = self.current_K - 1
        mask_freq = self.masks_gpu[level]              # (H, W)
        c_base = coeffs * mask_freq.unsqueeze(0).unsqueeze(0)
        x_base = idct2d(c_base)                        # low-freq base

        # --- Spatial residual selection ---
        x_full = idct2d(coeffs)
        residual = x_full - x_base                     # (B,C,H,W)

        # Extract patches
        r_patches = (
            residual[:, :, :self.nph*ps, :self.npw*ps]
            .reshape(B, C_ch, self.nph, ps, self.npw, ps)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(B, self.P, C_ch, ps, ps)
        )

        # Zero out non-selected patches
        if self._patch_keep_mask is not None:
            keep_mask = self._patch_keep_mask
        else:
            # Fallback: keep highest-energy patches per batch
            energy = r_patches.reshape(B, self.P, -1).pow(2).sum(dim=-1)
            avg_energy = energy.mean(dim=0)            # (P,)
            _, top_idx = avg_energy.topk(self.current_q)
            keep_mask = torch.zeros(self.P, device=self.device,
                                    dtype=torch.bool)
            keep_mask[top_idx] = True

        r_patches[:, ~keep_mask] = 0.0

        # Reconstruct spatial residual
        r_spatial = (
            r_patches
            .reshape(B, self.nph, self.npw, C_ch, ps, ps)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(B, C_ch, self.nph * ps, self.npw * ps)
        )

        out = x_base.clone()
        out[:, :, :self.nph*ps, :self.npw*ps] += r_spatial
        return out

    def get_bytes_ratio(self) -> float:
        """
        Heuristic fraction of nominal full-fidelity content served.

        This is a schedule-level proxy, not a measurement of actual storage I/O.
        """
        freq_ratio = self.current_K / self.num_bands
        if freq_ratio >= 1.0:
            return 1.0
        spatial_ratio = self.current_q / self.P
        return freq_ratio + (1.0 - freq_ratio) * spatial_ratio
