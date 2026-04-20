"""
Multi-fidelity transform module for TDCF.

Implements block-DCT based frequency decomposition (Section 4.2 of the paper)
and the nested mask family M^(0) ⊆ M^(1) ⊆ ... ⊆ M^(L) (Eq. 3-5).

For MNIST (28×28 grayscale), we use a full-image DCT rather than block-DCT
since the images are small enough to handle globally.

GPU-optimized: DCT matrices are cached per-device, all ops are batched.
"""

import torch
import numpy as np


# ---------------------------------------------------------------------------
# 2-D DCT / iDCT  (Type-II / Type-III, orthonormal) — GPU-cached
# ---------------------------------------------------------------------------

_DCT_CACHE = {}  # (N, device, dtype) -> matrix


def _get_dct_matrix(N: int, device: torch.device,
                    dtype: torch.dtype) -> torch.Tensor:
    """Build or retrieve cached NxN orthonormal DCT-II matrix."""
    key = (N, device, dtype)
    if key not in _DCT_CACHE:
        n = torch.arange(N, dtype=torch.float64)
        k = torch.arange(N, dtype=torch.float64)
        C = torch.cos(
            torch.pi * k.unsqueeze(1) * (2 * n.unsqueeze(0) + 1) / (2 * N)
        )
        C *= np.sqrt(2.0 / N)
        C[0, :] *= 1.0 / np.sqrt(2.0)
        _DCT_CACHE[key] = C.to(device=device, dtype=dtype)
    return _DCT_CACHE[key]


def dct2d(x: torch.Tensor) -> torch.Tensor:
    """
    Batched 2-D DCT-II (orthonormal).
    Forward:  C = C_h @ X @ C_w^T

    Args:
        x: (B, 1, H, W) tensor on any device.
    Returns:
        DCT coefficients with same shape, device, dtype.
    """
    B, C_ch, H, W = x.shape
    Ch = _get_dct_matrix(H, x.device, x.dtype)   # (H, H)
    Cw = _get_dct_matrix(W, x.device, x.dtype)   # (W, W)
    # x viewed as (B*C, H, W)
    xr = x.reshape(B * C_ch, H, W)
    # Ch @ xr  → (B*C, H, W),  then  result @ Cw^T → (B*C, H, W)
    out = Ch @ xr @ Cw.T
    return out.reshape(B, C_ch, H, W)


def idct2d(c: torch.Tensor) -> torch.Tensor:
    """
    Batched 2-D inverse DCT (Type-III, orthonormal).
    Inverse:  X = C_h^T @ C @ C_w   (orthonormal → C^{-1} = C^T)

    Args:
        c: (B, 1, H, W) DCT coefficient tensor.
    Returns:
        Reconstructed spatial images.
    """
    B, C_ch, H, W = c.shape
    Ch = _get_dct_matrix(H, c.device, c.dtype)
    Cw = _get_dct_matrix(W, c.device, c.dtype)
    cr = c.reshape(B * C_ch, H, W)
    out = Ch.T @ cr @ Cw
    return out.reshape(B, C_ch, H, W)


# ---------------------------------------------------------------------------
# Block-DCT / iDCT  (JPEG-style, spatially localized frequencies)
# ---------------------------------------------------------------------------

def block_dct2d(x: torch.Tensor, block_size: int = 8) -> torch.Tensor:
    """
    Block-DCT: divide image into non-overlapping blocks, DCT each.

    Unlike global DCT, this keeps frequency information spatially localized
    (exactly like JPEG), enabling per-patch frequency band selection.

    Args:
        x: (B, C, H, W) spatial images.
        block_size: size of each square block (default 8).
    Returns:
        (B, P, C, block_size, block_size) DCT coefficients per patch,
        where P = (H // block_size) * (W // block_size).
    """
    B, C, H, W = x.shape
    nph = H // block_size
    npw = W // block_size
    P = nph * npw

    blocks = x[:, :, :nph * block_size, :npw * block_size]
    blocks = (blocks
              .reshape(B, C, nph, block_size, npw, block_size)
              .permute(0, 2, 4, 1, 3, 5)
              .reshape(B * P, C, block_size, block_size))

    coeffs = dct2d(blocks)
    return coeffs.reshape(B, P, C, block_size, block_size)


def block_idct2d(coeffs: torch.Tensor,
                 nph: int, npw: int) -> torch.Tensor:
    """
    Block-iDCT: reconstruct spatial image from per-patch DCT coefficients.

    Args:
        coeffs: (B, P, C, block_size, block_size) coefficients.
        nph: number of patch rows.
        npw: number of patch columns.
    Returns:
        (B, C, H, W) spatial images.
    """
    B, P, C, bs_h, bs_w = coeffs.shape
    blocks = coeffs.reshape(B * P, C, bs_h, bs_w)
    spatial = idct2d(blocks)
    spatial = (spatial
               .reshape(B, nph, npw, C, bs_h, bs_w)
               .permute(0, 3, 1, 4, 2, 5)
               .reshape(B, C, nph * bs_h, npw * bs_w))
    return spatial


@torch.no_grad()
def precompute_block_dct_dataset(dataset, device: torch.device,
                                 block_size: int = 8,
                                 batch_size: int = 1024,
                                 num_workers: int = 0):
    """
    Pre-compute block-DCT coefficients for the entire dataset on GPU.

    Returns:
        coeffs: (N, P, C, bs, bs) tensor on device.
        labels: (N,) tensor on device.
    """
    from torch.utils.data import DataLoader

    pin_memory = device.type == "cuda"
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_memory)

    all_coeffs, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        c = block_dct2d(images, block_size)
        all_coeffs.append(c)
        all_labels.append(labels.to(device))

    return torch.cat(all_coeffs, dim=0), torch.cat(all_labels, dim=0)


# ---------------------------------------------------------------------------
# Frequency band indexing  (zig-zag order for 2-D DCT)
# ---------------------------------------------------------------------------


def zigzag_order(H: int, W: int) -> torch.Tensor:
    """
    Return a (H*W,) tensor of flat indices in zig-zag scan order.
    Orders DCT coefficients from lowest to highest frequency.
    """
    indices = []
    for s in range(H + W - 1):
        if s % 2 == 0:
            for i in range(min(s, H - 1), max(0, s - W + 1) - 1, -1):
                indices.append(i * W + (s - i))
        else:
            for j in range(min(s, W - 1), max(0, s - H + 1) - 1, -1):
                indices.append((s - j) * W + j)
    return torch.tensor(indices, dtype=torch.long)


def build_frequency_bands(H: int, W: int, num_bands: int) -> list:
    """
    Partition zig-zag-ordered coefficients into `num_bands` bands.

    Returns:
        List of 1-D index tensors per band (band 0 = DC / lowest freq).
    """
    zz = zigzag_order(H, W)
    total = H * W
    band_size = total // num_bands
    bands = []
    for b in range(num_bands):
        start = b * band_size
        end = (b + 1) * band_size if b < num_bands - 1 else total
        bands.append(zz[start:end])
    return bands


# ---------------------------------------------------------------------------
# Nested fidelity masks  (Eq. 4)
# ---------------------------------------------------------------------------

def build_nested_masks(H: int, W: int, num_levels: int) -> list:
    """
    Build L nested masks M^(0) ⊆ ... ⊆ M^(L-1).

    Level 0 keeps only the lowest-frequency band.
    Level L-1 keeps all coefficients (full fidelity).

    Returns:
        List of (H, W) float tensors (0.0 / 1.0).
    """
    bands = build_frequency_bands(H, W, num_levels)
    masks = []
    cumulative = torch.zeros(H * W, dtype=torch.float32)
    for level in range(num_levels):
        cumulative[bands[level]] = 1.0
        masks.append(cumulative.clone().reshape(H, W))
    return masks


# ---------------------------------------------------------------------------
# Batch fidelity reconstruction  (Eq. 5) — fully GPU-resident
# ---------------------------------------------------------------------------

def reconstruct_at_fidelity(x: torch.Tensor, level: int,
                            masks: list) -> torch.Tensor:
    """
    Reconstruct an entire batch at a given fidelity level.

    Args:
        x:     (B, 1, H, W) spatial images (GPU).
        level: integer fidelity level in [0, L-1].
        masks: list of (H, W) masks (will be moved to x's device).
    Returns:
        (B, 1, H, W) fidelity-truncated reconstructions.
    """
    c = dct2d(x)
    mask = masks[level].to(c.device)
    return idct2d(c * mask.unsqueeze(0).unsqueeze(0))


# ---------------------------------------------------------------------------
# Pre-compute full dataset DCT  (MNIST fits in GPU memory)
# ---------------------------------------------------------------------------

@torch.no_grad()
def precompute_dct_dataset(dataset, device: torch.device,
                           batch_size: int = 1024,
                           num_workers: int = 0):
    """
    Pre-compute DCT coefficients for the entire dataset on GPU.

    Args:
        dataset:    MNIST dataset returning (image, label) tuples.
        device:     Target device.
        batch_size: Processing batch size.
        num_workers: Number of DataLoader workers used during precompute.
    Returns:
        coeffs: (N, 1, H, W) tensor of DCT coefficients on `device`.
        labels: (N,) tensor of labels on `device`.
    """
    from torch.utils.data import DataLoader

    pin_memory = device.type == "cuda"
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_memory)

    all_coeffs, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        c = dct2d(images)
        all_coeffs.append(c)
        all_labels.append(labels.to(device))

    return torch.cat(all_coeffs, dim=0), torch.cat(all_labels, dim=0)
