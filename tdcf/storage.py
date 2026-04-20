"""
Band-sharded on-disk storage for TDCF coefficient I/O measurement.

Stores DCT coefficients grouped by frequency band so that the dataloader
can physically skip high-frequency bands when K < L. Each band is a
separate numpy memmap file — only the opened bands cost I/O.

Storage layout:
    <root>/
        metadata.json       # dataset shape, band structure, byte sizes
        labels.npy           # (N,) int64
        band_00.npy          # (N, C, coeffs_in_band) float32  — memmap
        band_01.npy
        ...
        band_{L-1}.npy

Design rationale (Section 5 of the paper):
  - Each band file is independently memory-mappable
  - Reading K bands = touching only K band shards in the loader
  - Metadata records both payload bytes and exact on-disk file sizes
  - Requested-byte accounting is exact at the coefficient payload level
"""

import os
import json
import numpy as np
import torch

from .transforms import dct2d, idct2d, build_frequency_bands, build_nested_masks, zigzag_order


def build_band_store(
    dataset,
    root: str,
    H: int,
    W: int,
    num_bands: int,
    device: torch.device = None,
    batch_size: int = 1024,
    num_workers: int = 0,
):
    """
    Convert a torchvision dataset into band-sharded on-disk storage.

    Args:
        dataset:     torchvision dataset returning (image_tensor, label).
        root:        Directory to write the sharded files into.
        H, W:        Spatial dimensions of the images.
        num_bands:   Number of frequency bands (L).
        device:      GPU device for DCT computation.
        batch_size:  Batch size for DCT precomputation.
        num_workers: DataLoader workers for raw image loading.

    Returns:
        metadata dict with shapes and byte sizes per band.
    """
    from torch.utils.data import DataLoader

    os.makedirs(root, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Figure out band → flat-index mapping
    bands = build_frequency_bands(H, W, num_bands)
    band_sizes = [len(b) for b in bands]
    # Convert band indices to numpy for fast scatter
    band_indices_np = [b.numpy() for b in bands]

    # 2) First pass: compute all DCT coefficients and collect labels
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    N = len(dataset)
    sample = dataset[0][0]
    C_ch = sample.shape[0]

    print(f"[storage] Building band store: N={N}, C={C_ch}, H={H}, W={W}, "
          f"bands={num_bands}")

    # 3) Create memmap files for each band
    #    Shape per band: (N, C, coeffs_in_band)
    band_mmaps = []
    band_payload_byte_sizes = []
    band_paths = []
    for b_idx, bsize in enumerate(band_sizes):
        path = os.path.join(root, f"band_{b_idx:02d}.npy")
        mm = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float32,
            shape=(N, C_ch, bsize),
        )
        band_mmaps.append(mm)
        band_paths.append(path)
        band_payload_byte_sizes.append(N * C_ch * bsize * 4)  # float32 = 4 bytes

    # Labels
    labels_all = np.empty(N, dtype=np.int64)

    # 4) Fill the memmaps
    offset = 0
    with torch.no_grad():
        for images, labels_batch in loader:
            images = images.to(device)
            coeffs = dct2d(images)  # (B, C, H, W)
            B = coeffs.shape[0]

            # Flatten spatial dims → (B, C, H*W)
            coeffs_flat = coeffs.reshape(B, C_ch, H * W).cpu().numpy()

            for b_idx, indices in enumerate(band_indices_np):
                # Gather this band's coefficients: (B, C, band_size)
                band_mmaps[b_idx][offset:offset + B] = coeffs_flat[:, :, indices]

            labels_all[offset:offset + B] = labels_batch.numpy()
            offset += B

            if offset % (batch_size * 10) == 0 or offset >= N:
                print(f"  [storage] {offset}/{N} samples processed")

    # 5) Flush memmaps
    for mm in band_mmaps:
        mm.flush()
        del mm

    # 6) Save labels
    labels_path = os.path.join(root, "labels.npy")
    np.save(labels_path, labels_all)

    # 7) Save metadata
    band_file_byte_sizes = [os.path.getsize(path) for path in band_paths]
    labels_byte_size = os.path.getsize(labels_path)
    total_payload_bytes = sum(band_payload_byte_sizes)
    total_store_bytes = sum(band_file_byte_sizes) + labels_byte_size
    metadata = {
        "N": N,
        "C": C_ch,
        "H": H,
        "W": W,
        "num_bands": num_bands,
        "band_sizes": band_sizes,
        "band_payload_byte_sizes": band_payload_byte_sizes,
        "band_file_byte_sizes": band_file_byte_sizes,
        "labels_byte_size": labels_byte_size,
        "total_payload_bytes": total_payload_bytes,
        "total_store_bytes": total_store_bytes,
        "band_indices": [b.tolist() for b in band_indices_np],
        "bytes_per_sample_full": C_ch * H * W * 4,
    }
    with open(os.path.join(root, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[storage] Done. {num_bands} band files, "
          f"{total_store_bytes / 1e6:.1f} MB total on disk, "
          f"saved to {root}/")

    return metadata


def build_band_store_from_tensor(
    coeffs: torch.Tensor,
    labels: torch.Tensor,
    root: str,
    num_bands: int = 16,
):
    """
    Build band-sharded storage directly from precomputed coefficient tensors.

    This avoids recomputing DCT, guaranteeing bit-exact coefficients between
    the pilot phase (which uses the same tensor) and the adaptive band-store
    phase.  Eliminates float32 rounding divergence that can cause NaN.

    Args:
        coeffs: (N, C, H, W) float32 DCT coefficient tensor (GPU or CPU).
        labels: (N,) int64 label tensor.
        root:   Directory to write the sharded files into.
        num_bands: Number of frequency bands (L).

    Returns:
        metadata dict.
    """
    os.makedirs(root, exist_ok=True)

    N, C_ch, H, W = coeffs.shape
    bands = build_frequency_bands(H, W, num_bands)
    band_sizes = [len(b) for b in bands]
    band_indices_np = [b.numpy() for b in bands]

    coeffs_flat = coeffs.reshape(N, C_ch, H * W).cpu().numpy()
    labels_np = labels.cpu().numpy().astype(np.int64)

    print(f"[storage] Building band store from tensor: N={N}, C={C_ch}, "
          f"H={H}, W={W}, bands={num_bands}")

    band_paths = []
    band_payload_byte_sizes = []
    for b_idx, bsize in enumerate(band_sizes):
        path = os.path.join(root, f"band_{b_idx:02d}.npy")
        band_data = coeffs_flat[:, :, band_indices_np[b_idx]].copy()
        np.save(path, band_data)
        band_paths.append(path)
        band_payload_byte_sizes.append(N * C_ch * bsize * 4)

    labels_path = os.path.join(root, "labels.npy")
    np.save(labels_path, labels_np)

    band_file_byte_sizes = [os.path.getsize(p) for p in band_paths]
    labels_byte_size = os.path.getsize(labels_path)
    total_payload_bytes = sum(band_payload_byte_sizes)
    total_store_bytes = sum(band_file_byte_sizes) + labels_byte_size
    metadata = {
        "N": N, "C": C_ch, "H": H, "W": W,
        "num_bands": num_bands,
        "band_sizes": band_sizes,
        "band_payload_byte_sizes": band_payload_byte_sizes,
        "band_file_byte_sizes": band_file_byte_sizes,
        "labels_byte_size": labels_byte_size,
        "total_payload_bytes": total_payload_bytes,
        "total_store_bytes": total_store_bytes,
        "band_indices": [b.tolist() for b in band_indices_np],
        "bytes_per_sample_full": C_ch * H * W * 4,
    }
    with open(os.path.join(root, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[storage] Done. {num_bands} band files, "
          f"{total_store_bytes / 1e6:.1f} MB total, saved to {root}/")
    return metadata


def build_patch_residual_store_from_tensor(
    coeffs: torch.Tensor,
    root: str,
    k_values,
    patch_size: int,
    num_bands: int = 16,
    chunk_size: int = 1024,
):
    """
    Build patch-sharded residual storage for the K values actually used.

    For each requested K, this stores the spatial residual
    `x_full - x_base(K)` patch-by-patch so the physical dataloader can
    reconstruct the same served image as TDCFServer without reading the
    dropped high-frequency coefficient bands.
    """
    os.makedirs(root, exist_ok=True)
    k_values = sorted({int(k) for k in k_values if int(k) < num_bands})
    if not k_values:
        metadata = {
            "N": int(coeffs.shape[0]),
            "C": int(coeffs.shape[1]),
            "H": int(coeffs.shape[2]),
            "W": int(coeffs.shape[3]),
            "patch_size": int(patch_size),
            "nph": int(coeffs.shape[2] // patch_size),
            "npw": int(coeffs.shape[3] // patch_size),
            "num_patches": int((coeffs.shape[2] // patch_size) * (coeffs.shape[3] // patch_size)),
            "k_values": [],
            "residuals": {},
        }
        with open(os.path.join(root, "residual_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        return metadata

    N, C_ch, H, W = coeffs.shape
    nph = H // patch_size
    npw = W // patch_size
    num_patches = nph * npw
    patch_shape = (N, C_ch, patch_size, patch_size)
    device = coeffs.device

    masks = build_nested_masks(H, W, num_bands)
    mask_cache = {
        K: masks[K - 1].to(device).unsqueeze(0).unsqueeze(0)
        for K in k_values
    }

    patch_mmaps = {}
    residuals_meta = {}
    print(f"[storage] Building residual patch store: N={N}, C={C_ch}, "
          f"H={H}, W={W}, patch={patch_size}, K={k_values}")

    for K in k_values:
        k_dir = os.path.join(root, f"residual_k{K:02d}")
        os.makedirs(k_dir, exist_ok=True)
        patch_files = []
        patch_payload_byte_sizes = []
        patch_mmaps[K] = []
        for p_idx in range(num_patches):
            rel_path = os.path.join(f"residual_k{K:02d}", f"patch_{p_idx:02d}.npy")
            abs_path = os.path.join(root, rel_path)
            mm = np.lib.format.open_memmap(
                abs_path,
                mode="w+",
                dtype=np.float32,
                shape=patch_shape,
            )
            patch_mmaps[K].append(mm)
            patch_files.append(rel_path)
            patch_payload_byte_sizes.append(N * C_ch * patch_size * patch_size * 4)
        residuals_meta[str(K)] = {
            "patch_files": patch_files,
            "patch_payload_byte_sizes": patch_payload_byte_sizes,
        }

    with torch.no_grad():
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            coeff_chunk = coeffs[start:end].to(device)
            x_full = idct2d(coeff_chunk)
            B = end - start

            for K in k_values:
                x_base = idct2d(coeff_chunk * mask_cache[K])
                residual = x_full - x_base
                residual_patches = (
                    residual[:, :, :nph * patch_size, :npw * patch_size]
                    .reshape(B, C_ch, nph, patch_size, npw, patch_size)
                    .permute(0, 2, 4, 1, 3, 5)
                    .reshape(B, num_patches, C_ch, patch_size, patch_size)
                    .cpu()
                    .numpy()
                )
                for p_idx in range(num_patches):
                    patch_mmaps[K][p_idx][start:end] = residual_patches[:, p_idx]

            if start == 0 or end == N or ((end // chunk_size) % 10 == 0):
                print(f"  [storage] residual {end}/{N} samples processed")

    for K in k_values:
        patch_file_byte_sizes = []
        total_payload_bytes = 0
        for p_idx, mm in enumerate(patch_mmaps[K]):
            mm.flush()
            del mm
            rel_path = residuals_meta[str(K)]["patch_files"][p_idx]
            abs_path = os.path.join(root, rel_path)
            patch_file_byte_sizes.append(os.path.getsize(abs_path))
            total_payload_bytes += residuals_meta[str(K)]["patch_payload_byte_sizes"][p_idx]
        residuals_meta[str(K)]["patch_file_byte_sizes"] = patch_file_byte_sizes
        residuals_meta[str(K)]["total_payload_bytes"] = total_payload_bytes
        residuals_meta[str(K)]["total_store_bytes"] = sum(patch_file_byte_sizes)

    metadata = {
        "N": N,
        "C": C_ch,
        "H": H,
        "W": W,
        "patch_size": patch_size,
        "nph": nph,
        "npw": npw,
        "num_patches": num_patches,
        "k_values": k_values,
        "residuals": residuals_meta,
    }
    with open(os.path.join(root, "residual_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    total_store_bytes = sum(
        residuals_meta[str(K)]["total_store_bytes"] for K in k_values
    )
    print(f"[storage] Done. residual patches for K={k_values}, "
          f"{total_store_bytes / 1e6:.1f} MB total, saved to {root}/")
    return metadata


def build_block_band_store_from_tensor(
    coeffs: torch.Tensor,
    labels: torch.Tensor,
    root: str,
    nph: int,
    npw: int,
    num_bands_per_patch: int = 16,
):
    """
    Build block-DCT band-sharded storage from precomputed coefficients.

    Storage layout:
        <root>/
            metadata.json
            labels.npy
            patch_00.npy   # (N, C, bs*bs) coefficients in zig-zag order
            patch_01.npy
            ...
            patch_{P-1}.npy

    Each patch file stores ALL coefficients for that patch in zig-zag
    (low→high frequency) order.  The loader reads only the first
    K * band_size coefficients to truncate high frequencies.

    Args:
        coeffs: (N, P, C, bs_h, bs_w) block-DCT coefficients.
        labels: (N,) int64 label tensor.
        root:   Output directory.
        nph, npw: patch grid dimensions.
        num_bands_per_patch: Number of frequency bands per patch.
    """
    os.makedirs(root, exist_ok=True)

    N, P, C_ch, bs_h, bs_w = coeffs.shape
    total_coeffs = bs_h * bs_w
    assert P == nph * npw

    # Zig-zag ordering within each block
    zz = zigzag_order(bs_h, bs_w).numpy()
    band_size = total_coeffs // num_bands_per_patch
    band_boundaries = [b * band_size for b in range(num_bands_per_patch + 1)]
    band_boundaries[-1] = total_coeffs  # handle remainder

    # Flatten spatial dims of each patch and reorder to zig-zag
    coeffs_flat = coeffs.reshape(N, P, C_ch, total_coeffs).cpu().numpy()
    coeffs_zz = coeffs_flat[:, :, :, zz]  # (N, P, C, total_coeffs) in zz order

    labels_np = labels.cpu().numpy().astype(np.int64)

    print(f"[storage] Building block-DCT store: N={N}, P={P}, C={C_ch}, "
          f"block={bs_h}x{bs_w}, bands_per_patch={num_bands_per_patch}")

    patch_paths = []
    patch_payload_byte_sizes = []
    for p_idx in range(P):
        path = os.path.join(root, f"patch_{p_idx:02d}.npy")
        np.save(path, coeffs_zz[:, p_idx].copy())  # (N, C, total_coeffs)
        patch_paths.append(path)
        patch_payload_byte_sizes.append(N * C_ch * total_coeffs * 4)

    labels_path = os.path.join(root, "labels.npy")
    np.save(labels_path, labels_np)

    # Per-sample per-patch signal energy for sample-adaptive q selection.
    # patch_signal[n, p] = mean(|coeffs[n, p, :, :]|) across channels & freqs.
    # This is tiny (~N*P*4 bytes ≈ 3 MB) and enables the loader to shift
    # the q important patches to wherever the object actually is in each image.
    patch_signal = np.abs(coeffs_flat).mean(axis=(2, 3))   # (N, P)
    patch_signal_path = os.path.join(root, "patch_signal.npy")
    np.save(patch_signal_path, patch_signal.astype(np.float32))

    patch_file_byte_sizes = [os.path.getsize(p) for p in patch_paths]
    labels_byte_size = os.path.getsize(labels_path)
    patch_signal_byte_size = os.path.getsize(patch_signal_path)
    total_payload_bytes = sum(patch_payload_byte_sizes)
    total_store_bytes = sum(patch_file_byte_sizes) + labels_byte_size + patch_signal_byte_size

    metadata = {
        "format": "block_dct",
        "N": N, "C": C_ch,
        "block_size_h": bs_h, "block_size_w": bs_w,
        "nph": nph, "npw": npw, "P": P,
        "total_coeffs_per_patch": total_coeffs,
        "num_bands_per_patch": num_bands_per_patch,
        "band_size": band_size,
        "band_boundaries": band_boundaries,
        "zigzag_order": zz.tolist(),
        "patch_payload_byte_sizes": patch_payload_byte_sizes,
        "patch_file_byte_sizes": patch_file_byte_sizes,
        "labels_byte_size": labels_byte_size,
        "patch_signal_byte_size": patch_signal_byte_size,
        "total_payload_bytes": total_payload_bytes,
        "total_store_bytes": total_store_bytes,
        "bytes_per_sample_full": P * C_ch * total_coeffs * 4,
    }
    with open(os.path.join(root, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[storage] Done. {P} patch files, "
          f"{total_store_bytes / 1e6:.1f} MB total, saved to {root}/")
    return metadata


def get_band_byte_sizes(root: str) -> dict:
    """Load metadata and return per-band byte sizes."""
    with open(os.path.join(root, "metadata.json")) as f:
        return json.load(f)
