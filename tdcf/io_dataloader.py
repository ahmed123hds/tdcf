"""
Physical I/O-tracking dataloader for TDCF.

Reads band-sharded on-disk storage and tracks exact coefficient payload
bytes requested from the band shards. High-frequency bands are skipped
entirely when K < L.

Provides two metrics:
  1. bytes_requested  — exact coefficient bytes requested from band files
  2. wall_clock       — end-to-end epoch time

Usage:
    store = BandShardedStore("path/to/store")
    loader = store.get_loader(batch_size=256, shuffle=True)

    store.set_fidelity(K=9)
    for coeffs_partial, labels in loader:
        images = idct2d(coeffs_partial)   # truncated reconstruction
        ...
    print(f"Bytes read this epoch: {store.bytes_read_epoch}")
"""

import os
import json
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from .transforms import idct2d


class BandShardedStore:
    """
    Manages band-sharded on-disk storage with I/O tracking.

    Each frequency band is stored as a separate numpy memmap file.
    Setting K controls how many band files are actually opened and read.
    """

    def __init__(self, root: str, device: torch.device = None):
        self.root = root
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load metadata
        with open(os.path.join(root, "metadata.json")) as f:
            self.meta = json.load(f)

        self.N = self.meta["N"]
        self.C = self.meta["C"]
        self.H = self.meta["H"]
        self.W = self.meta["W"]
        self.num_bands = self.meta["num_bands"]
        self.band_sizes = self.meta["band_sizes"]
        self.band_payload_byte_sizes = self.meta["band_payload_byte_sizes"]
        self.band_indices = self.meta["band_indices"]

        # Load labels (small, always in memory)
        self.labels = np.load(os.path.join(root, "labels.npy"))

        # Lazily open band memmaps on first use so unused high-frequency
        # shards never need to be opened for a lower-K run.
        self._band_paths = [
            os.path.join(root, f"band_{b:02d}.npy")
            for b in range(self.num_bands)
        ]
        self._band_mmaps = [None] * self.num_bands

        # Precompute flat-index scatter targets for reconstruction
        self._scatter_indices = []
        for b in range(self.num_bands):
            self._scatter_indices.append(
                np.array(self.band_indices[b], dtype=np.int64)
            )

        # Current fidelity
        self.current_K = self.num_bands

        # I/O tracking
        self.band_bytes_read_epoch = 0
        self.band_bytes_read_total = 0
        self.bytes_read_epoch = 0
        self.bytes_read_total = 0

    def set_fidelity(self, K: int):
        """Set band cutoff for the current epoch."""
        self.current_K = min(K, self.num_bands)

    def reset_epoch_io(self):
        """Reset per-epoch byte counter."""
        self.band_bytes_read_epoch = 0
        self.bytes_read_epoch = 0

    def _get_band_mmap(self, band_idx: int):
        """Open a band shard lazily and return its read-only memmap."""
        mm = self._band_mmaps[band_idx]
        if mm is None:
            mm = np.load(self._band_paths[band_idx], mmap_mode="r")
            self._band_mmaps[band_idx] = mm
        return mm

    def _read_samples(self, indices: np.ndarray, K: int = None) -> torch.Tensor:
        """
        Read DCT coefficients for given sample indices at current fidelity K.

        Only accesses band files 0..K-1. Returns zero for bands K..L-1.
        Tracks exact bytes read.

        Args:
            indices: (B,) sample indices to read.
        Returns:
            (B, C, H, W) DCT coefficient tensor with high-freq bands zeroed.
        """
        B = len(indices)
        # Allocate full coefficient grid (zeros for unread bands)
        coeffs_flat = np.zeros((B, self.C, self.H * self.W), dtype=np.float32)

        target_K = self.current_K if K is None else min(K, self.num_bands)

        # Only read bands 0..K-1
        for b in range(target_K):
            band_data = self._get_band_mmap(b)[indices]
            scatter_idx = self._scatter_indices[b]
            coeffs_flat[:, :, scatter_idx] = band_data

            # Track bytes: B samples × C channels × band_size × 4 bytes
            bytes_read = B * self.C * self.band_sizes[b] * 4
            self.band_bytes_read_epoch += bytes_read
            self.band_bytes_read_total += bytes_read
            self.bytes_read_epoch += bytes_read
            self.bytes_read_total += bytes_read

        coeffs = coeffs_flat.reshape(B, self.C, self.H, self.W)
        return torch.from_numpy(coeffs)

    def get_dataset(self) -> "BandShardedDataset":
        """Return a PyTorch Dataset wrapping this store."""
        return BandShardedDataset(self)

    def get_loader(self, batch_size: int = 256, shuffle: bool = True,
                   num_workers: int = 0) -> DataLoader:
        """
        Return a DataLoader with index-based batching.

        Exact byte accounting currently requires num_workers=0 so all reads
        happen in the main process that owns the counters.
        """
        if num_workers != 0:
            raise ValueError(
                "BandShardedStore exact I/O accounting requires num_workers=0."
            )
        ds = self.get_dataset()
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=0, pin_memory=(self.device.type == "cuda"),
            collate_fn=self._collate,
        )

    def get_index_loader(self, batch_size: int = 256, shuffle: bool = True,
                         num_workers: int = 0) -> DataLoader:
        """
        Return an index/label loader for stores that serve batches directly.
        """
        if num_workers != 0:
            raise ValueError(
                "BandShardedStore exact I/O accounting requires num_workers=0."
            )
        ds = self.get_dataset()
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=0, pin_memory=(self.device.type == "cuda"),
        )

    def _collate(self, batch):
        """Custom collate: gather indices, do one bulk read."""
        indices = np.array([b[0] for b in batch])
        labels = np.array([b[1] for b in batch])
        coeffs = self._read_samples(indices)
        return coeffs, torch.from_numpy(labels)

    def get_bytes_per_sample_at_K(self, K: int = None) -> int:
        """Exact bytes read per sample at fidelity K."""
        K = K or self.current_K
        return sum(
            self.C * self.band_sizes[b] * 4
            for b in range(K)
        )

    def get_io_ratio(self, K: int = None) -> float:
        """Fraction of full-fidelity coefficient bytes requested."""
        K = K or self.current_K
        full = self.meta["bytes_per_sample_full"]
        return self.get_bytes_per_sample_at_K(K) / full

    def get_epoch_band_ratio(self) -> float:
        """Exact band-byte ratio realized over the current epoch."""
        full_epoch = self.N * self.meta["bytes_per_sample_full"]
        return self.band_bytes_read_epoch / full_epoch

    def get_epoch_total_ratio(self) -> float:
        """Exact total-byte ratio realized over the current epoch."""
        full_epoch = self.N * self.meta["bytes_per_sample_full"]
        return self.bytes_read_epoch / full_epoch

    def summary(self, K: int = None) -> str:
        """Human-readable I/O summary."""
        K = K or self.current_K
        bps = self.get_bytes_per_sample_at_K(K)
        full = self.meta["bytes_per_sample_full"]
        ratio = bps / full
        return (
            f"Band store: {self.root}\n"
            f"  Samples: {self.N}  Channels: {self.C}  "
            f"Size: {self.H}×{self.W}  Bands: {self.num_bands}\n"
            f"  Current K: {K}/{self.num_bands}\n"
            f"  Bytes/sample at K={K}: {bps} / {full} = {ratio:.3f}\n"
            f"  Epoch I/O so far: {self.bytes_read_epoch / 1e6:.1f} MB\n"
            f"  Total I/O: {self.bytes_read_total / 1e6:.1f} MB"
        )


class PhysicalTDCFStore(BandShardedStore):
    """
    Band store plus on-disk residual-patch shards for exact K+q serving.

    Residual patches are built only for the K values that appear in the
    learned schedule, which keeps the physical store manageable while
    matching the in-memory TDCF serving policy.
    """

    def __init__(self, root: str, device: torch.device = None):
        super().__init__(root, device=device)
        residual_meta_path = os.path.join(root, "residual_metadata.json")
        if not os.path.exists(residual_meta_path):
            raise FileNotFoundError(
                f"Missing residual patch metadata: {residual_meta_path}"
            )

        with open(residual_meta_path) as f:
            self.residual_meta = json.load(f)

        self.patch_size = self.residual_meta["patch_size"]
        self.nph = self.residual_meta["nph"]
        self.npw = self.residual_meta["npw"]
        self.P = self.residual_meta["num_patches"]
        self.available_patch_K = {int(k) for k in self.residual_meta["k_values"]}

        self._patch_paths = {}
        self._patch_mmaps = {}
        for k_str, k_meta in self.residual_meta["residuals"].items():
            K = int(k_str)
            self._patch_paths[K] = [
                os.path.join(root, rel_path) for rel_path in k_meta["patch_files"]
            ]
            self._patch_mmaps[K] = [None] * len(k_meta["patch_files"])

        self.current_q = self.P
        self.current_patch_indices = None
        self.residual_bytes_read_epoch = 0
        self.residual_bytes_read_total = 0

    def reset_epoch_io(self):
        super().reset_epoch_io()
        self.residual_bytes_read_epoch = 0

    def set_fidelity(self, K: int, q: int, patch_sensitivity: np.ndarray = None):
        self.current_K = min(K, self.num_bands)
        self.current_q = min(q, self.P)

        if self.current_q >= self.P:
            self.current_patch_indices = None
            return

        if self.current_K not in self.available_patch_K:
            raise ValueError(
                f"No residual patch store available for K={self.current_K}. "
                f"Available K values: {sorted(self.available_patch_K)}"
            )
        if patch_sensitivity is None:
            raise ValueError(
                "PhysicalTDCFStore requires patch_sensitivity when q < P."
            )

        top_idx = np.argsort(patch_sensitivity)[::-1][:self.current_q]
        self.current_patch_indices = np.array(top_idx, dtype=np.int64)

    def _get_patch_mmap(self, K: int, patch_idx: int):
        mm = self._patch_mmaps[K][patch_idx]
        if mm is None:
            mm = np.load(self._patch_paths[K][patch_idx], mmap_mode="r")
            self._patch_mmaps[K][patch_idx] = mm
        return mm

    def _read_residual_image(self, indices: np.ndarray) -> torch.Tensor:
        B = len(indices)
        ps = self.patch_size
        residual = np.zeros(
            (B, self.C, self.nph * ps, self.npw * ps),
            dtype=np.float32,
        )
        for patch_idx in self.current_patch_indices:
            patch_data = self._get_patch_mmap(self.current_K, patch_idx)[indices]
            row = patch_idx // self.npw
            col = patch_idx % self.npw
            residual[:, :, row * ps:(row + 1) * ps, col * ps:(col + 1) * ps] = patch_data

            bytes_read = B * self.C * ps * ps * 4
            self.residual_bytes_read_epoch += bytes_read
            self.residual_bytes_read_total += bytes_read
            self.bytes_read_epoch += bytes_read
            self.bytes_read_total += bytes_read

        return torch.from_numpy(residual)

    @torch.no_grad()
    def serve_indices(self, indices) -> torch.Tensor:
        """
        Serve a batch of images directly from on-disk storage.
        """
        if torch.is_tensor(indices):
            indices = indices.detach().cpu().numpy()
        else:
            indices = np.asarray(indices, dtype=np.int64)

        # If q restores every patch, it is cheaper and exact to read the full
        # coefficient representation rather than the duplicated residual store.
        if self.current_q >= self.P:
            coeffs = self._read_samples(indices, K=self.num_bands)
            return idct2d(coeffs.to(self.device, non_blocking=True))

        coeffs = self._read_samples(indices, K=self.current_K)
        x_base = idct2d(coeffs.to(self.device, non_blocking=True))
        residual = self._read_residual_image(indices).to(self.device, non_blocking=True)
        out = x_base.clone()
        out[:, :, :self.nph * self.patch_size, :self.npw * self.patch_size] += residual
        return out

    def get_patch_bytes_per_sample(self, q: int = None) -> int:
        q = self.current_q if q is None else q
        q = min(q, self.P)
        if q >= self.P:
            return 0
        return q * self.C * self.patch_size * self.patch_size * 4

    def get_total_bytes_per_sample(self, K: int = None, q: int = None) -> int:
        K = self.current_K if K is None else min(K, self.num_bands)
        q = self.current_q if q is None else min(q, self.P)
        if q >= self.P:
            return self.meta["bytes_per_sample_full"]
        return self.get_bytes_per_sample_at_K(K) + self.get_patch_bytes_per_sample(q)

    def get_total_io_ratio(self, K: int = None, q: int = None) -> float:
        full = self.meta["bytes_per_sample_full"]
        return self.get_total_bytes_per_sample(K=K, q=q) / full


class BandShardedDataset(Dataset):
    """
    Thin index-returning dataset. The actual I/O happens in the
    collate function via BandShardedStore._read_samples().
    """

    def __init__(self, store: BandShardedStore):
        self.store = store

    def __len__(self):
        return self.store.N

    def __getitem__(self, idx):
        # Return just the index and label — bulk read happens in collate
        return idx, self.store.labels[idx]


# ---------------------------------------------------------------------------
# Convenience: build store from torchvision dataset
# ---------------------------------------------------------------------------

def build_cifar100_store(data_dir: str = "./data", store_dir: str = None,
                         device: torch.device = None, batch_size: int = 1024,
                         num_workers: int = 0):
    """Build band-sharded stores for CIFAR-100 train and test splits."""
    from torchvision import datasets, transforms as T
    from .storage import build_band_store

    store_dir = store_dir or os.path.join(data_dir, "cifar100_band_store")
    plain_tfm = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_ds = datasets.CIFAR100(data_dir, True,  download=True, transform=plain_tfm)
    test_ds  = datasets.CIFAR100(data_dir, False, download=True, transform=plain_tfm)

    train_root = os.path.join(store_dir, "train")
    test_root  = os.path.join(store_dir, "test")

    print("=" * 50)
    print("Building CIFAR-100 band-sharded store (train)")
    print("=" * 50)
    meta_train = build_band_store(
        train_ds, train_root, H=32, W=32, num_bands=16, device=device,
        batch_size=batch_size, num_workers=num_workers)

    print("=" * 50)
    print("Building CIFAR-100 band-sharded store (test)")
    print("=" * 50)
    meta_test = build_band_store(
        test_ds, test_root, H=32, W=32, num_bands=16, device=device,
        batch_size=batch_size, num_workers=num_workers)

    return train_root, test_root, meta_train, meta_test


# ---------------------------------------------------------------------------
# Block-DCT store: per-patch band-selective I/O  (JPEG-style)
# ---------------------------------------------------------------------------

class BlockBandStore:
    """
    Block-DCT band store with **sample-adaptive** per-patch frequency selection.

    Each patch file stores (N, C, total_coeffs) in zig-zag order.
    For each sample in a batch, the loader computes a hybrid importance score:

        score[n, p] = patch_sensitivity[p] × patch_signal[n, p]

    and picks the top-q patches *per sample* to read at K_high bands.
    The remaining patches get only K_low bands (typically 1 = DC only).

    This ensures the object's patches always get high fidelity regardless
    of where the object lands after random crop/flip.
    """

    def __init__(self, root: str, device: torch.device = None):
        self.root = root
        self.device = device or torch.device("cpu")

        with open(os.path.join(root, "metadata.json")) as f:
            self.meta = json.load(f)

        self.N = self.meta["N"]
        self.C = self.meta["C"]
        self.P = self.meta["P"]
        self.nph = self.meta["nph"]
        self.npw = self.meta["npw"]
        self.bs_h = self.meta["block_size_h"]
        self.bs_w = self.meta["block_size_w"]
        self.total_coeffs = self.meta["total_coeffs_per_patch"]
        self.num_bands = self.meta["num_bands_per_patch"]
        self.band_size = self.meta["band_size"]
        self.bytes_per_sample_full = self.meta["bytes_per_sample_full"]

        # Zig-zag order for scatter-back
        zz = np.array(self.meta["zigzag_order"], dtype=np.int64)
        self.zz = zz

        self.labels = np.load(os.path.join(root, "labels.npy"), mmap_mode="r")

        # Per-sample per-patch signal energy (N, P) — tiny, memory-mapped
        sig_path = os.path.join(root, "patch_signal.npy")
        if os.path.exists(sig_path):
            self.patch_signal = np.load(sig_path, mmap_mode="r")
        else:
            # Fallback: uniform signal → degrades to global-q
            self.patch_signal = None

        # Lazy-open patch memmaps
        self._patch_mmaps = [None] * self.P

        # Fidelity state
        self.K_high = self.num_bands
        self.K_low = 1
        self.q = self.P
        self.patch_sensitivity = np.ones(self.P, dtype=np.float32) / self.P
        self.band_sensitivity = np.ones(self.num_bands, dtype=np.float32) / self.num_bands
        self.patch_policy = 'gradient'

        # Pre-computed coeffs counts for scatter
        self.n_coeffs_high = self.K_high * self.band_size
        self.n_coeffs_low = self.K_low * self.band_size

        # Budget in band-slots (for greedy allocation)
        self.budget_bands = self.P * self.num_bands

        # I/O tracking
        self.bytes_read_epoch = 0
        self.bytes_read_total = 0
        self.samples_read_epoch = 0

    def _get_patch_mmap(self, p_idx: int):
        if self._patch_mmaps[p_idx] is None:
            path = os.path.join(self.root, f"patch_{p_idx:02d}.npy")
            self._patch_mmaps[p_idx] = np.load(path, mmap_mode="r")
        return self._patch_mmaps[p_idx]

    def reset_epoch_io(self):
        self.bytes_read_epoch = 0
        self.samples_read_epoch = 0

    def set_fidelity(self, K_high: int, K_low: int, q: int,
                     patch_sensitivity: np.ndarray = None,
                     band_sensitivity: np.ndarray = None,
                     patch_policy: str = 'gradient'):
        """
        Set per-patch fidelity parameters.

        Args:
            K_high: bands for important patches.
            K_low:  bands for background patches.
            q:      number of important patches *per sample*.
            patch_sensitivity: (P,) global gradient importance.
            band_sensitivity:  (num_bands,) per-band gradient importance.
            patch_policy: 'gradient', 'random', 'static', or 'greedy'.
        """
        self.K_high = min(K_high, self.num_bands)
        self.K_low = min(K_low, self.num_bands)
        self.q = min(q, self.P)
        self.n_coeffs_high = self.K_high * self.band_size
        self.n_coeffs_low = self.K_low * self.band_size
        self.patch_policy = patch_policy

        # Total band-slot budget (same bytes as binary K_high/K_low).
        # Use clipped values so the greedy allocator cannot over-budget.
        self.budget_bands = (
            self.q * self.K_high + (self.P - self.q) * self.K_low
        )

        if patch_sensitivity is not None:
            self.patch_sensitivity = np.asarray(
                patch_sensitivity, dtype=np.float32).copy()
        else:
            self.patch_sensitivity = np.ones(self.P, dtype=np.float32) / self.P

        if band_sensitivity is not None:
            self.band_sensitivity = np.asarray(
                band_sensitivity, dtype=np.float32).copy()
        else:
            self.band_sensitivity = np.ones(self.num_bands, dtype=np.float32) / self.num_bands

    def set_budget(self, total_bands: int,
                   patch_sensitivity: np.ndarray = None,
                   band_sensitivity: np.ndarray = None,
                   k_low: int = 1,
                   patch_policy: str = 'greedy'):
        """
        Budget-constrained fidelity allocation.

        Instead of coverage thresholds (K_high, q) that drift toward
        full I/O on datasets with flat sensitivity, this method directly
        controls the total number of band-slots read across all patches.

        The greedy allocator distributes this budget to maximize:
            Σ_p Σ_{b ∈ allocated(p)} g̃(p) × s(b)

        Args:
            total_bands: Total band-slot budget (max = P × num_bands).
            patch_sensitivity: (P,) per-patch importance weights.
            band_sensitivity:  (num_bands,) per-band importance weights.
            k_low:  Minimum bands per patch (floor guarantee).
        """
        self.budget_bands = max(self.P * k_low,
                                min(total_bands, self.P * self.num_bands))
        self.K_low = k_low
        self.patch_policy = patch_policy

        if patch_policy == 'greedy':
            self.K_high = self.num_bands  # greedy will decide actual K per patch
            self.q = self.P               # greedy handles all patches
        else:
            self.K_high = self.num_bands
            extra_per_patch = max(self.K_high - self.K_low, 1)
            extra_total = max(self.budget_bands - self.P * self.K_low, 0)
            self.q = int(np.clip(
                round(extra_total / extra_per_patch), 0, self.P
            ))

        self.n_coeffs_high = self.K_high * self.band_size
        self.n_coeffs_low = self.K_low * self.band_size

        if patch_sensitivity is not None:
            self.patch_sensitivity = np.asarray(
                patch_sensitivity, dtype=np.float32).copy()
        if band_sensitivity is not None:
            self.band_sensitivity = np.asarray(
                band_sensitivity, dtype=np.float32).copy()

    # ── Patch selection policies ────────────────────────────────────────

    def _compute_importance_mask(self, indices: np.ndarray) -> np.ndarray:
        """
        Compute per-sample importance mask (for gradient/random/static policies).

        Returns:
            is_important: (B, P) bool — True for top-q patches in each sample.
        """
        B = len(indices)

        if self.q <= 0:
            return np.zeros((B, self.P), dtype=bool)
        if self.q >= self.P:
            return np.ones((B, self.P), dtype=bool)

        if self.patch_policy == 'random':
            scores = np.random.rand(B, self.P)
        elif self.patch_policy == 'static':
            scores = np.broadcast_to(
                self.patch_sensitivity[np.newaxis, :], (B, self.P)).copy()
        else:  # 'gradient'
            if self.patch_signal is not None:
                sig = np.array(self.patch_signal[indices])
                scores = self.patch_sensitivity[np.newaxis, :] * sig
            else:
                scores = np.broadcast_to(
                    self.patch_sensitivity[np.newaxis, :], (B, self.P)).copy()

        top_q_idx = np.argpartition(-scores, self.q - 1, axis=1)[:, :self.q]
        is_important = np.zeros((B, self.P), dtype=bool)
        np.put_along_axis(is_important, top_q_idx, True, axis=1)
        return is_important

    def _compute_k_allocation(self, indices: np.ndarray) -> np.ndarray:
        """
        Compute per-sample, per-patch band allocation.

        Returns:
            k_alloc: (B, P) int — number of frequency bands for each patch.
        """
        if self.patch_policy == 'greedy':
            return self._greedy_k_allocation(indices)

        # Binary allocation for gradient/random/static policies
        is_important = self._compute_importance_mask(indices)
        k_alloc = np.where(is_important, self.K_high, self.K_low)
        return k_alloc.astype(np.int32)

    def _greedy_k_allocation(self, indices: np.ndarray) -> np.ndarray:
        """
        Greedy per-patch variable-K allocation under a fixed byte budget.

        Uses marginal value = band_sensitivity[k]^2 * patch_signal[n,p]^2
        to greedily assign bands to patches that benefit most.
        Same total bytes as binary K_high/K_low, but smarter distribution.
        """
        B = len(indices)
        budget = self.budget_bands

        # Start every patch at K_low
        k_alloc = np.full((B, self.P), self.K_low, dtype=np.int32)
        extra = budget - self.P * self.K_low
        if extra <= 0:
            return k_alloc

        # Per-sample patch signal energy
        if self.patch_signal is not None:
            sig_sq = np.array(self.patch_signal[indices]) ** 2  # (B, P)
        else:
            sig_sq = np.ones((B, self.P), dtype=np.float32)

        # Band sensitivity squared, starting from K_low
        max_extra = self.num_bands - self.K_low
        bs_sq = self.band_sensitivity[self.K_low:] ** 2       # (max_extra,)

        # All marginal values: (B, P, max_extra)
        # marginals[n, p, j] = sig_sq[n,p] * bs_sq[j]
        marginals = sig_sq[:, :, np.newaxis] * bs_sq[np.newaxis, np.newaxis, :]
        flat = marginals.reshape(B, -1)  # (B, P * max_extra)

        # Select top-`extra` marginal values per sample.
        extra = min(extra, self.P * max_extra)
        if extra >= flat.shape[1]:
            top_idx = np.broadcast_to(
                np.arange(flat.shape[1], dtype=np.int64), (B, flat.shape[1])
            )
        else:
            top_idx = np.argpartition(-flat, extra - 1, axis=1)[:, :extra]

        # Convert flat indices → (patch, band_offset)
        patch_ids = top_idx // max_extra

        # Count selected bands per (sample, patch)
        batch_idx = np.repeat(np.arange(B), extra)
        np.add.at(k_alloc, (batch_idx, patch_ids.ravel()), 1)

        # Clamp to num_bands
        np.clip(k_alloc, self.K_low, self.num_bands, out=k_alloc)
        return k_alloc

    # ── Batch I/O ─────────────────────────────────────────────────────────

    def _read_batch(self, indices: np.ndarray) -> torch.Tensor:
        """Read a batch with per-sample, per-patch variable-K band selection."""
        B = len(indices)
        self.samples_read_epoch += B

        coeffs_out = np.zeros(
            (B, self.P, self.C, self.total_coeffs), dtype=np.float32)

        k_alloc = self._compute_k_allocation(indices)  # (B, P) int

        for p_idx in range(self.P):
            mm = self._get_patch_mmap(p_idx)
            k_for_patch = k_alloc[:, p_idx]  # (B,)

            for k_val in np.unique(k_for_patch):
                rows = np.where(k_for_patch == k_val)[0]
                if len(rows) == 0:
                    continue

                n_coeffs = k_val * self.band_size
                if n_coeffs == 0:
                    continue

                zz_k = self.zz[:n_coeffs]
                data = mm[indices[rows], :, :n_coeffs]

                for ci in range(n_coeffs):
                    coeffs_out[rows, p_idx, :, zz_k[ci]] = data[:, :, ci]

                self.bytes_read_epoch += int(len(rows) * self.C * n_coeffs * 4)
                self.bytes_read_total += int(len(rows) * self.C * n_coeffs * 4)

        coeffs_out = coeffs_out.reshape(B, self.P, self.C, self.bs_h, self.bs_w)
        return torch.from_numpy(coeffs_out)

    @torch.no_grad()
    def serve_indices(self, indices) -> torch.Tensor:
        """Read coefficients and reconstruct spatial images."""
        if torch.is_tensor(indices):
            indices = indices.detach().cpu().numpy()
        else:
            indices = np.asarray(indices, dtype=np.int64)

        from .transforms import block_idct2d
        coeffs = self._read_batch(indices)
        coeffs_gpu = coeffs.to(self.device, non_blocking=True)
        return block_idct2d(coeffs_gpu, self.nph, self.npw)

    def get_io_ratio(self) -> float:
        """Theoretical I/O ratio for current fidelity settings.

        With sample-adaptive q, each sample reads exactly q patches at
        K_high and (P-q) patches at K_low, regardless of *which* patches.
        """
        coeffs_full = self.P * self.total_coeffs
        if self.patch_policy == 'greedy':
            coeffs_read = self.budget_bands * self.band_size
        else:
            coeffs_read = (self.q * self.K_high +
                           (self.P - self.q) * self.K_low) * self.band_size
        return coeffs_read / coeffs_full

    def get_epoch_io_ratio(self) -> float:
        """Actual bytes read this epoch vs full-fidelity bytes."""
        if self.bytes_read_epoch == 0 or self.samples_read_epoch == 0:
            return 0.0
        full_epoch_bytes = self.samples_read_epoch * self.bytes_per_sample_full
        return self.bytes_read_epoch / full_epoch_bytes

    def get_loader(self, batch_size: int = 256, shuffle: bool = True,
                   num_workers: int = 0):
        """Return a DataLoader yielding (indices, labels) tuples."""
        if num_workers != 0:
            raise ValueError(
                "BlockBandStore requires num_workers=0 for exact byte tracking.")
        ds = BlockBandDataset(self)
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=0, pin_memory=False,
            collate_fn=self._collate)

    def _collate(self, batch):
        indices = np.array([b[0] for b in batch], dtype=np.int64)
        labels = np.array([b[1] for b in batch], dtype=np.int64)
        return torch.from_numpy(indices), torch.from_numpy(labels)


class BlockBandDataset(Dataset):
    """Thin index+label dataset for BlockBandStore."""

    def __init__(self, store: BlockBandStore):
        self.store = store

    def __len__(self):
        return self.store.N

    def __getitem__(self, idx):
        return idx, self.store.labels[idx]


class CropAwareBlockBandStore:
    """
    Crop-aware block-DCT store for large images.

    Images are pre-resized to a deterministic canvas with shorter side fixed
    (for example 256), padded to the maximum patch grid in the dataset, and
    stored patch-by-patch in zig-zag coefficient order. At serving time the
    store can sample a crop window, read only the overlapping patches, then
    reconstruct the cropped region and resize it to the model's input size.

    This is the physical-I/O substrate for a future ImageNet-1K store-backed
    trainer. Phase 1 supports full-fidelity crop-aware serving and exact patch
    byte accounting. Budgeted partial-band serving can be layered on top.
    """

    def __init__(
        self,
        root: str,
        device: torch.device = None,
        output_size: int = 224,
        crop_scale=(0.08, 1.0),
        crop_ratio=(3.0 / 4.0, 4.0 / 3.0),
    ):
        self.root = root
        self.device = device or torch.device("cpu")
        self.output_size = output_size
        self.crop_scale = crop_scale
        self.crop_ratio = crop_ratio

        with open(os.path.join(root, "metadata.json")) as f:
            self.meta = json.load(f)

        if self.meta.get("format") != "crop_aware_block_dct":
            raise ValueError(
                f"{root} is not a crop-aware block store "
                f"(format={self.meta.get('format')!r})."
            )

        self.N = self.meta["N"]
        self.C = self.meta["C"]
        self.P = self.meta["P"]
        self.nph = self.meta["nph"]
        self.npw = self.meta["npw"]
        self.bs_h = self.meta["block_size_h"]
        self.bs_w = self.meta["block_size_w"]
        self.total_coeffs = self.meta["total_coeffs_per_patch"]
        self.num_bands = self.meta["num_bands_per_patch"]
        self.band_size = self.meta["band_size"]
        self.canvas_h = self.meta["canvas_h"]
        self.canvas_w = self.meta["canvas_w"]
        self.resize_shorter = self.meta["resize_shorter"]
        self.bytes_per_sample_full = self.meta["bytes_per_sample_full"]

        self.zz = np.array(self.meta["zigzag_order"], dtype=np.int64)
        self.labels = np.load(os.path.join(root, "labels.npy"), mmap_mode="r")
        self.sample_shapes = np.load(os.path.join(root, "sample_shapes.npy"), mmap_mode="r")
        self.patch_signal = np.load(os.path.join(root, "patch_signal.npy"), mmap_mode="r")

        self._patch_mmaps = [None] * self.P

        self.bytes_read_epoch = 0
        self.bytes_read_total = 0
        self.samples_read_epoch = 0

    def _get_patch_mmap(self, p_idx: int):
        if self._patch_mmaps[p_idx] is None:
            path = os.path.join(self.root, f"patch_{p_idx:03d}.npy")
            if not os.path.exists(path):
                path = os.path.join(self.root, f"patch_{p_idx:02d}.npy")
            self._patch_mmaps[p_idx] = np.load(path, mmap_mode="r")
        return self._patch_mmaps[p_idx]

    def reset_epoch_io(self):
        self.bytes_read_epoch = 0
        self.samples_read_epoch = 0

    def get_dataset(self):
        return BlockBandDataset(self)

    def get_index_loader(self, batch_size: int = 256, shuffle: bool = True,
                         num_workers: int = 0):
        if num_workers != 0:
            raise ValueError(
                "CropAwareBlockBandStore exact I/O accounting requires num_workers=0."
            )
        ds = self.get_dataset()
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=0, pin_memory=False,
        )

    def _sample_crop_params(self, indices: np.ndarray):
        params = np.zeros((len(indices), 4), dtype=np.int64)
        for row, idx in enumerate(indices):
            h, w = self.sample_shapes[idx]
            dummy = torch.empty(self.C, int(h), int(w))
            top, left, height, width = T.RandomResizedCrop.get_params(
                dummy, scale=self.crop_scale, ratio=self.crop_ratio
            )
            params[row] = (top, left, height, width)
        return params

    def _patch_bbox_from_crop(self, top: int, left: int, height: int, width: int):
        row0 = max(0, top // self.bs_h)
        col0 = max(0, left // self.bs_w)
        row1 = min(self.nph, math.ceil((top + height) / self.bs_h))
        col1 = min(self.npw, math.ceil((left + width) / self.bs_w))
        return row0, row1, col0, col1

    def _read_visible_patches(self, indices: np.ndarray, crop_params: np.ndarray):
        B = len(indices)
        coeffs_out = np.zeros(
            (B, self.P, self.C, self.total_coeffs), dtype=np.float32
        )
        visible_mask = np.zeros((B, self.P), dtype=bool)

        for b, (top, left, height, width) in enumerate(crop_params):
            row0, row1, col0, col1 = self._patch_bbox_from_crop(
                int(top), int(left), int(height), int(width)
            )
            for row in range(row0, row1):
                for col in range(col0, col1):
                    p_idx = row * self.npw + col
                    mm = self._get_patch_mmap(p_idx)
                    coeffs_out[b, p_idx] = mm[indices[b]]
                    visible_mask[b, p_idx] = True
                    bytes_read = self.C * self.total_coeffs * 4
                    self.bytes_read_epoch += bytes_read
                    self.bytes_read_total += bytes_read

        self.samples_read_epoch += B
        coeffs_out = coeffs_out.reshape(B, self.P, self.C, self.bs_h, self.bs_w)
        return torch.from_numpy(coeffs_out), visible_mask

    @torch.no_grad()
    def serve_indices(self, indices, crop_params: Optional[np.ndarray] = None):
        """
        Serve a batch of crop-aware images at full fidelity.

        Returns:
            x: (B, C, output_size, output_size) tensor on self.device
            crop_params: (B, 4) integer crop parameters
            visible_mask: (B, P) bool mask of patches touched by the crop
        """
        if torch.is_tensor(indices):
            indices = indices.detach().cpu().numpy()
        else:
            indices = np.asarray(indices, dtype=np.int64)

        if crop_params is None:
            crop_params = self._sample_crop_params(indices)
        else:
            crop_params = np.asarray(crop_params, dtype=np.int64)

        from .transforms import block_idct2d

        coeffs, visible_mask = self._read_visible_patches(indices, crop_params)
        coeffs = coeffs.to(self.device, non_blocking=True)
        canvas = block_idct2d(coeffs, self.nph, self.npw)

        crops = []
        for b, (top, left, height, width) in enumerate(crop_params):
            sample_h, sample_w = self.sample_shapes[indices[b]]
            crop = canvas[
                b:b + 1,
                :,
                int(top):min(int(top + height), int(sample_h)),
                int(left):min(int(left + width), int(sample_w)),
            ]
            crop = TF.resize(
                crop,
                [self.output_size, self.output_size],
                interpolation=TF.InterpolationMode.BILINEAR,
                antialias=True,
            )
            crops.append(crop)

        x = torch.cat(crops, dim=0)
        return x, crop_params, visible_mask

    def get_io_ratio(self) -> float:
        if self.bytes_read_epoch == 0 or self.samples_read_epoch == 0:
            return 0.0
        full_epoch_bytes = self.samples_read_epoch * self.bytes_per_sample_full
        return self.bytes_read_epoch / full_epoch_bytes
