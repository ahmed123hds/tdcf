"""Fast sharded int8 DCT coefficient store.

This store follows the same high-throughput principle used by production
training loaders: batches read contiguous records from large shard files.
It avoids tiny random compressed records in the hot path.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from tdcf.transforms import block_idct2d, zigzag_order


@dataclass
class FastShardInfo:
    shard_id: int
    count: int
    band_paths: List[str]
    band_arrays: Dict[int, np.memmap]


class FastQuantizedDCTDataset(Dataset):
    def __init__(self, store: "FastQuantizedDCTStore"):
        self.store = store

    def __len__(self):
        return self.store.N

    def __getitem__(self, idx):
        return int(idx), int(self.store.labels[idx])


class FastShardBatchSampler(Sampler[List[int]]):
    """Yield contiguous same-shard batches, then shuffle batch order."""

    def __init__(
        self,
        num_samples: int,
        records_per_shard: int,
        batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 42,
    ):
        self.num_samples = int(num_samples)
        self.records_per_shard = int(records_per_shard)
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _all_batches(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        batches = []
        for start in range(0, self.num_samples, self.records_per_shard):
            end = min(start + self.records_per_shard, self.num_samples)
            usable = end - start
            if self.drop_last:
                usable = (usable // self.batch_size) * self.batch_size
            for off in range(0, usable, self.batch_size):
                batch = list(range(start + off, start + off + self.batch_size))
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        batches = self._all_batches()
        usable = (len(batches) // self.num_replicas) * self.num_replicas
        batches = batches[:usable]
        for batch in batches[self.rank::self.num_replicas]:
            yield batch

    def __len__(self):
        total = len(self._all_batches())
        usable = (total // self.num_replicas) * self.num_replicas
        return usable // self.num_replicas


class FastQuantizedDCTStore:
    def __init__(self, root: str, device=None):
        self.root = root
        self.device = device or torch.device("cpu")
        with open(os.path.join(root, "metadata.json")) as f:
            self.meta = json.load(f)
        if self.meta.get("format") != "fast_quantized_dct_v1":
            raise ValueError(f"{root} is not a fast quantized DCT store")

        self.N = int(self.meta["N"])
        self.C = int(self.meta["C"])
        self.view_size = int(self.meta["view_size"])
        self.block_size = int(self.meta["block_size"])
        self.nph = int(self.meta["nph"])
        self.npw = int(self.meta["npw"])
        self.P = self.nph * self.npw
        self.num_bands = int(self.meta["num_bands"])
        self.band_size = int(self.meta["band_size"])
        self.total_coeffs = int(self.meta["total_coeffs"])
        self.records_per_shard = int(self.meta["records_per_shard"])
        self.dtype = np.dtype(self.meta["dtype"])
        self.scales = np.load(os.path.join(root, "scales.npy")).astype(np.float32)
        self.labels = np.load(os.path.join(root, "labels.npy"), mmap_mode="r")
        self.zz = np.asarray(self.meta["zigzag_order"], dtype=np.int64)
        self._zz_torch_cache = {}

        self.shards: List[FastShardInfo] = []
        for shard_meta in self.meta["shards"]:
            shard_dir = os.path.join(root, shard_meta["dir"])
            band_paths = [
                os.path.join(shard_dir, f"band_{band:02d}.npy")
                for band in range(self.num_bands)
            ]
            self.shards.append(
                FastShardInfo(
                    shard_id=int(shard_meta["shard_id"]),
                    count=int(shard_meta["count"]),
                    band_paths=band_paths,
                    band_arrays={},
                )
            )

        self.budget_ratio = 1.0
        self.reset_epoch_io()

    def close(self):
        for shard in self.shards:
            shard.band_arrays.clear()

    def get_dataset(self):
        return FastQuantizedDCTDataset(self)

    def set_budget_ratio(self, ratio: float):
        self.budget_ratio = float(max(0.0, min(1.0, ratio)))

    def set_full_fidelity(self):
        self.budget_ratio = 1.0

    def reset_epoch_io(self):
        self.bytes_read_epoch = 0
        self.full_bytes_epoch = 0
        self.read_ops_epoch = 0

    def get_io_ratio(self):
        if self.full_bytes_epoch == 0:
            return 0.0
        return self.bytes_read_epoch / self.full_bytes_epoch

    def get_read_stats(self):
        return {
            "read_ops_epoch": int(self.read_ops_epoch),
            "bytes_read_epoch": int(self.bytes_read_epoch),
            "full_bytes_epoch": int(self.full_bytes_epoch),
            "avg_bytes_per_read": 0.0 if self.read_ops_epoch == 0 else self.bytes_read_epoch / self.read_ops_epoch,
            "p95_bytes_per_read": 0.0,
        }

    def _get_zz_torch(self, device):
        key = str(device)
        out = self._zz_torch_cache.get(key)
        if out is None:
            out = torch.as_tensor(self.zz, dtype=torch.long, device=device)
            self._zz_torch_cache[key] = out
        return out

    def _get_band_array(self, shard: FastShardInfo, band: int):
        arr = shard.band_arrays.get(band)
        if arr is None:
            arr = np.load(shard.band_paths[band], mmap_mode="r")
            shard.band_arrays[band] = arr
        return arr

    def _resolve_batch(self, indices: np.ndarray):
        shard_ids = indices // self.records_per_shard
        unique = np.unique(shard_ids)
        if len(unique) != 1:
            raise ValueError(f"Expected same-shard batch, got {unique.tolist()}")
        shard_id = int(unique[0])
        local = indices - shard_id * self.records_per_shard
        return self.shards[shard_id], local.astype(np.int64)

    @torch.no_grad()
    def serve_indices(self, indices):
        if torch.is_tensor(indices):
            indices = indices.detach().cpu().numpy()
        else:
            indices = np.asarray(indices, dtype=np.int64)

        shard, local = self._resolve_batch(indices)
        k_bands = int(math.ceil(self.budget_ratio * self.num_bands))
        k_bands = max(1, min(self.num_bands, k_bands))
        coeffs_zz = np.zeros(
            (len(indices), self.P, self.C, self.total_coeffs),
            dtype=np.float32,
        )

        full_bytes_per_band = len(indices) * self.P * self.C * self.band_size
        self.full_bytes_epoch += full_bytes_per_band * self.num_bands
        for band in range(k_bands):
            arr = self._get_band_array(shard, band)
            if np.all(np.diff(local) == 1):
                q = arr[int(local[0]):int(local[-1]) + 1]
            else:
                q = arr[local]
            q = q.astype(np.float32)
            if self.scales.shape[0] == 1:
                q *= float(self.scales[0, band])
            else:
                q *= self.scales[:, band].reshape(1, 1, self.C, 1)
            start = band * self.band_size
            end = min((band + 1) * self.band_size, self.total_coeffs)
            coeffs_zz[:, :, :, start:end] = q
            self.bytes_read_epoch += int(q.size)
            self.read_ops_epoch += 1

        coeffs_zz = torch.from_numpy(coeffs_zz).to(self.device, non_blocking=True)
        zz_torch = self._get_zz_torch(coeffs_zz.device)
        coeffs_flat = torch.zeros_like(coeffs_zz, device=self.device)
        coeffs_flat[:, :, :, zz_torch] = coeffs_zz
        coeffs = coeffs_flat.reshape(
            coeffs_flat.shape[0],
            self.P,
            self.C,
            self.block_size,
            self.block_size,
        )
        return block_idct2d(coeffs, self.nph, self.npw).clamp_(0.0, 1.0)
