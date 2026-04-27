"""
Compressed quantized DCT store for fixed-size ImageNet views.

This is a physical-I/O experiment path:
  - preprocess images to a fixed SxS view, e.g. 288x288
  - store block-DCT coefficients as quantized int8 chunks
  - compress each chunk/tile/band payload with zlib
  - serve block-aligned random 224x224 crops by reading only needed bands

The goal is not to replace the online ImageNet training pipeline. The goal is
to prove that the fidelity budget can become real physical coefficient I/O when
the dataset is stored in a coefficient-addressable format.
"""

from __future__ import annotations

import json
import math
import os
import random
import zlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from tdcf.transforms import block_idct2d, zigzag_order


def build_tile_specs(nph: int, npw: int, tile_blocks: int) -> List[dict]:
    specs = []
    tile_idx = 0
    for tr0 in range(0, nph, tile_blocks):
        for tc0 in range(0, npw, tile_blocks):
            th = min(tile_blocks, nph - tr0)
            tw = min(tile_blocks, npw - tc0)
            block_ids = [
                (tr0 + rr) * npw + (tc0 + cc)
                for rr in range(th)
                for cc in range(tw)
            ]
            specs.append(
                {
                    "tile_index": tile_idx,
                    "tr0": tr0,
                    "tc0": tc0,
                    "th": th,
                    "tw": tw,
                    "block_ids": block_ids,
                    "blocks_in_tile": len(block_ids),
                }
            )
            tile_idx += 1
    return specs


class QuantizedDCTDataset(Dataset):
    def __init__(self, store: "QuantizedFixedDCTStore"):
        self.store = store

    def __len__(self):
        return self.store.N

    def __getitem__(self, idx):
        return int(idx), int(self.store.labels[idx])


class ChunkBatchSampler(Sampler[List[int]]):
    """Group batches within compressed chunks to reduce read amplification."""

    def __init__(
        self,
        dataset_len: int,
        chunk_size: int,
        batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 42,
    ):
        self.dataset_len = int(dataset_len)
        self.chunk_size = int(chunk_size)
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _all_batches(self) -> List[List[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        batches: List[List[int]] = []
        chunk_ids = list(range(math.ceil(self.dataset_len / self.chunk_size)))
        if self.shuffle:
            rng.shuffle(chunk_ids)

        for chunk_id in chunk_ids:
            start = chunk_id * self.chunk_size
            end = min(start + self.chunk_size, self.dataset_len)
            local = np.arange(start, end, dtype=np.int64)
            if self.shuffle:
                rng.shuffle(local)
            if self.drop_last:
                usable = (len(local) // self.batch_size) * self.batch_size
                local = local[:usable]
            for off in range(0, len(local), self.batch_size):
                batch = local[off:off + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                if len(batch) > 0:
                    batches.append(batch.tolist())
        return batches

    def __iter__(self):
        batches = self._all_batches()
        for batch in batches[self.rank::self.num_replicas]:
            yield batch

    def __len__(self):
        total = len(self._all_batches())
        return math.ceil(max(total - self.rank, 0) / self.num_replicas)


class QuantizedFixedDCTStore:
    def __init__(
        self,
        root: str,
        device: Optional[torch.device] = None,
        output_size: int = 224,
    ):
        self.root = root
        self.device = device or torch.device("cpu")
        self.output_size = int(output_size)

        with open(os.path.join(root, "metadata.json")) as f:
            self.meta = json.load(f)
        if self.meta.get("format") != "quantized_fixed_dct_v1":
            raise ValueError(
                f"{root} is not a quantized fixed DCT store "
                f"(format={self.meta.get('format')!r})."
            )

        self.N = int(self.meta["N"])
        self.C = int(self.meta["C"])
        self.view_size = int(self.meta["view_size"])
        self.block_size = int(self.meta["block_size"])
        self.nph = int(self.meta["nph"])
        self.npw = int(self.meta["npw"])
        self.P = int(self.meta["P"])
        self.total_coeffs = int(self.meta["total_coeffs_per_patch"])
        self.num_bands = int(self.meta["num_bands_per_patch"])
        self.band_size = int(self.meta["band_size"])
        self.chunk_size = int(self.meta["chunk_size"])
        self.num_chunks = int(self.meta["num_chunks"])
        self.tile_blocks = int(self.meta["tile_blocks"])
        self.num_tiles = int(self.meta["num_tiles"])
        self.dtype = np.dtype(self.meta["dtype"])

        self.labels = np.load(os.path.join(root, "labels.npy"), mmap_mode="r")
        self.scales = np.load(os.path.join(root, "scales.npy")).astype(np.float32)
        self.chunk_sizes = np.load(os.path.join(root, "chunk_sizes.npy"), mmap_mode="r")
        self.offsets = np.load(os.path.join(root, "offsets.npy"), mmap_mode="r")
        self.lengths = np.load(os.path.join(root, "lengths.npy"), mmap_mode="r")
        self.raw_lengths = np.load(os.path.join(root, "raw_lengths.npy"), mmap_mode="r")
        self.tile_specs = build_tile_specs(self.nph, self.npw, self.tile_blocks)

        self.zz = np.asarray(self.meta["zigzag_order"], dtype=np.int64)
        self.zz_torch = torch.tensor(self.zz, dtype=torch.long, device=self.device)
        self.band_files = [
            open(os.path.join(root, f"band_{band:02d}.bin"), "rb")
            for band in range(self.num_bands)
        ]

        self.mode = "full"
        self.budget_ratio = 1.0
        self.k_low = 1
        self.patch_policy = "greedy"
        self.patch_sensitivity = np.ones(self.P, dtype=np.float32) / self.P
        self.band_sensitivity = np.ones(self.num_bands, dtype=np.float32) / self.num_bands

        self.bytes_read_epoch = 0
        self.bytes_read_total = 0
        self.full_bytes_epoch = 0
        self.full_bytes_total = 0
        self.samples_read_epoch = 0
        self.read_ops_epoch = 0
        self.read_bytes_events: List[int] = []

    def close(self):
        for handle in getattr(self, "band_files", []):
            try:
                handle.close()
            except Exception:
                pass

    def reset_epoch_io(self):
        self.bytes_read_epoch = 0
        self.full_bytes_epoch = 0
        self.samples_read_epoch = 0
        self.read_ops_epoch = 0
        self.read_bytes_events = []

    def get_dataset(self):
        return QuantizedDCTDataset(self)

    def set_full_fidelity(self):
        self.mode = "full"
        self.budget_ratio = 1.0
        self.k_low = 1

    def set_budget_ratio(
        self,
        budget_ratio: float,
        patch_sensitivity: Optional[np.ndarray] = None,
        band_sensitivity: Optional[np.ndarray] = None,
        k_low: int = 1,
        patch_policy: str = "greedy",
    ):
        self.mode = "budget"
        self.budget_ratio = float(max(0.0, min(1.0, budget_ratio)))
        self.k_low = int(min(max(k_low, 0), self.num_bands))
        self.patch_policy = patch_policy
        if patch_sensitivity is not None:
            self.patch_sensitivity = np.asarray(patch_sensitivity, dtype=np.float32).copy()
        if band_sensitivity is not None:
            self.band_sensitivity = np.asarray(band_sensitivity, dtype=np.float32).copy()

    def _sample_crop_params(self, batch_size: int, deterministic_val: bool = False):
        max_off = self.view_size - self.output_size
        if max_off < 0:
            raise ValueError("output_size cannot exceed stored view_size")
        if deterministic_val:
            top = (max_off // 2) // self.block_size * self.block_size
            left = (max_off // 2) // self.block_size * self.block_size
            return np.tile(
                np.asarray([top, left, self.output_size, self.output_size], dtype=np.int64),
                (batch_size, 1),
            )

        max_step = max_off // self.block_size
        params = np.zeros((batch_size, 4), dtype=np.int64)
        for i in range(batch_size):
            top = random.randint(0, max_step) * self.block_size if max_step > 0 else 0
            left = random.randint(0, max_step) * self.block_size if max_step > 0 else 0
            params[i] = (top, left, self.output_size, self.output_size)
        return params

    def _build_visible_mask(self, crop_params: np.ndarray):
        visible = np.zeros((len(crop_params), self.P), dtype=bool)
        for b, (top, left, height, width) in enumerate(crop_params):
            row0 = max(0, int(top) // self.block_size)
            col0 = max(0, int(left) // self.block_size)
            row1 = min(self.nph, math.ceil((int(top) + int(height)) / self.block_size))
            col1 = min(self.npw, math.ceil((int(left) + int(width)) / self.block_size))
            for row in range(row0, row1):
                visible[b, row * self.npw + col0:row * self.npw + col1] = True
        return visible

    def _compute_k_allocation(self, visible_mask: np.ndarray):
        if self.mode == "full":
            out = np.zeros((len(visible_mask), self.P), dtype=np.int32)
            out[visible_mask] = self.num_bands
            return out

        out = np.zeros((len(visible_mask), self.P), dtype=np.int32)
        max_extra = self.num_bands - self.k_low
        band_scores = self.band_sensitivity[self.k_low:] ** 2

        for b in range(len(visible_mask)):
            visible_ids = np.flatnonzero(visible_mask[b])
            V = len(visible_ids)
            if V == 0:
                continue
            target_budget = int(round(self.budget_ratio * V * self.num_bands))
            target_budget = max(V * self.k_low, min(V * self.num_bands, target_budget))
            out[b, visible_ids] = self.k_low
            extra_total = target_budget - V * self.k_low
            if extra_total <= 0 or max_extra <= 0:
                continue
            if self.patch_policy == "random":
                patch_scores = np.random.rand(V).astype(np.float32)
            else:
                patch_scores = self.patch_sensitivity[visible_ids].astype(np.float32)
            marginals = patch_scores[:, None] * band_scores[None, :]
            flat = marginals.reshape(-1)
            extra_total = min(extra_total, flat.size)
            top_idx = np.argpartition(-flat, extra_total - 1)[:extra_total]
            patch_local = top_idx // max_extra
            np.add.at(out[b], visible_ids[patch_local], 1)
        np.clip(out, 0, self.num_bands, out=out)
        return out

    def _chunk_groups(self, indices: np.ndarray):
        chunk_ids = indices // self.chunk_size
        for chunk_id in np.unique(chunk_ids):
            rows = np.flatnonzero(chunk_ids == chunk_id)
            local_rows = indices[rows] - chunk_id * self.chunk_size
            yield int(chunk_id), rows, local_rows.astype(np.int64)

    def _read_record(self, chunk_id: int, tile_index: int, band_index: int):
        offset = int(self.offsets[chunk_id, tile_index, band_index])
        length = int(self.lengths[chunk_id, tile_index, band_index])
        raw_length = int(self.raw_lengths[chunk_id, tile_index, band_index])
        handle = self.band_files[band_index]
        handle.seek(offset)
        payload = handle.read(length)
        raw = zlib.decompress(payload)
        if len(raw) != raw_length:
            raise IOError(
                f"Corrupt record chunk={chunk_id} tile={tile_index} band={band_index}: "
                f"expected {raw_length} bytes, got {len(raw)}"
            )
        self.bytes_read_epoch += length
        self.bytes_read_total += length
        self.read_ops_epoch += 1
        self.read_bytes_events.append(length)
        return raw

    def _full_record_length(self, chunk_id: int, tile_index: int, band_index: int) -> int:
        return int(self.lengths[chunk_id, tile_index, band_index])

    @torch.no_grad()
    def read_coeffs(
        self,
        indices,
        crop_params: Optional[np.ndarray] = None,
        deterministic_val: bool = False,
    ):
        if torch.is_tensor(indices):
            indices = indices.detach().cpu().numpy()
        else:
            indices = np.asarray(indices, dtype=np.int64)

        if crop_params is None:
            crop_params = self._sample_crop_params(len(indices), deterministic_val=deterministic_val)
        else:
            crop_params = np.asarray(crop_params, dtype=np.int64)

        visible_mask = self._build_visible_mask(crop_params)
        k_alloc = self._compute_k_allocation(visible_mask)
        coeffs_zz = np.zeros((len(indices), self.P, self.C, self.total_coeffs), dtype=np.float32)

        for chunk_id, batch_rows, local_rows in self._chunk_groups(indices):
            chunk_n = int(self.chunk_sizes[chunk_id])
            for spec in self.tile_specs:
                tile_index = int(spec["tile_index"])
                block_ids = np.asarray(spec["block_ids"], dtype=np.int64)
                tile_visible = visible_mask[np.ix_(batch_rows, block_ids)]
                if not tile_visible.any():
                    continue

                for band_index in range(self.num_bands):
                    self.full_bytes_epoch += self._full_record_length(chunk_id, tile_index, band_index)
                    self.full_bytes_total += self._full_record_length(chunk_id, tile_index, band_index)
                    needs_band = (k_alloc[np.ix_(batch_rows, block_ids)] > band_index).any()
                    if not needs_band:
                        continue

                    raw = self._read_record(chunk_id, tile_index, band_index)
                    band_size = min((band_index + 1) * self.band_size, self.total_coeffs) - band_index * self.band_size
                    arr = np.frombuffer(raw, dtype=self.dtype).reshape(
                        chunk_n, len(block_ids), self.C, band_size
                    )
                    selected = arr[local_rows].astype(np.float32)
                    if self.scales.shape[0] == 1:
                        selected *= float(self.scales[0, band_index])
                    else:
                        selected *= self.scales[:, band_index].reshape(1, 1, self.C, 1)
                    start = band_index * self.band_size
                    end = start + band_size
                    coeffs_zz[np.ix_(batch_rows, block_ids, np.arange(self.C), np.arange(start, end))] = selected

        self.samples_read_epoch += len(indices)
        return torch.from_numpy(coeffs_zz), crop_params, visible_mask, k_alloc

    def reconstruct_crops(self, coeffs_zz: torch.Tensor, crop_params: np.ndarray):
        coeffs_zz = coeffs_zz.to(self.device, non_blocking=True)
        coeffs_flat = torch.zeros_like(coeffs_zz, device=self.device)
        coeffs_flat[:, :, :, self.zz_torch] = coeffs_zz
        coeffs = coeffs_flat.reshape(
            coeffs_flat.shape[0], self.P, self.C, self.block_size, self.block_size
        )
        canvas = block_idct2d(coeffs, self.nph, self.npw)
        crops = []
        for b, (top, left, height, width) in enumerate(crop_params):
            crop = canvas[
                b:b + 1,
                :,
                int(top):int(top + height),
                int(left):int(left + width),
            ]
            crops.append(crop)
        return torch.cat(crops, dim=0).clamp_(0.0, 1.0)

    @torch.no_grad()
    def serve_indices(self, indices, crop_params: Optional[np.ndarray] = None, deterministic_val: bool = False):
        coeffs_zz, crop_params, visible_mask, k_alloc = self.read_coeffs(
            indices, crop_params=crop_params, deterministic_val=deterministic_val
        )
        x = self.reconstruct_crops(coeffs_zz, crop_params)
        return x, crop_params, visible_mask, k_alloc

    def get_io_ratio(self) -> float:
        if self.full_bytes_epoch == 0:
            return 0.0
        return self.bytes_read_epoch / self.full_bytes_epoch

    def get_read_stats(self) -> dict:
        if not self.read_bytes_events:
            return {
                "read_ops_epoch": 0,
                "avg_bytes_per_read": 0.0,
                "p95_bytes_per_read": 0.0,
                "bytes_read_epoch": int(self.bytes_read_epoch),
                "full_bytes_epoch": int(self.full_bytes_epoch),
            }
        arr = np.asarray(self.read_bytes_events, dtype=np.float64)
        return {
            "read_ops_epoch": int(self.read_ops_epoch),
            "avg_bytes_per_read": float(arr.mean()),
            "p95_bytes_per_read": float(np.percentile(arr, 95)),
            "bytes_read_epoch": int(self.bytes_read_epoch),
            "full_bytes_epoch": int(self.full_bytes_epoch),
        }
