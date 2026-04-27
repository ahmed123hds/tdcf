"""Original-size compressed quantized DCT store.

This is the general physical-I/O path:

  raw image -> pad to block grid -> block DCT -> zig-zag -> int8 quantization
  -> zlib-compressed tile/band chunks

At training time a random crop is sampled in the original image coordinate
system. The store reads only the intersecting block tiles and only the selected
frequency bands, dequantizes them, reconstructs the crop with inverse DCT, and
resizes it to the model input size.
"""

from __future__ import annotations

import json
import math
import os
import random
import zlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from tdcf.transforms import block_idct2d, zigzag_order


def bucket_key_from_shape(height: int, width: int, block_size: int) -> Tuple[int, int]:
    return math.ceil(int(height) / block_size), math.ceil(int(width) / block_size)


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
            specs.append({
                "tile_index": tile_idx,
                "tr0": tr0,
                "tc0": tc0,
                "th": th,
                "tw": tw,
                "block_ids": block_ids,
                "blocks_in_tile": len(block_ids),
            })
            tile_idx += 1
    return specs


@dataclass
class QuantBucketInfo:
    bucket_id: int
    bucket_dir: str
    nph: int
    npw: int
    canvas_h: int
    canvas_w: int
    P: int
    num_samples: int
    tile_specs: List[dict]
    sample_ids: np.ndarray
    sample_shapes: np.ndarray
    chunk_sizes: np.ndarray
    offsets: np.ndarray
    lengths: np.ndarray
    raw_lengths: np.ndarray
    patch_signal: np.ndarray
    band_files: List


class QuantizedDCTDataset(Dataset):
    def __init__(self, store: "OriginalQuantizedDCTStore"):
        self.store = store

    def __len__(self):
        return self.store.N

    def __getitem__(self, idx):
        return int(idx), int(self.store.labels[idx])


class BucketBatchSampler(Sampler[List[int]]):
    """Batch samples from the same shape bucket for static TPU tensors."""

    def __init__(
        self,
        bucket_ids: Sequence[int],
        batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 42,
    ):
        self.bucket_ids = np.asarray(bucket_ids, dtype=np.int32)
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        self._indices_by_bucket = {
            int(b): np.flatnonzero(self.bucket_ids == b)
            for b in np.unique(self.bucket_ids)
        }

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _all_batches(self) -> List[List[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        batches = []
        bucket_ids = list(self._indices_by_bucket.keys())
        if self.shuffle:
            rng.shuffle(bucket_ids)
        for bucket_id in bucket_ids:
            local = self._indices_by_bucket[bucket_id].copy()
            if self.shuffle:
                rng.shuffle(local)
            if self.drop_last:
                local = local[: (len(local) // self.batch_size) * self.batch_size]
            for start in range(0, len(local), self.batch_size):
                batch = local[start:start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                if len(batch):
                    batches.append(batch.tolist())
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        batches = self._all_batches()
        for batch in batches[self.rank::self.num_replicas]:
            yield batch

    def __len__(self):
        total = len(self._all_batches())
        return math.ceil(max(total - self.rank, 0) / self.num_replicas)


class OriginalQuantizedDCTStore:
    def __init__(
        self,
        root: str,
        device: Optional[torch.device] = None,
        output_size: int = 224,
        crop_scale=(0.08, 1.0),
        crop_ratio=(3.0 / 4.0, 4.0 / 3.0),
    ):
        self.root = root
        self.device = device or torch.device("cpu")
        self.output_size = int(output_size)
        self.crop_scale = crop_scale
        self.crop_ratio = crop_ratio

        with open(os.path.join(root, "metadata.json")) as f:
            self.meta = json.load(f)
        if self.meta.get("format") != "original_quantized_dct_v1":
            raise ValueError(
                f"{root} is not an original-size quantized DCT store "
                f"(format={self.meta.get('format')!r})."
            )

        self.N = int(self.meta["N"])
        self.C = int(self.meta["C"])
        self.block_size = int(self.meta["block_size"])
        self.num_bands = int(self.meta["num_bands_per_patch"])
        self.band_size = int(self.meta["band_size"])
        self.total_coeffs = int(self.meta["total_coeffs_per_patch"])
        self.tile_blocks = int(self.meta["tile_blocks"])
        self.chunk_size = int(self.meta["chunk_size"])
        self.dtype = np.dtype(self.meta["dtype"])
        self.zz = np.asarray(self.meta["zigzag_order"], dtype=np.int64)
        self.zz_torch = torch.tensor(self.zz, dtype=torch.long, device=self.device)
        self.scales = np.load(os.path.join(root, "scales.npy")).astype(np.float32)

        self.labels = np.load(os.path.join(root, "labels.npy"), mmap_mode="r")
        self.sample_bucket_ids = np.load(os.path.join(root, "sample_bucket_ids.npy"), mmap_mode="r")
        self.sample_bucket_positions = np.load(os.path.join(root, "sample_bucket_positions.npy"), mmap_mode="r")

        self.bucket_infos: Dict[int, QuantBucketInfo] = {}
        for bucket_meta in self.meta["buckets"]:
            bucket_id = int(bucket_meta["bucket_id"])
            bucket_dir = os.path.join(root, bucket_meta["dir"])
            nph = int(bucket_meta["nph"])
            npw = int(bucket_meta["npw"])
            band_files = [
                open(os.path.join(bucket_dir, f"band_{band:02d}.bin"), "rb")
                for band in range(self.num_bands)
            ]
            self.bucket_infos[bucket_id] = QuantBucketInfo(
                bucket_id=bucket_id,
                bucket_dir=bucket_dir,
                nph=nph,
                npw=npw,
                canvas_h=nph * self.block_size,
                canvas_w=npw * self.block_size,
                P=nph * npw,
                num_samples=int(bucket_meta["num_samples"]),
                tile_specs=build_tile_specs(nph, npw, self.tile_blocks),
                sample_ids=np.load(os.path.join(bucket_dir, "sample_ids.npy"), mmap_mode="r"),
                sample_shapes=np.load(os.path.join(bucket_dir, "sample_shapes.npy"), mmap_mode="r"),
                chunk_sizes=np.load(os.path.join(bucket_dir, "chunk_sizes.npy"), mmap_mode="r"),
                offsets=np.load(os.path.join(bucket_dir, "offsets.npy"), mmap_mode="r"),
                lengths=np.load(os.path.join(bucket_dir, "lengths.npy"), mmap_mode="r"),
                raw_lengths=np.load(os.path.join(bucket_dir, "raw_lengths.npy"), mmap_mode="r"),
                patch_signal=np.load(os.path.join(bucket_dir, "patch_signal.npy"), mmap_mode="r"),
                band_files=band_files,
            )

        self.mode = "full"
        self.budget_ratio = 1.0
        self.k_low = 1
        self.patch_policy = "greedy"
        self.band_sensitivity = np.ones(self.num_bands, dtype=np.float32) / self.num_bands
        self.patch_sensitivity_by_bucket: Dict[int, np.ndarray] = {}
        self.reset_epoch_io()
        self.bytes_read_total = 0
        self.full_bytes_total = 0

    def close(self):
        for bucket in self.bucket_infos.values():
            for handle in bucket.band_files:
                handle.close()

    def reset_epoch_io(self):
        self.bytes_read_epoch = 0
        self.full_bytes_epoch = 0
        self.samples_read_epoch = 0
        self.read_ops_epoch = 0
        self.read_bytes_events: List[int] = []

    def get_dataset(self):
        return QuantizedDCTDataset(self)

    def set_full_fidelity(self):
        self.mode = "full"
        self.budget_ratio = 1.0

    def set_budget_ratio(
        self,
        budget_ratio: float,
        patch_sensitivity_by_bucket: Optional[Dict[int, np.ndarray]] = None,
        band_sensitivity: Optional[np.ndarray] = None,
        k_low: int = 1,
        patch_policy: str = "greedy",
    ):
        self.mode = "budget"
        self.budget_ratio = float(max(0.0, min(1.0, budget_ratio)))
        self.k_low = int(min(max(k_low, 0), self.num_bands))
        self.patch_policy = patch_policy
        if patch_sensitivity_by_bucket is not None:
            self.patch_sensitivity_by_bucket = {
                int(k): np.asarray(v, dtype=np.float32).copy()
                for k, v in patch_sensitivity_by_bucket.items()
            }
        if band_sensitivity is not None:
            self.band_sensitivity = np.asarray(band_sensitivity, dtype=np.float32).copy()

    def _resolve_bucket_batch(self, indices: np.ndarray):
        bucket_ids = np.asarray(self.sample_bucket_ids[indices], dtype=np.int64)
        unique = np.unique(bucket_ids)
        if len(unique) != 1:
            raise ValueError(f"Expected same-bucket batch, got {unique.tolist()}")
        bucket_id = int(unique[0])
        local_positions = np.asarray(self.sample_bucket_positions[indices], dtype=np.int64)
        return bucket_id, local_positions

    def _sample_crop_params(self, bucket: QuantBucketInfo, local_positions: np.ndarray, deterministic_val=False):
        params = np.zeros((len(local_positions), 4), dtype=np.int64)
        for row, local_idx in enumerate(local_positions):
            h, w = bucket.sample_shapes[local_idx]
            if deterministic_val:
                crop = min(int(h), int(w))
                top = max((int(h) - crop) // 2, 0)
                left = max((int(w) - crop) // 2, 0)
                params[row] = (top, left, crop, crop)
                continue
            dummy = torch.empty(self.C, int(h), int(w))
            top, left, height, width = T.RandomResizedCrop.get_params(
                dummy, scale=self.crop_scale, ratio=self.crop_ratio
            )
            params[row] = (top, left, height, width)
        return params

    def _build_visible_mask(self, bucket: QuantBucketInfo, crop_params: np.ndarray):
        visible = np.zeros((len(crop_params), bucket.P), dtype=bool)
        for b, (top, left, height, width) in enumerate(crop_params):
            row0 = max(0, int(top) // self.block_size)
            col0 = max(0, int(left) // self.block_size)
            row1 = min(bucket.nph, math.ceil((int(top) + int(height)) / self.block_size))
            col1 = min(bucket.npw, math.ceil((int(left) + int(width)) / self.block_size))
            for row in range(row0, row1):
                visible[b, row * bucket.npw + col0:row * bucket.npw + col1] = True
        return visible

    def _compute_k_allocation(self, bucket_id, bucket, local_positions, visible_mask):
        if self.mode == "full":
            out = np.zeros((len(local_positions), bucket.P), dtype=np.int32)
            out[visible_mask] = self.num_bands
            return out

        out = np.zeros((len(local_positions), bucket.P), dtype=np.int32)
        max_extra = self.num_bands - self.k_low
        band_scores = self.band_sensitivity[self.k_low:] ** 2
        base = self.patch_sensitivity_by_bucket.get(bucket_id)
        if base is None:
            base = np.ones(bucket.P, dtype=np.float32) / bucket.P

        for b in range(len(local_positions)):
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
            elif self.patch_policy == "static":
                patch_scores = base[visible_ids].astype(np.float32)
            else:
                sig = bucket.patch_signal[local_positions[b], visible_ids].astype(np.float32)
                patch_scores = base[visible_ids].astype(np.float32) * sig
            marginals = patch_scores[:, None] * band_scores[None, :]
            flat = marginals.reshape(-1)
            extra_total = min(extra_total, flat.size)
            top_idx = np.argpartition(-flat, extra_total - 1)[:extra_total]
            patch_local = top_idx // max_extra
            np.add.at(out[b], visible_ids[patch_local], 1)
        np.clip(out, 0, self.num_bands, out=out)
        return out

    def _chunk_groups(self, local_positions, bucket):
        chunk_ids = local_positions // self.chunk_size
        for chunk_id in np.unique(chunk_ids):
            rows = np.flatnonzero(chunk_ids == chunk_id)
            local_rows = local_positions[rows] - chunk_id * self.chunk_size
            yield int(chunk_id), rows, local_rows.astype(np.int64)

    def _read_record(self, bucket, chunk_id: int, tile_index: int, band_index: int):
        offset = int(bucket.offsets[chunk_id, tile_index, band_index])
        length = int(bucket.lengths[chunk_id, tile_index, band_index])
        raw_length = int(bucket.raw_lengths[chunk_id, tile_index, band_index])
        handle = bucket.band_files[band_index]
        handle.seek(offset)
        payload = handle.read(length)
        raw = zlib.decompress(payload)
        if len(raw) != raw_length:
            raise IOError("Corrupt quantized DCT record")
        self.bytes_read_epoch += length
        self.bytes_read_total += length
        self.read_ops_epoch += 1
        self.read_bytes_events.append(length)
        return raw

    @torch.no_grad()
    def read_coeffs(self, indices, crop_params=None, deterministic_val=False):
        if torch.is_tensor(indices):
            indices = indices.detach().cpu().numpy()
        else:
            indices = np.asarray(indices, dtype=np.int64)
        bucket_id, local_positions = self._resolve_bucket_batch(indices)
        bucket = self.bucket_infos[bucket_id]
        if crop_params is None:
            crop_params = self._sample_crop_params(bucket, local_positions, deterministic_val)
        else:
            crop_params = np.asarray(crop_params, dtype=np.int64)

        visible_mask = self._build_visible_mask(bucket, crop_params)
        k_alloc = self._compute_k_allocation(bucket_id, bucket, local_positions, visible_mask)
        coeffs_zz = np.zeros((len(indices), bucket.P, self.C, self.total_coeffs), dtype=np.float32)

        for chunk_id, batch_rows, local_rows in self._chunk_groups(local_positions, bucket):
            chunk_n = int(bucket.chunk_sizes[chunk_id])
            for spec in bucket.tile_specs:
                tile_index = int(spec["tile_index"])
                block_ids = np.asarray(spec["block_ids"], dtype=np.int64)
                tile_visible = visible_mask[np.ix_(batch_rows, block_ids)]
                if not tile_visible.any():
                    continue
                for band_index in range(self.num_bands):
                    self.full_bytes_epoch += int(bucket.lengths[chunk_id, tile_index, band_index])
                    self.full_bytes_total += int(bucket.lengths[chunk_id, tile_index, band_index])
                    if not (k_alloc[np.ix_(batch_rows, block_ids)] > band_index).any():
                        continue
                    raw = self._read_record(bucket, chunk_id, tile_index, band_index)
                    band_size = min((band_index + 1) * self.band_size, self.total_coeffs) - band_index * self.band_size
                    arr = np.frombuffer(raw, dtype=np.int8).reshape(chunk_n, len(block_ids), self.C, band_size)
                    selected = arr[local_rows].astype(np.float32)
                    if self.scales.shape[0] == 1:
                        selected *= float(self.scales[0, band_index])
                    else:
                        selected *= self.scales[:, band_index].reshape(1, 1, self.C, 1)
                    start = band_index * self.band_size
                    end = start + band_size
                    coeffs_zz[np.ix_(batch_rows, block_ids, np.arange(self.C), np.arange(start, end))] = selected

        self.samples_read_epoch += len(indices)
        return torch.from_numpy(coeffs_zz), crop_params, visible_mask, k_alloc, bucket_id

    def reconstruct_crops(self, coeffs_zz: torch.Tensor, indices, crop_params, bucket_id: int):
        if torch.is_tensor(indices):
            indices = indices.detach().cpu().numpy()
        else:
            indices = np.asarray(indices, dtype=np.int64)
        bucket_id_check, local_positions = self._resolve_bucket_batch(indices)
        if bucket_id_check != bucket_id:
            raise ValueError("Bucket mismatch during reconstruction")
        bucket = self.bucket_infos[bucket_id]

        coeffs_zz = coeffs_zz.to(self.device, non_blocking=True)
        coeffs_flat = torch.zeros_like(coeffs_zz, device=self.device)
        coeffs_flat[:, :, :, self.zz_torch] = coeffs_zz
        coeffs = coeffs_flat.reshape(coeffs_flat.shape[0], bucket.P, self.C, self.block_size, self.block_size)
        canvas = block_idct2d(coeffs, bucket.nph, bucket.npw)
        crops = []
        for b, (top, left, height, width) in enumerate(crop_params):
            sample_h, sample_w = bucket.sample_shapes[local_positions[b]]
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
        return torch.cat(crops, dim=0).clamp_(0.0, 1.0)

    @torch.no_grad()
    def serve_indices(self, indices, crop_params=None, deterministic_val=False):
        coeffs_zz, crop_params, visible_mask, k_alloc, bucket_id = self.read_coeffs(
            indices, crop_params=crop_params, deterministic_val=deterministic_val
        )
        x = self.reconstruct_crops(coeffs_zz, indices, crop_params, bucket_id)
        return x, crop_params, visible_mask, k_alloc, bucket_id

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
