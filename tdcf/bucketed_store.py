import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from .transforms import block_idct2d


def bucket_key_from_shape(height: int, width: int, block_size: int) -> Tuple[int, int]:
    return (
        math.ceil(int(height) / block_size),
        math.ceil(int(width) / block_size),
    )


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


@dataclass
class BucketInfo:
    bucket_id: int
    bucket_dir: str
    nph: int
    npw: int
    canvas_h: int
    canvas_w: int
    P: int
    num_samples: int
    tile_blocks: int
    tile_specs: List[dict]
    sample_ids: np.ndarray
    sample_shapes: np.ndarray
    patch_signal: np.ndarray
    bytes_per_sample_full: int


class BucketedTileBandDataset(Dataset):
    def __init__(self, store: "BucketedTileBandStore"):
        self.store = store

    def __len__(self):
        return self.store.N

    def __getitem__(self, idx):
        return int(idx), int(self.store.labels[idx])


class BucketBatchSampler(Sampler[List[int]]):
    """
    Batch sampler that groups samples by bucket so every batch has one patch grid.
    """

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
            int(bucket_id): np.flatnonzero(self.bucket_ids == bucket_id)
            for bucket_id in np.unique(self.bucket_ids)
        }

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _all_batches(self) -> List[List[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        batches: List[List[int]] = []

        for bucket_id, indices in self._indices_by_bucket.items():
            local = indices.copy()
            if self.shuffle:
                rng.shuffle(local)

            if self.drop_last:
                usable = (len(local) // self.batch_size) * self.batch_size
                local = local[:usable]

            for start in range(0, len(local), self.batch_size):
                batch = local[start:start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                if len(batch) == 0:
                    continue
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


class BucketedTileBandStore:
    """
    Bucketed, tiled, band-grouped DCT store for crop-aware training.

    Each image belongs to an exact block-grid bucket `(nph, npw)`. Inside a
    bucket, DCT blocks are stored in small spatial tiles, and each tile is
    split physically by frequency band. A crop query therefore becomes:

      1. resolve sample -> bucket
      2. sample crop rectangle
      3. find intersecting tiles
      4. read only needed tile-band groups
      5. reconstruct and resize

    This keeps exact coefficient-byte accounting while avoiding the infeasible
    "one giant canvas for the whole split" design.
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
        self.output_size = int(output_size)
        self.crop_scale = crop_scale
        self.crop_ratio = crop_ratio

        with open(os.path.join(root, "metadata.json")) as f:
            self.meta = json.load(f)
        if self.meta.get("format") != "bucketed_tile_band_dct":
            raise ValueError(
                f"{root} is not a bucketed tile-band DCT store "
                f"(format={self.meta.get('format')!r})."
            )

        self.N = int(self.meta["N"])
        self.C = int(self.meta["C"])
        self.block_size = int(self.meta["block_size"])
        self.num_bands = int(self.meta["num_bands_per_patch"])
        self.band_size = int(self.meta["band_size"])
        self.total_coeffs = int(self.meta["total_coeffs_per_patch"])
        self.tile_blocks = int(self.meta["tile_blocks"])
        self.resize_shorter = int(self.meta["resize_shorter"])

        self.zz = np.array(self.meta["zigzag_order"], dtype=np.int64)
        self.zz_torch = torch.tensor(self.zz, dtype=torch.long, device=self.device)
        self.labels = np.load(os.path.join(root, "labels.npy"), mmap_mode="r")
        self.sample_bucket_ids = np.load(
            os.path.join(root, "sample_bucket_ids.npy"), mmap_mode="r"
        )
        self.sample_bucket_positions = np.load(
            os.path.join(root, "sample_bucket_positions.npy"), mmap_mode="r"
        )

        self.bucket_infos: Dict[int, BucketInfo] = {}
        self._tile_band_mmaps: Dict[Tuple[int, int, int], np.ndarray] = {}

        for bucket_meta in self.meta["buckets"]:
            bucket_id = int(bucket_meta["bucket_id"])
            bucket_dir = os.path.join(root, bucket_meta["dir"])
            sample_ids = np.load(os.path.join(bucket_dir, "sample_ids.npy"), mmap_mode="r")
            sample_shapes = np.load(os.path.join(bucket_dir, "sample_shapes.npy"), mmap_mode="r")
            patch_signal = np.load(os.path.join(bucket_dir, "patch_signal.npy"), mmap_mode="r")
            tile_specs = build_tile_specs(
                int(bucket_meta["nph"]),
                int(bucket_meta["npw"]),
                self.tile_blocks,
            )
            self.bucket_infos[bucket_id] = BucketInfo(
                bucket_id=bucket_id,
                bucket_dir=bucket_dir,
                nph=int(bucket_meta["nph"]),
                npw=int(bucket_meta["npw"]),
                canvas_h=int(bucket_meta["canvas_h"]),
                canvas_w=int(bucket_meta["canvas_w"]),
                P=int(bucket_meta["P"]),
                num_samples=int(bucket_meta["num_samples"]),
                tile_blocks=self.tile_blocks,
                tile_specs=tile_specs,
                sample_ids=sample_ids,
                sample_shapes=sample_shapes,
                patch_signal=patch_signal,
                bytes_per_sample_full=int(bucket_meta["bytes_per_sample_full"]),
            )

        # Current fidelity state
        self.mode = "full"
        self.budget_ratio = 1.0
        self.k_low = 1
        self.patch_policy = "greedy"
        self.band_sensitivity = np.ones(self.num_bands, dtype=np.float32) / self.num_bands
        self.patch_sensitivity_by_bucket: Dict[int, np.ndarray] = {}
        self.K_high = self.num_bands
        self.q_ratio = 1.0

        # Exact coefficient-byte accounting
        self.bytes_read_epoch = 0
        self.bytes_read_total = 0
        self.full_layout_bytes_epoch = 0
        self.full_layout_bytes_total = 0
        self.samples_read_epoch = 0
        self.read_ops_epoch = 0
        self.read_bytes_events: List[int] = []

    def reset_epoch_io(self):
        self.bytes_read_epoch = 0
        self.full_layout_bytes_epoch = 0
        self.samples_read_epoch = 0
        self.read_ops_epoch = 0
        self.read_bytes_events = []

    def get_dataset(self):
        return BucketedTileBandDataset(self)

    def set_full_fidelity(self):
        self.mode = "full"
        self.budget_ratio = 1.0
        self.k_low = 1
        self.K_high = self.num_bands
        self.q_ratio = 1.0
        self.patch_policy = "greedy"

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

    def set_fidelity(
        self,
        K_high: int,
        K_low: int,
        q_ratio: float,
        patch_sensitivity_by_bucket: Optional[Dict[int, np.ndarray]] = None,
        band_sensitivity: Optional[np.ndarray] = None,
        patch_policy: str = "gradient",
    ):
        self.mode = "fidelity"
        self.K_high = int(min(max(K_high, 1), self.num_bands))
        self.k_low = int(min(max(K_low, 0), self.num_bands))
        self.q_ratio = float(max(0.0, min(1.0, q_ratio)))
        self.patch_policy = patch_policy
        if patch_sensitivity_by_bucket is not None:
            self.patch_sensitivity_by_bucket = {
                int(k): np.asarray(v, dtype=np.float32).copy()
                for k, v in patch_sensitivity_by_bucket.items()
            }
        if band_sensitivity is not None:
            self.band_sensitivity = np.asarray(band_sensitivity, dtype=np.float32).copy()

    def _resolve_bucket_batch(self, indices: np.ndarray) -> Tuple[int, np.ndarray]:
        bucket_ids = np.asarray(self.sample_bucket_ids[indices], dtype=np.int64)
        unique = np.unique(bucket_ids)
        if len(unique) != 1:
            raise ValueError(
                f"BucketedTileBandStore requires same-bucket batches, got buckets {unique.tolist()}"
            )
        bucket_id = int(unique[0])
        local_positions = np.asarray(self.sample_bucket_positions[indices], dtype=np.int64)
        return bucket_id, local_positions

    def _get_tile_band_mmap(self, bucket_id: int, tile_index: int, band_index: int):
        key = (bucket_id, tile_index, band_index)
        mm = self._tile_band_mmaps.get(key)
        if mm is None:
            bucket = self.bucket_infos[bucket_id]
            path = os.path.join(
                bucket.bucket_dir,
                f"tile_{tile_index:03d}_band_{band_index:02d}.npy",
            )
            mm = np.load(path, mmap_mode="r")
            self._tile_band_mmaps[key] = mm
        return mm

    def _sample_crop_params(self, bucket: BucketInfo, local_positions: np.ndarray):
        params = np.zeros((len(local_positions), 4), dtype=np.int64)
        for row, local_idx in enumerate(local_positions):
            h, w = bucket.sample_shapes[local_idx]
            dummy = torch.empty(self.C, int(h), int(w))
            top, left, height, width = T.RandomResizedCrop.get_params(
                dummy, scale=self.crop_scale, ratio=self.crop_ratio
            )
            params[row] = (top, left, height, width)
        return params

    def _center_crop_params(self, bucket: BucketInfo, local_positions: np.ndarray):
        params = np.zeros((len(local_positions), 4), dtype=np.int64)
        for row, local_idx in enumerate(local_positions):
            h, w = bucket.sample_shapes[local_idx]
            top = max((int(h) - self.output_size) // 2, 0)
            left = max((int(w) - self.output_size) // 2, 0)
            height = min(self.output_size, int(h))
            width = min(self.output_size, int(w))
            params[row] = (top, left, height, width)
        return params

    def _patch_bbox_from_crop(
        self,
        bucket: BucketInfo,
        top: int,
        left: int,
        height: int,
        width: int,
    ):
        row0 = max(0, top // self.block_size)
        col0 = max(0, left // self.block_size)
        row1 = min(bucket.nph, math.ceil((top + height) / self.block_size))
        col1 = min(bucket.npw, math.ceil((left + width) / self.block_size))
        return row0, row1, col0, col1

    def _build_visible_mask(self, bucket: BucketInfo, crop_params: np.ndarray):
        B = len(crop_params)
        visible_mask = np.zeros((B, bucket.P), dtype=bool)
        for b, (top, left, height, width) in enumerate(crop_params):
            row0, row1, col0, col1 = self._patch_bbox_from_crop(
                bucket, int(top), int(left), int(height), int(width)
            )
            for row in range(row0, row1):
                start = row * bucket.npw + col0
                end = row * bucket.npw + col1
                visible_mask[b, start:end] = True
        return visible_mask

    def _patch_scores(
        self,
        bucket_id: int,
        bucket: BucketInfo,
        local_positions: np.ndarray,
        visible_ids: np.ndarray,
        sample_row: int,
    ) -> np.ndarray:
        if self.patch_policy == "random":
            return np.random.rand(len(visible_ids)).astype(np.float32)
        if self.patch_policy == "static":
            base = self.patch_sensitivity_by_bucket.get(bucket_id)
            if base is None:
                base = np.ones(bucket.P, dtype=np.float32) / bucket.P
            return base[visible_ids]

        base = self.patch_sensitivity_by_bucket.get(bucket_id)
        if base is None:
            base = np.ones(bucket.P, dtype=np.float32) / bucket.P
        signal = bucket.patch_signal[local_positions[sample_row], visible_ids].astype(np.float32)
        return base[visible_ids] * signal

    def _compute_importance_mask(
        self,
        bucket_id: int,
        bucket: BucketInfo,
        local_positions: np.ndarray,
        visible_mask: np.ndarray,
    ) -> np.ndarray:
        B = len(local_positions)
        important = np.zeros((B, bucket.P), dtype=bool)
        for b in range(B):
            visible_ids = np.flatnonzero(visible_mask[b])
            V = len(visible_ids)
            if V == 0:
                continue
            q_vis = int(round(self.q_ratio * V))
            q_vis = max(0, min(V, q_vis))
            if q_vis == 0:
                continue
            if q_vis >= V:
                important[b, visible_ids] = True
                continue
            scores = self._patch_scores(bucket_id, bucket, local_positions, visible_ids, b)
            top_local = np.argpartition(-scores, q_vis - 1)[:q_vis]
            important[b, visible_ids[top_local]] = True
        return important

    def _compute_k_allocation(
        self,
        bucket_id: int,
        bucket: BucketInfo,
        local_positions: np.ndarray,
        visible_mask: np.ndarray,
    ) -> np.ndarray:
        B = len(local_positions)
        k_alloc = np.zeros((B, bucket.P), dtype=np.int32)

        if self.mode == "full":
            k_alloc[visible_mask] = self.num_bands
            return k_alloc

        if self.mode == "budget":
            if self.patch_policy == "greedy":
                max_extra = self.num_bands - self.k_low
                bs_sq = self.band_sensitivity[self.k_low:] ** 2
                for b in range(B):
                    visible_ids = np.flatnonzero(visible_mask[b])
                    V = len(visible_ids)
                    if V == 0:
                        continue
                    target_budget = int(round(self.budget_ratio * V * self.num_bands))
                    target_budget = max(V * self.k_low, min(V * self.num_bands, target_budget))
                    k_alloc[b, visible_ids] = self.k_low
                    extra_total = target_budget - V * self.k_low
                    if extra_total <= 0 or max_extra <= 0:
                        continue
                    sig_sq = bucket.patch_signal[local_positions[b], visible_ids].astype(np.float32) ** 2
                    marginals = sig_sq[:, np.newaxis] * bs_sq[np.newaxis, :]
                    flat = marginals.reshape(-1)
                    extra_total = min(extra_total, flat.size)
                    if extra_total <= 0:
                        continue
                    top_idx = np.argpartition(-flat, extra_total - 1)[:extra_total]
                    patch_local = top_idx // max_extra
                    np.add.at(k_alloc[b], visible_ids[patch_local], 1)
                np.clip(k_alloc, 0, self.num_bands, out=k_alloc)
                return k_alloc

            important = self._compute_importance_mask(bucket_id, bucket, local_positions, visible_mask)
            k_alloc = np.where(important, self.num_bands, self.k_low).astype(np.int32)
            k_alloc[~visible_mask] = 0
            return k_alloc

        # fidelity mode
        important = self._compute_importance_mask(bucket_id, bucket, local_positions, visible_mask)
        k_alloc = np.where(important, self.K_high, self.k_low).astype(np.int32)
        k_alloc[~visible_mask] = 0
        return k_alloc

    def _read_visible_tiles(
        self,
        bucket_id: int,
        bucket: BucketInfo,
        local_positions: np.ndarray,
        visible_mask: np.ndarray,
        k_alloc: np.ndarray,
    ):
        B = len(local_positions)
        coeffs_zz = np.zeros((B, bucket.P, self.C, self.total_coeffs), dtype=np.float32)

        for spec in bucket.tile_specs:
            tile_index = spec["tile_index"]
            block_ids = np.asarray(spec["block_ids"], dtype=np.int64)
            tile_visible = visible_mask[:, block_ids]
            if not tile_visible.any():
                continue

            rows_full = np.flatnonzero(tile_visible.any(axis=1))
            full_tile_bytes = int(len(rows_full) * spec["blocks_in_tile"] * self.C * self.total_coeffs * 4)
            self.full_layout_bytes_epoch += full_tile_bytes
            self.full_layout_bytes_total += full_tile_bytes

            for band_idx in range(self.num_bands):
                needs_band = (k_alloc[:, block_ids] > band_idx).any(axis=1)
                if not needs_band.any():
                    continue

                rows = np.flatnonzero(needs_band)
                mm = self._get_tile_band_mmap(bucket_id, tile_index, band_idx)
                data = mm[local_positions[rows]]  # (rows, blocks_in_tile, C, band_size)
                start = band_idx * self.band_size
                end = min((band_idx + 1) * self.band_size, self.total_coeffs)
                coeffs_zz[np.ix_(rows, block_ids, np.arange(self.C), np.arange(start, end))] = data

                bytes_read = int(len(rows) * spec["blocks_in_tile"] * self.C * (end - start) * 4)
                self.bytes_read_epoch += bytes_read
                self.bytes_read_total += bytes_read
                self.read_ops_epoch += 1
                self.read_bytes_events.append(bytes_read)

        self.samples_read_epoch += B
        return torch.from_numpy(coeffs_zz)

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

        bucket_id, local_positions = self._resolve_bucket_batch(indices)
        bucket = self.bucket_infos[bucket_id]

        if crop_params is None:
            crop_params = (
                self._center_crop_params(bucket, local_positions)
                if deterministic_val
                else self._sample_crop_params(bucket, local_positions)
            )
        else:
            crop_params = np.asarray(crop_params, dtype=np.int64)

        visible_mask = self._build_visible_mask(bucket, crop_params)
        k_alloc = self._compute_k_allocation(bucket_id, bucket, local_positions, visible_mask)
        coeffs_zz = self._read_visible_tiles(bucket_id, bucket, local_positions, visible_mask, k_alloc)
        return coeffs_zz, crop_params, visible_mask, k_alloc, bucket_id

    def reconstruct_crops(
        self,
        coeffs_zz: torch.Tensor,
        indices,
        crop_params: np.ndarray,
        bucket_id: int,
    ):
        if torch.is_tensor(indices):
            indices = indices.detach().cpu().numpy()
        else:
            indices = np.asarray(indices, dtype=np.int64)

        bucket = self.bucket_infos[bucket_id]
        bucket_id_check, local_positions = self._resolve_bucket_batch(indices)
        if bucket_id_check != bucket_id:
            raise ValueError("Reconstruction bucket mismatch.")

        coeffs_zz = coeffs_zz.to(self.device, non_blocking=True)
        coeffs_flat = torch.zeros_like(coeffs_zz, device=self.device)
        coeffs_flat[:, :, :, self.zz_torch] = coeffs_zz
        coeffs = coeffs_flat.reshape(
            coeffs_flat.shape[0],
            bucket.P,
            self.C,
            self.block_size,
            self.block_size,
        )
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
        return torch.cat(crops, dim=0)

    @torch.no_grad()
    def serve_indices(
        self,
        indices,
        crop_params: Optional[np.ndarray] = None,
        deterministic_val: bool = False,
    ):
        coeffs_zz, crop_params, visible_mask, k_alloc, bucket_id = self.read_coeffs(
            indices, crop_params=crop_params, deterministic_val=deterministic_val
        )
        x = self.reconstruct_crops(coeffs_zz, indices, crop_params, bucket_id)
        return x, crop_params, visible_mask, k_alloc, bucket_id

    def get_io_ratio(self) -> float:
        if self.full_layout_bytes_epoch == 0:
            return 0.0
        return self.bytes_read_epoch / self.full_layout_bytes_epoch

    def get_read_stats(self) -> dict:
        if not self.read_bytes_events:
            return {
                "read_ops_epoch": 0,
                "avg_bytes_per_read": 0.0,
                "p95_bytes_per_read": 0.0,
            }
        arr = np.asarray(self.read_bytes_events, dtype=np.float64)
        return {
            "read_ops_epoch": int(self.read_ops_epoch),
            "avg_bytes_per_read": float(arr.mean()),
            "p95_bytes_per_read": float(np.percentile(arr, 95)),
        }
