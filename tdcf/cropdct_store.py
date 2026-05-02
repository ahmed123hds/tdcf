"""CropDCT: crop-native spatial-tile × frequency-band DCT store.

This is the concrete implementation of the CropDCT format:

  image -> original-resolution 8x8 block DCT -> JPEG-like quantized int
  coefficients -> independent compressed chunks for each spatial tile and
  frequency band.

The reader answers crop queries by touching only overlapping spatial tiles and
requested frequency bands. This makes physical read bytes proportional to the
spatial crop and frequency budget.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from tdcf.cropdct_codec import make_codec
from tdcf.transforms import block_dct2d, block_idct2d, zigzag_order


DEFAULT_BANDS = ((0, 1), (1, 6), (6, 21), (21, 64))

LUMA_Q50 = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.float32,
)

CHROMA_Q50 = np.array(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ],
    dtype=np.float32,
)


def jpeg_quality_tables(quality: int, block_size: int = 8) -> np.ndarray:
    if block_size != 8:
        raise ValueError("JPEG quantization tables are defined for block_size=8")
    quality = int(max(1, min(100, quality)))
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    luma = np.floor((LUMA_Q50 * scale + 50) / 100).clip(1, 255)
    chroma = np.floor((CHROMA_Q50 * scale + 50) / 100).clip(1, 255)
    return np.stack([luma, chroma, chroma], axis=0).astype(np.float32)


def rgb_to_ycbcr_255(x: torch.Tensor) -> torch.Tensor:
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0
    return torch.stack([y, cb, cr], dim=1)


def ycbcr_255_to_rgb(x: torch.Tensor) -> torch.Tensor:
    y = x[:, 0]
    cb = x[:, 1] - 128.0
    cr = x[:, 2] - 128.0
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return torch.stack([r, g, b], dim=1).div(255.0).clamp_(0.0, 1.0)


def pad_to_block(x: torch.Tensor, block_size: int) -> torch.Tensor:
    _, _, h, w = x.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")


def tile_specs(nph: int, npw: int, tile_blocks: int) -> List[dict]:
    out = []
    tile_id = 0
    for r0 in range(0, nph, tile_blocks):
        for c0 in range(0, npw, tile_blocks):
            th = min(tile_blocks, nph - r0)
            tw = min(tile_blocks, npw - c0)
            block_ids = [
                (r0 + rr) * npw + (c0 + cc)
                for rr in range(th)
                for cc in range(tw)
            ]
            out.append(
                {
                    "tile_id": tile_id,
                    "r0": r0,
                    "c0": c0,
                    "th": th,
                    "tw": tw,
                    "block_ids": block_ids,
                }
            )
            tile_id += 1
    return out


@lru_cache(maxsize=512)
def cached_tile_specs(nph: int, npw: int, tile_blocks: int) -> Tuple[dict, ...]:
    return tuple(tile_specs(nph, npw, tile_blocks))


def tiles_overlapping_crop(
    nph: int,
    npw: int,
    block_size: int,
    tile_blocks: int,
    crop_box: Tuple[int, int, int, int],
) -> List[int]:
    top, left, height, width = map(int, crop_box)
    row0 = max(0, top // block_size)
    col0 = max(0, left // block_size)
    row1 = min(nph, math.ceil((top + height) / block_size))
    col1 = min(npw, math.ceil((left + width) / block_size))
    tile_cols = math.ceil(npw / tile_blocks)
    tr0 = row0 // tile_blocks
    tc0 = col0 // tile_blocks
    tr1 = math.ceil(row1 / tile_blocks)
    tc1 = math.ceil(col1 / tile_blocks)
    ids = []
    for tr in range(tr0, tr1):
        for tc in range(tc0, tc1):
            ids.append(tr * tile_cols + tc)
    return ids


def random_resized_crop_box(
    height: int,
    width: int,
    scale=(0.08, 1.0),
    ratio=(3.0 / 4.0, 4.0 / 3.0),
) -> Tuple[int, int, int, int]:
    area = height * width
    log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
    for _ in range(10):
        target_area = area * random.uniform(scale[0], scale[1])
        aspect = math.exp(random.uniform(log_ratio[0], log_ratio[1]))
        crop_w = int(round(math.sqrt(target_area * aspect)))
        crop_h = int(round(math.sqrt(target_area / aspect)))
        if 0 < crop_w <= width and 0 < crop_h <= height:
            top = random.randint(0, height - crop_h)
            left = random.randint(0, width - crop_w)
            return top, left, crop_h, crop_w
    crop = min(height, width)
    return (height - crop) // 2, (width - crop) // 2, crop, crop


@dataclass
class CropDCTImage:
    image_id: int
    shard_id: int
    row: int
    label: int
    height: int
    width: int
    nph: int
    npw: int
    num_tiles: int
    chunk_base: int


class CropDCTShard:
    def __init__(self, root: str, shard_meta: dict, codec):
        self.root = root
        self.meta = shard_meta
        self.shard_id = int(shard_meta["shard_id"])
        shard_dir = os.path.join(root, shard_meta["dir"])
        self.payload_path = os.path.join(shard_dir, "payload.bin")
        self.images = np.load(os.path.join(shard_dir, "images.npy"), allow_pickle=False)
        self.chunks = np.load(os.path.join(shard_dir, "chunks.npy"), allow_pickle=False)
        self.codec = codec
        self._payload = None

    def open(self):
        if self._payload is None or self._payload.closed:
            self._payload = open(self.payload_path, "rb", buffering=0)
        return self._payload

    def close(self):
        if self._payload is not None:
            self._payload.close()
            self._payload = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_exc):
        self.close()

    def read_chunk(self, chunk_index: int) -> bytes:
        rec = self.chunks[int(chunk_index)]
        payload = os.pread(self.open().fileno(), int(rec["length"]), int(rec["offset"]))
        return self.codec.decompress(payload, int(rec["raw_length"]))

    def read_span(self, start: int, end: int) -> bytes:
        return os.pread(self.open().fileno(), int(end) - int(start), int(start))


class CropDCTStore:
    def __init__(self, root: str, device: Optional[torch.device] = None):
        self.root = root
        self.device = device or torch.device("cpu")
        with open(os.path.join(root, "metadata.json")) as f:
            self.meta = json.load(f)
        if self.meta.get("format") != "cropdct_v1":
            raise ValueError(f"{root} is not a CropDCT v1 store")
        if self.meta.get("coefficient_dtype") != "int16" or self.meta.get("ac_storage_dtype") != "int16":
            raise ValueError(
                f"{root} uses old CropDCT AC storage "
                f"({self.meta.get('coefficient_dtype')}/{self.meta.get('ac_storage_dtype')}). "
                "Rebuild the CropDCT store with the int16 AC writer."
            )
        self.block_size = int(self.meta["block_size"])
        self.tile_blocks = int(self.meta["tile_blocks"])
        self.band_specs = [tuple(x) for x in self.meta["band_specs"]]
        self.num_bands = len(self.band_specs)
        self.codec = make_codec(self.meta["compression"], self.meta.get("compression_level", 1))
        self.zz = np.asarray(self.meta["zigzag_order"], dtype=np.int64)
        self.qtables = np.load(os.path.join(root, "quant_tables.npy")).astype(np.float32)
        self.qtables_flat = self.qtables.reshape(3, -1)
        self.global_index = np.load(os.path.join(root, "global_index.npy"), allow_pickle=False)
        self.shards = {
            int(s["shard_id"]): CropDCTShard(root, s, self.codec)
            for s in self.meta["shards"]
        }
        if len(self.shards) != len(self.meta["shards"]):
            raise ValueError("CropDCT shard metadata is inconsistent")
        self._full_image_bytes_cache: Dict[int, int] = {}
        self.reset_stats()

    def close(self):
        for shard in self.shards.values():
            shard.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def reset_stats(self):
        self.bytes_read = 0
        self.touched_full_band_bytes = 0
        self.full_image_bytes = 0
        self.read_ops = 0

    def get_read_stats(self):
        return {
            "bytes_read": int(self.bytes_read),
            "touched_full_band_bytes": int(self.touched_full_band_bytes),
            "full_image_bytes": int(self.full_image_bytes),
            "read_ops": int(self.read_ops),
            "crop_io_ratio": 0.0
            if self.touched_full_band_bytes == 0
            else self.bytes_read / self.touched_full_band_bytes,
            "image_io_ratio": 0.0
            if self.full_image_bytes == 0
            else self.bytes_read / self.full_image_bytes,
        }

    def image_record(self, image_id: int) -> CropDCTImage:
        rec = self.global_index[int(image_id)]
        shard = self.shards[int(rec["shard_id"])]
        img = shard.images[int(rec["row"])]
        return CropDCTImage(
            image_id=int(image_id),
            shard_id=int(rec["shard_id"]),
            row=int(rec["row"]),
            label=int(img["label"]),
            height=int(img["height"]),
            width=int(img["width"]),
            nph=int(img["nph"]),
            npw=int(img["npw"]),
            num_tiles=int(img["num_tiles"]),
            chunk_base=int(img["chunk_base"]),
        )

    def _chunk_index(self, image: CropDCTImage, tile_id: int, band_id: int) -> int:
        return image.chunk_base + int(tile_id) * self.num_bands + int(band_id)

    def _full_image_compressed_bytes(self, image: CropDCTImage) -> int:
        cached = self._full_image_bytes_cache.get(image.image_id)
        if cached is not None:
            return cached
        shard = self.shards[image.shard_id]
        total = 0
        for tile_id in range(image.num_tiles):
            for band_id in range(self.num_bands):
                total += int(shard.chunks[self._chunk_index(image, tile_id, band_id)]["length"])
        self._full_image_bytes_cache[image.image_id] = total
        return total

    def read_crop_coeffs(
        self,
        image_id: int,
        crop_box: Tuple[int, int, int, int],
        freq_bands: Sequence[int],
    ) -> Tuple[torch.Tensor, Tuple[int, int, int, int], CropDCTImage]:
        freq_set = frozenset(int(x) for x in freq_bands)
        image = self.image_record(image_id)
        shard = self.shards[image.shard_id]
        tile_ids = tiles_overlapping_crop(
            image.nph, image.npw, self.block_size, self.tile_blocks, crop_box
        )
        specs = cached_tile_specs(image.nph, image.npw, self.tile_blocks)

        top, left, height, width = map(int, crop_box)
        row0 = max(0, top // self.block_size)
        col0 = max(0, left // self.block_size)
        row1 = min(image.nph, math.ceil((top + height) / self.block_size))
        col1 = min(image.npw, math.ceil((left + width) / self.block_size))
        local_nph = max(1, row1 - row0)
        local_npw = max(1, col1 - col0)
        coeffs_zz = np.zeros((local_nph * local_npw, 3, 64), dtype=np.float32)
        self.full_image_bytes += self._full_image_compressed_bytes(image)

        for tile_id in tile_ids:
            spec = specs[int(tile_id)]
            block_ids = np.asarray(spec["block_ids"], dtype=np.int64)
            global_rows = block_ids // image.npw
            global_cols = block_ids % image.npw
            keep = (
                (global_rows >= row0)
                & (global_rows < row1)
                & (global_cols >= col0)
                & (global_cols < col1)
            )
            if not keep.any():
                continue
            kept_cols = np.flatnonzero(keep)
            local_ids = (global_rows[keep] - row0) * local_npw + (global_cols[keep] - col0)
            metas = [
                shard.chunks[self._chunk_index(image, tile_id, band_id)]
                for band_id in range(self.num_bands)
            ]
            for chunk_meta in metas:
                self.touched_full_band_bytes += int(chunk_meta["length"])
            selected_payloads = {}
            selected_bands = [b for b in range(self.num_bands) if b in freq_set]
            for group in _consecutive_groups(selected_bands):
                group_metas = [metas[b] for b in group]
                span_start = min(int(m["offset"]) for m in group_metas)
                span_end = max(int(m["offset"]) + int(m["length"]) for m in group_metas)
                span = shard.read_span(span_start, span_end)
                self.bytes_read += span_end - span_start
                self.read_ops += 1
                for band_id, chunk_meta in zip(group, group_metas):
                    offset = int(chunk_meta["offset"]) - span_start
                    selected_payloads[band_id] = span[offset:offset + int(chunk_meta["length"])]
            for band_id, payload in selected_payloads.items():
                chunk_meta = metas[band_id]
                raw = self.codec.decompress(payload, int(chunk_meta["raw_length"]))
                b0, b1 = self.band_specs[band_id]
                arr = np.frombuffer(raw, dtype=np.int16).reshape(len(block_ids), 3, b1 - b0)
                if band_id == 0:
                    # DC is DPCM-coded across the full tile. We must decode the
                    # whole chain before selecting crop-overlapping blocks.
                    arr = _undpcm_dc(arr).astype(np.float32)
                else:
                    arr = arr.astype(np.float32)
                q = self.qtables_flat[:, self.zz[b0:b1]].reshape(1, 3, b1 - b0)
                coeffs_zz[local_ids, :, b0:b1] = arr[kept_cols] * q
        return torch.from_numpy(coeffs_zz), (row0, col0, local_nph, local_npw), image

    def read_crop(
        self,
        image_id: int,
        crop_box: Optional[Tuple[int, int, int, int]] = None,
        freq_bands: Optional[Sequence[int]] = None,
        output_size: Optional[int] = 224,
    ) -> Tuple[torch.Tensor, int]:
        image = self.image_record(image_id)
        if crop_box is None:
            crop_box = random_resized_crop_box(image.height, image.width)
        if freq_bands is None:
            freq_bands = list(range(self.num_bands))
        coeffs_zz, rect, image = self.read_crop_coeffs(image_id, crop_box, freq_bands)
        row0, col0, local_nph, local_npw = rect
        coeffs_zz = coeffs_zz.to(self.device)
        zz = torch.as_tensor(self.zz, device=self.device, dtype=torch.long)
        coeffs_flat = torch.zeros_like(coeffs_zz, device=self.device)
        coeffs_flat[:, :, zz] = coeffs_zz
        coeffs = coeffs_flat.reshape(1, local_nph * local_npw, 3, self.block_size, self.block_size)
        ycbcr_shifted = block_idct2d(coeffs, local_nph, local_npw) + 128.0
        rgb = ycbcr_255_to_rgb(ycbcr_shifted)

        top, left, height, width = map(int, crop_box)
        local_top = top - row0 * self.block_size
        local_left = left - col0 * self.block_size
        crop = rgb[
            :,
            :,
            local_top:local_top + height,
            local_left:local_left + width,
        ]
        if output_size is not None and (crop.shape[-1] != output_size or crop.shape[-2] != output_size):
            crop = F.interpolate(crop, size=(output_size, output_size), mode="bilinear", align_corners=False)
        return crop.clamp_(0.0, 1.0), image.label


def _dpcm_dc(arr: np.ndarray) -> np.ndarray:
    vals = arr.astype(np.int16, copy=False)
    out = vals.copy()
    out[1:] = vals[1:] - vals[:-1]
    return out


def _undpcm_dc(arr: np.ndarray) -> np.ndarray:
    return np.cumsum(arr.astype(np.int16, copy=False), axis=0).astype(np.int16)


def _consecutive_groups(values: Sequence[int]) -> List[List[int]]:
    groups: List[List[int]] = []
    for value in values:
        value = int(value)
        if not groups or value != groups[-1][-1] + 1:
            groups.append([value])
        else:
            groups[-1].append(value)
    return groups


class CropDCTWriter:
    def __init__(
        self,
        root: str,
        records_per_shard: int = 1024,
        block_size: int = 8,
        tile_blocks: int = 32,
        band_specs: Sequence[Tuple[int, int]] = DEFAULT_BANDS,
        quality: int = 95,
        compression: str = "zstd",
        compression_level: int = 1,
        device: Optional[torch.device] = None,
    ):
        self.root = root
        self.records_per_shard = int(records_per_shard)
        self.block_size = int(block_size)
        self.tile_blocks = int(tile_blocks)
        self.band_specs = [tuple(map(int, x)) for x in band_specs]
        self.quality = int(quality)
        self.compression = compression
        self.compression_level = int(compression_level)
        self.codec = make_codec(compression, level=compression_level)
        self.device = device or torch.device("cpu")
        self.qtables = jpeg_quality_tables(quality, block_size)
        self.qtables_flat = self.qtables.reshape(3, -1)
        self.zz = zigzag_order(block_size, block_size).numpy()
        self.shards_meta = []
        self.global_rows = []
        self._reset_shard(0)
        os.makedirs(root, exist_ok=True)

    def _reset_shard(self, shard_id: int):
        self.shard_id = int(shard_id)
        self.shard_rows = []
        self.chunk_rows = []
        self.payload = None
        self.payload_offset = 0
        self.rows_in_shard = 0

    def _open_payload(self):
        if self.payload is None:
            shard_dir = os.path.join(self.root, f"shard_{self.shard_id:05d}")
            os.makedirs(shard_dir, exist_ok=True)
            self.payload = open(os.path.join(shard_dir, "payload.bin"), "wb")
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    @torch.no_grad()
    def add(self, image: torch.Tensor, label: int, source_key: str = "", source_url: str = ""):
        if image.ndim != 3:
            raise ValueError("CropDCTWriter.add expects CHW image")
        image_id = len(self.global_rows)
        _, height, width = image.shape
        x = image.unsqueeze(0).float().mul(255.0).to(self.device)
        x = rgb_to_ycbcr_255(x) - 128.0
        x = pad_to_block(x, self.block_size)
        nph = x.shape[-2] // self.block_size
        npw = x.shape[-1] // self.block_size
        coeffs = block_dct2d(x, self.block_size)
        flat = coeffs.reshape(nph * npw, 3, 64).detach().cpu().numpy()
        zz_coeffs = flat[:, :, self.zz]
        specs = tile_specs(nph, npw, self.tile_blocks)

        chunk_base = len(self.chunk_rows)
        payload = self._open_payload()
        for spec in specs:
            block_ids = np.asarray(spec["block_ids"], dtype=np.int64)
            tile = zz_coeffs[block_ids]
            for band_id, (b0, b1) in enumerate(self.band_specs):
                q = self.qtables_flat[:, self.zz[b0:b1]].reshape(1, 3, b1 - b0)
                if band_id == 0:
                    quant = np.rint(tile[:, :, b0:b1] / q).clip(-32768, 32767).astype(np.int16)
                    raw_arr = _dpcm_dc(quant)
                else:
                    quant = np.rint(tile[:, :, b0:b1] / q).clip(-32768, 32767).astype(np.int16)
                    raw_arr = quant
                raw = np.ascontiguousarray(raw_arr).tobytes()
                compressed = self.codec.compress(raw)
                offset = self.payload_offset
                payload.write(compressed)
                self.payload_offset += len(compressed)
                self.chunk_rows.append(
                    (
                        int(spec["tile_id"]),
                        int(band_id),
                        int(offset),
                        int(len(compressed)),
                        int(len(raw)),
                        int(len(block_ids)),
                    )
                )
        self.shard_rows.append(
            (
                int(image_id),
                int(label),
                int(height),
                int(width),
                int(nph),
                int(npw),
                int(len(specs)),
                int(chunk_base),
                str(source_key),
                str(source_url),
            )
        )
        self.global_rows.append((int(image_id), int(self.shard_id), int(self.rows_in_shard)))
        self.rows_in_shard += 1
        if self.rows_in_shard >= self.records_per_shard:
            self._finalize_shard()
            self._reset_shard(self.shard_id + 1)

    def _finalize_shard(self):
        if self.rows_in_shard == 0:
            return
        if self.payload is not None:
            self.payload.close()
            self.payload = None
        shard_dir = os.path.join(self.root, f"shard_{self.shard_id:05d}")
        image_dtype = np.dtype(
            [
                ("image_id", "<i8"),
                ("label", "<i8"),
                ("height", "<i4"),
                ("width", "<i4"),
                ("nph", "<i4"),
                ("npw", "<i4"),
                ("num_tiles", "<i4"),
                ("chunk_base", "<i8"),
                ("source_key", "U256"),
                ("source_url", "U512"),
            ]
        )
        chunk_dtype = np.dtype(
            [
                ("tile_id", "<i4"),
                ("band_id", "<i2"),
                ("offset", "<i8"),
                ("length", "<i4"),
                ("raw_length", "<i4"),
                ("num_blocks", "<i4"),
            ]
        )
        np.save(os.path.join(shard_dir, "images.npy"), np.asarray(self.shard_rows, dtype=image_dtype))
        np.save(os.path.join(shard_dir, "chunks.npy"), np.asarray(self.chunk_rows, dtype=chunk_dtype))
        self.shards_meta.append(
            {
                "shard_id": int(self.shard_id),
                "dir": f"shard_{self.shard_id:05d}",
                "num_images": int(self.rows_in_shard),
                "num_chunks": int(len(self.chunk_rows)),
                "payload_bytes": int(self.payload_offset),
            }
        )

    def close(self):
        self._finalize_shard()
        np.save(
            os.path.join(self.root, "quant_tables.npy"),
            self.qtables.astype(np.float32),
        )
        global_dtype = np.dtype([("image_id", "<i8"), ("shard_id", "<i4"), ("row", "<i4")])
        np.save(os.path.join(self.root, "global_index.npy"), np.asarray(self.global_rows, dtype=global_dtype))
        meta = {
            "format": "cropdct_v1",
            "block_size": int(self.block_size),
            "tile_blocks": int(self.tile_blocks),
            "band_specs": [list(x) for x in self.band_specs],
            "quality": int(self.quality),
            "compression": self.compression,
            "compression_level": int(self.compression_level),
            "color_space": "YCbCr",
            "level_shift": 128,
            "coefficient_dtype": "int16",
            "dc_storage_dtype": "int16_dpcm",
            "ac_storage_dtype": "int16",
            "zigzag_order": self.zz.tolist(),
            "records_per_shard": int(self.records_per_shard),
            "num_images": int(len(self.global_rows)),
            "shards": self.shards_meta,
        }
        with open(os.path.join(self.root, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
