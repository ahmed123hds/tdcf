"""Native JPEG coefficient experiment for Imagenette.

This is a research harness, not a production training dataloader. It answers a
specific question:

    If we stay in the original JPEG coefficient domain, can storage approach
    the original JPEG size while preserving decoded-image fidelity?

The implementation parses baseline JPEG files directly, decodes their native
quantized DCT coefficients, preserves original chroma subsampling, estimates
per-band entropy lower bounds, and reconstructs a small subset for a PSNR
sanity check against PIL/libjpeg decoding.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from tdcf.cropdct_store import DEFAULT_BANDS, ycbcr_255_to_rgb
from tdcf.transforms import block_idct2d, zigzag_order


STANDALONE_MARKERS = {0x01, 0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9}
SOF_BASELINE = 0xC0
SOF_PROGRESSIVE = 0xC2
SOS = 0xDA
DQT = 0xDB
DHT = 0xC4
DRI = 0xDD


@dataclass
class FrameComponent:
    cid: int
    h: int
    v: int
    tq: int


@dataclass
class ScanComponent:
    cid: int
    td: int
    ta: int


@dataclass
class JPEGScan:
    components: List[ScanComponent]
    ss: int
    se: int
    ah: int
    al: int
    entropy: bytes


@dataclass
class NativeJPEG:
    path: str
    width: int
    height: int
    precision: int
    sof_marker: int
    components: List[FrameComponent]
    quant_tables_zz: Dict[int, np.ndarray]
    huff_tables: Dict[Tuple[int, int], "HuffmanTable"]
    restart_interval: int
    scans: List[JPEGScan]
    header_bytes: int
    entropy_bytes: int


class HuffmanTable:
    def __init__(self, bits: Sequence[int], values: Sequence[int]):
        self.bits = [int(x) for x in bits]
        self.values = [int(x) for x in values]
        self.lookup: Dict[Tuple[int, int], int] = {}
        code = 0
        idx = 0
        for length, count in enumerate(self.bits, start=1):
            for _ in range(count):
                self.lookup[(length, code)] = self.values[idx]
                idx += 1
                code += 1
            code <<= 1

    def decode(self, reader: "EntropyBitReader") -> int:
        code = 0
        for length in range(1, 17):
            code = (code << 1) | reader.read_bit()
            symbol = self.lookup.get((length, code))
            if symbol is not None:
                return symbol
        raise ValueError("Invalid JPEG Huffman code")


class EntropyBitReader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
        self.bit_buf = 0
        self.bit_count = 0

    def read_byte(self) -> int:
        if self.pos >= len(self.data):
            raise EOFError("Unexpected end of JPEG entropy stream")
        b = self.data[self.pos]
        self.pos += 1
        if b != 0xFF:
            return b
        if self.pos >= len(self.data):
            raise EOFError("Dangling 0xff in JPEG entropy stream")
        marker = self.data[self.pos]
        if marker == 0x00:
            self.pos += 1
            return 0xFF
        if 0xD0 <= marker <= 0xD7:
            raise RuntimeError("Restart marker reached inside a Huffman symbol")
        raise RuntimeError(f"Unexpected marker 0xff{marker:02x} inside entropy stream")

    def read_bit(self) -> int:
        if self.bit_count == 0:
            self.bit_buf = self.read_byte()
            self.bit_count = 8
        self.bit_count -= 1
        return (self.bit_buf >> self.bit_count) & 1

    def read_bits(self, n: int) -> int:
        out = 0
        for _ in range(n):
            out = (out << 1) | self.read_bit()
        return out

    def byte_align(self) -> None:
        self.bit_count = 0

    def consume_restart(self) -> Optional[int]:
        self.byte_align()
        while self.pos < len(self.data) and self.data[self.pos] == 0xFF:
            if self.pos + 1 >= len(self.data):
                return None
            marker = self.data[self.pos + 1]
            if marker == 0xFF:
                self.pos += 1
                continue
            if 0xD0 <= marker <= 0xD7:
                self.pos += 2
                return marker
            if marker == 0x00:
                return None
            return None
        return None


def be16(data: bytes, pos: int) -> int:
    return (data[pos] << 8) | data[pos + 1]


def parse_jpeg(path: str) -> NativeJPEG:
    data = Path(path).read_bytes()
    if len(data) < 4 or data[0:2] != b"\xff\xd8":
        raise ValueError(f"{path} is not a JPEG file")

    quant_tables: Dict[int, np.ndarray] = {}
    huff_tables: Dict[Tuple[int, int], HuffmanTable] = {}
    components: List[FrameComponent] = []
    scans: List[JPEGScan] = []
    width = height = precision = 0
    sof_marker = -1
    restart_interval = 0
    entropy_bytes = 0

    pos = 2
    while pos < len(data):
        if data[pos] != 0xFF:
            raise ValueError(f"Expected JPEG marker at byte {pos} in {path}")
        while pos < len(data) and data[pos] == 0xFF:
            pos += 1
        if pos >= len(data):
            break
        marker = data[pos]
        pos += 1
        if marker == 0xD9:
            break
        if marker in STANDALONE_MARKERS:
            continue
        if pos + 2 > len(data):
            raise ValueError(f"Truncated marker segment in {path}")
        seg_len = be16(data, pos)
        pos += 2
        payload = data[pos:pos + seg_len - 2]
        pos += seg_len - 2

        if marker == DQT:
            p = 0
            while p < len(payload):
                pq_tq = payload[p]
                p += 1
                precision_bits = pq_tq >> 4
                table_id = pq_tq & 0x0F
                if precision_bits == 0:
                    vals = np.frombuffer(payload[p:p + 64], dtype=np.uint8).astype(np.int32)
                    p += 64
                elif precision_bits == 1:
                    vals = np.array([be16(payload, p + 2 * i) for i in range(64)], dtype=np.int32)
                    p += 128
                else:
                    raise ValueError(f"Invalid DQT precision {precision_bits} in {path}")
                quant_tables[table_id] = vals
        elif marker in (SOF_BASELINE, SOF_PROGRESSIVE):
            sof_marker = marker
            precision = payload[0]
            height = be16(payload, 1)
            width = be16(payload, 3)
            ncomp = payload[5]
            components = []
            p = 6
            for _ in range(ncomp):
                cid = payload[p]
                hv = payload[p + 1]
                tq = payload[p + 2]
                p += 3
                components.append(FrameComponent(cid=cid, h=hv >> 4, v=hv & 0x0F, tq=tq))
        elif marker == DHT:
            p = 0
            while p < len(payload):
                tc_th = payload[p]
                p += 1
                table_class = tc_th >> 4
                table_id = tc_th & 0x0F
                bits = list(payload[p:p + 16])
                p += 16
                count = sum(bits)
                values = list(payload[p:p + count])
                p += count
                huff_tables[(table_class, table_id)] = HuffmanTable(bits, values)
        elif marker == DRI:
            restart_interval = be16(payload, 0)
        elif marker == SOS:
            nscan = payload[0]
            p = 1
            scan_components = []
            for _ in range(nscan):
                cid = payload[p]
                tdta = payload[p + 1]
                p += 2
                scan_components.append(ScanComponent(cid=cid, td=tdta >> 4, ta=tdta & 0x0F))
            ss = payload[p]
            se = payload[p + 1]
            ahal = payload[p + 2]
            scan_start = pos
            j = pos
            while j < len(data):
                if data[j] != 0xFF:
                    j += 1
                    continue
                if j + 1 >= len(data):
                    break
                nxt = data[j + 1]
                if nxt == 0x00 or (0xD0 <= nxt <= 0xD7):
                    j += 2
                    continue
                break
            entropy = data[scan_start:j]
            entropy_bytes += len(entropy)
            scans.append(
                JPEGScan(
                    components=scan_components,
                    ss=ss,
                    se=se,
                    ah=ahal >> 4,
                    al=ahal & 0x0F,
                    entropy=entropy,
                )
            )
            pos = j

    return NativeJPEG(
        path=path,
        width=width,
        height=height,
        precision=precision,
        sof_marker=sof_marker,
        components=components,
        quant_tables_zz=quant_tables,
        huff_tables=huff_tables,
        restart_interval=restart_interval,
        scans=scans,
        header_bytes=max(0, len(data) - entropy_bytes),
        entropy_bytes=entropy_bytes,
    )


def receive_extend(reader: EntropyBitReader, size: int) -> int:
    if size == 0:
        return 0
    value = reader.read_bits(size)
    threshold = 1 << (size - 1)
    if value < threshold:
        value -= (1 << size) - 1
    return value


def decode_block(
    reader: EntropyBitReader,
    dc_table: HuffmanTable,
    ac_table: HuffmanTable,
    prev_dc: int,
) -> Tuple[np.ndarray, int]:
    block = np.zeros(64, dtype=np.int16)
    dc_size = dc_table.decode(reader)
    dc = prev_dc + receive_extend(reader, dc_size)
    block[0] = np.clip(dc, -32768, 32767)
    k = 1
    while k < 64:
        rs = ac_table.decode(reader)
        run = rs >> 4
        size = rs & 0x0F
        if size == 0:
            if run == 15:
                k += 16
                continue
            break
        k += run
        if k >= 64:
            break
        block[k] = np.clip(receive_extend(reader, size), -32768, 32767)
        k += 1
    return block, int(dc)


def component_padded_blocks(jpeg: NativeJPEG, comp: FrameComponent) -> Tuple[int, int]:
    max_h = max(c.h for c in jpeg.components)
    max_v = max(c.v for c in jpeg.components)
    mcu_cols = math.ceil(jpeg.width / (8 * max_h))
    mcu_rows = math.ceil(jpeg.height / (8 * max_v))
    return mcu_rows * comp.v, mcu_cols * comp.h


def component_visible_size(jpeg: NativeJPEG, comp: FrameComponent) -> Tuple[int, int]:
    max_h = max(c.h for c in jpeg.components)
    max_v = max(c.v for c in jpeg.components)
    return math.ceil(jpeg.height * comp.v / max_v), math.ceil(jpeg.width * comp.h / max_h)


def decode_baseline_coefficients(jpeg: NativeJPEG) -> Dict[int, np.ndarray]:
    if jpeg.sof_marker != SOF_BASELINE:
        raise ValueError("Only baseline sequential JPEG is supported in this experiment")
    if len(jpeg.scans) != 1:
        raise ValueError("Only single-scan baseline JPEG is supported in this experiment")
    scan = jpeg.scans[0]
    if (scan.ss, scan.se, scan.ah, scan.al) != (0, 63, 0, 0):
        raise ValueError("Only full-range baseline scans are supported")

    comp_by_id = {c.cid: c for c in jpeg.components}
    scan_by_id = {c.cid: c for c in scan.components}
    max_h = max(c.h for c in jpeg.components)
    max_v = max(c.v for c in jpeg.components)
    mcu_cols = math.ceil(jpeg.width / (8 * max_h))
    mcu_rows = math.ceil(jpeg.height / (8 * max_v))

    coeffs: Dict[int, np.ndarray] = {}
    for comp in jpeg.components:
        by, bx = component_padded_blocks(jpeg, comp)
        coeffs[comp.cid] = np.zeros((by, bx, 64), dtype=np.int16)

    reader = EntropyBitReader(scan.entropy)
    prev_dc = {comp.cid: 0 for comp in jpeg.components}
    mcu_seen = 0
    for my in range(mcu_rows):
        for mx in range(mcu_cols):
            for sc in scan.components:
                comp = comp_by_id[sc.cid]
                dc_table = jpeg.huff_tables[(0, sc.td)]
                ac_table = jpeg.huff_tables[(1, sc.ta)]
                for vy in range(comp.v):
                    for hx in range(comp.h):
                        block, dc = decode_block(reader, dc_table, ac_table, prev_dc[comp.cid])
                        prev_dc[comp.cid] = dc
                        by = my * comp.v + vy
                        bx = mx * comp.h + hx
                        coeffs[comp.cid][by, bx, :] = block
            mcu_seen += 1
            if jpeg.restart_interval and mcu_seen % jpeg.restart_interval == 0:
                prev_dc = {comp.cid: 0 for comp in jpeg.components}
                reader.consume_restart()
    return coeffs


def reconstruct_rgb(jpeg: NativeJPEG, coeffs_zz: Dict[int, np.ndarray]) -> torch.Tensor:
    zz = zigzag_order(8, 8).numpy()
    planes = []
    for comp in jpeg.components:
        arr_zz = coeffs_zz[comp.cid].astype(np.float32)
        by, bx, _ = arr_zz.shape
        q_zz = jpeg.quant_tables_zz[comp.tq].astype(np.float32).reshape(1, 1, 64)
        deq_zz = arr_zz * q_zz
        flat = np.zeros_like(deq_zz, dtype=np.float32)
        flat[:, :, zz] = deq_zz
        coeff = torch.from_numpy(flat.reshape(1, by * bx, 1, 8, 8))
        plane = block_idct2d(coeff, by, bx) + 128.0
        vis_h, vis_w = component_visible_size(jpeg, comp)
        plane = plane[:, :, :vis_h, :vis_w]
        if vis_h != jpeg.height or vis_w != jpeg.width:
            plane = F.interpolate(
                plane,
                size=(jpeg.height, jpeg.width),
                mode="bilinear",
                align_corners=False,
            )
        planes.append(plane)
    ycbcr = torch.cat(planes, dim=1)
    if ycbcr.shape[1] == 1:
        return ycbcr.repeat(1, 3, 1, 1).div(255.0).clamp(0.0, 1.0)
    return ycbcr_255_to_rgb(ycbcr)


def psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    mse = float(torch.mean((x - y) ** 2).item())
    return 120.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)


def read_pil_tensor(path: str) -> torch.Tensor:
    with Image.open(path) as img:
        return T.ToTensor()(img.convert("RGB"))


def entropy_bits(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    out = 0.0
    for count in counts.values():
        p = count / total
        out -= p * math.log2(p)
    return out


def image_paths(data_root: str, size: str, split: str, max_images: int) -> List[str]:
    suffix = {"full": "", "320px": "-320", "160px": "-160"}[size]
    root = Path(data_root) / f"imagenette2{suffix}" / split
    paths = sorted(str(p) for p in root.glob("*/*.JPEG"))
    if max_images > 0:
        paths = paths[:max_images]
    if not paths:
        raise RuntimeError(f"No JPEG files found under {root}")
    return paths


def subsampling_key(jpeg: NativeJPEG) -> str:
    if len(jpeg.components) == 1:
        return "grayscale"
    hv = [(c.h, c.v) for c in jpeg.components]
    if hv[:3] == [(2, 2), (1, 1), (1, 1)]:
        return "4:2:0"
    if hv[:3] == [(2, 1), (1, 1), (1, 1)]:
        return "4:2:2"
    if hv[:3] == [(1, 1), (1, 1), (1, 1)]:
        return "4:4:4"
    return ",".join(f"{h}x{v}" for h, v in hv)


def analyze(paths: List[str], args: argparse.Namespace) -> Dict[str, object]:
    band_counts = [Counter() for _ in DEFAULT_BANDS]
    band_symbols = [0 for _ in DEFAULT_BANDS]
    component_symbols = 0
    original_bytes = 0
    entropy_scan_bytes = 0
    header_bytes = 0
    supported = 0
    unsupported: Counter = Counter()
    subsampling: Counter = Counter()
    psnrs: List[float] = []
    maes: List[float] = []
    max_diffs: List[float] = []
    examples = []

    for idx, path in enumerate(paths):
        original_bytes += os.path.getsize(path)
        try:
            jpeg = parse_jpeg(path)
            coeffs = decode_baseline_coefficients(jpeg)
        except Exception as exc:
            unsupported[type(exc).__name__ + ":" + str(exc).split("\n", 1)[0][:80]] += 1
            continue
        supported += 1
        entropy_scan_bytes += jpeg.entropy_bytes
        header_bytes += jpeg.header_bytes
        subsampling[subsampling_key(jpeg)] += 1

        for comp in jpeg.components:
            arr = coeffs[comp.cid]
            component_symbols += int(arr.size)
            for band_id, (b0, b1) in enumerate(DEFAULT_BANDS):
                vals = arr[:, :, b0:b1].reshape(-1)
                values, freqs = np.unique(vals, return_counts=True)
                for value, freq in zip(values, freqs):
                    band_counts[band_id][int(value)] += int(freq)
                band_symbols[band_id] += int(vals.size)

        if len(psnrs) < args.fidelity_samples:
            recon = reconstruct_rgb(jpeg, coeffs).squeeze(0).cpu()
            ref = read_pil_tensor(path)
            if tuple(recon.shape) == tuple(ref.shape):
                diff = (recon - ref).abs()
                psnrs.append(psnr(recon, ref))
                maes.append(float(diff.mean().item()))
                max_diffs.append(float(diff.max().item()))
                if len(examples) < 5:
                    examples.append(
                        {
                            "path": path,
                            "psnr": psnrs[-1],
                            "mae": maes[-1],
                            "max_diff": max_diffs[-1],
                            "subsampling": subsampling_key(jpeg),
                            "bytes": os.path.getsize(path),
                        }
                    )

        if (idx + 1) % args.log_every == 0 or idx + 1 == len(paths):
            print(
                f"[native-jpeg] scanned {idx + 1}/{len(paths)} files | "
                f"supported={supported} unsupported={sum(unsupported.values())}",
                flush=True,
            )

    bands = []
    entropy_lb_total = 0.0
    for band_id, counts in enumerate(band_counts):
        h = entropy_bits(counts)
        lb = h * band_symbols[band_id] / 8.0
        entropy_lb_total += lb
        zero_count = counts.get(0, 0)
        bands.append(
            {
                "band_id": band_id,
                "coeff_range": list(DEFAULT_BANDS[band_id]),
                "symbols": int(band_symbols[band_id]),
                "zero_fraction": 0.0 if band_symbols[band_id] == 0 else zero_count / band_symbols[band_id],
                "entropy_bits_per_symbol": h,
                "entropy_lower_bound_bytes": lb,
            }
        )

    raw_int16_bytes = component_symbols * 2
    fidelity = {
        "samples": len(psnrs),
        "psnr_mean": float(np.mean(psnrs)) if psnrs else 0.0,
        "psnr_std": float(np.std(psnrs)) if psnrs else 0.0,
        "mae_mean": float(np.mean(maes)) if maes else 0.0,
        "mae_std": float(np.std(maes)) if maes else 0.0,
        "max_diff_mean": float(np.mean(max_diffs)) if max_diffs else 0.0,
        "max_diff_std": float(np.std(max_diffs)) if max_diffs else 0.0,
    }
    native_entropy_plus_headers = entropy_lb_total + header_bytes
    return {
        "files": len(paths),
        "supported_baseline_files": supported,
        "unsupported_files": int(sum(unsupported.values())),
        "unsupported_reasons": dict(unsupported),
        "subsampling": dict(subsampling),
        "original_jpeg_bytes": int(original_bytes),
        "original_entropy_scan_bytes": int(entropy_scan_bytes),
        "original_header_bytes": int(header_bytes),
        "native_raw_int16_bytes": int(raw_int16_bytes),
        "native_entropy_lower_bound_bytes": float(entropy_lb_total),
        "native_entropy_plus_original_headers_bytes": float(native_entropy_plus_headers),
        "native_entropy_lb_over_original": 0.0 if original_bytes == 0 else entropy_lb_total / original_bytes,
        "native_entropy_plus_headers_over_original": 0.0
        if original_bytes == 0
        else native_entropy_plus_headers / original_bytes,
        "raw_int16_over_original": 0.0 if original_bytes == 0 else raw_int16_bytes / original_bytes,
        "bands": bands,
        "fidelity_vs_pil_decode": fidelity,
        "examples": examples,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Native JPEG coefficient storage experiment")
    p.add_argument("--data_root", type=str, default="experiments/imagenette_cropdct/work/data")
    p.add_argument("--out_dir", type=str, default="experiments/imagenette_cropdct/work/native_jpeg")
    p.add_argument("--size", choices=["full", "320px", "160px"], default="full")
    p.add_argument("--split", choices=["train", "val"], default="train")
    p.add_argument("--max_images", type=int, default=1000)
    p.add_argument("--fidelity_samples", type=int, default=50)
    p.add_argument("--log_every", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = image_paths(args.data_root, args.size, args.split, args.max_images)
    print("=" * 72)
    print("Native JPEG coefficient experiment")
    print(f"size={args.size} split={args.split} files={len(paths)}")
    print(f"data_root={args.data_root}")
    print("=" * 72)
    report = {
        "args": vars(args),
        "report": analyze(paths, args),
    }
    report_path = out_dir / f"native_jpeg_{args.size}_{args.split}_{len(paths)}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    r = report["report"]
    fid = r["fidelity_vs_pil_decode"]
    print("=" * 72)
    print("Native JPEG report")
    print(
        f"supported={r['supported_baseline_files']}/{r['files']} "
        f"unsupported={r['unsupported_files']}"
    )
    print(f"subsampling={r['subsampling']}")
    print(
        f"original={r['original_jpeg_bytes'] / 1e9:.4f}GB "
        f"entropy_scan={r['original_entropy_scan_bytes'] / 1e9:.4f}GB "
        f"headers={r['original_header_bytes'] / 1e9:.4f}GB"
    )
    print(
        f"native_entropy_lb={r['native_entropy_lower_bound_bytes'] / 1e9:.4f}GB "
        f"entropy+headers={r['native_entropy_plus_original_headers_bytes'] / 1e9:.4f}GB "
        f"raw_int16={r['native_raw_int16_bytes'] / 1e9:.4f}GB"
    )
    print(
        f"entropy_lb/original={r['native_entropy_lb_over_original']:.3f} "
        f"entropy+headers/original={r['native_entropy_plus_headers_over_original']:.3f} "
        f"raw_int16/original={r['raw_int16_over_original']:.3f}"
    )
    print(
        f"reconstruct_vs_pil: PSNR={fid['psnr_mean']:.2f}±{fid['psnr_std']:.2f}dB "
        f"MAE={fid['mae_mean']:.6f} MAX={fid['max_diff_mean']:.6f} "
        f"samples={fid['samples']}"
    )
    print(f"report={report_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()

