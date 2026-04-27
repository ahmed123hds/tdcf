#!/usr/bin/env python3
"""Smoke test the compressed quantized DCT store read path."""

import argparse
import os
import sys
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tdcf.quantized_store import QuantizedFixedDCTStore


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--store", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--output_size", type=int, default=224)
    p.add_argument("--budget", type=float, default=0.5)
    args = p.parse_args()

    store = QuantizedFixedDCTStore(
        args.store,
        device=torch.device("cpu"),
        output_size=args.output_size,
    )
    indices = np.arange(min(args.batch_size, store.N), dtype=np.int64)

    store.set_full_fidelity()
    x, _crop, _visible, _k = store.serve_indices(indices, deterministic_val=True)
    print(
        f"full: x={tuple(x.shape)} min={float(x.min()):.4f} max={float(x.max()):.4f} "
        f"io={store.get_io_ratio():.3f} stats={store.get_read_stats()}"
    )

    store.reset_epoch_io()
    patch_s = np.ones(store.P, dtype=np.float32) / store.P
    band_s = np.ones(store.num_bands, dtype=np.float32) / store.num_bands
    store.set_budget_ratio(args.budget, patch_sensitivity=patch_s, band_sensitivity=band_s)
    x, _crop, _visible, k = store.serve_indices(indices, deterministic_val=False)
    print(
        f"budget: x={tuple(x.shape)} k=[{int(k.min())},{int(k.max())}] "
        f"io={store.get_io_ratio():.3f} stats={store.get_read_stats()}"
    )
    store.close()


if __name__ == "__main__":
    main()
