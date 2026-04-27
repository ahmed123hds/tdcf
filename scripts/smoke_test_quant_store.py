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

from tdcf.quantized_store import OriginalQuantizedDCTStore


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--store", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--output_size", type=int, default=224)
    p.add_argument("--budget", type=float, default=0.5)
    args = p.parse_args()

    store = OriginalQuantizedDCTStore(
        args.store,
        device=torch.device("cpu"),
        output_size=args.output_size,
    )
    bucket_id = int(store.sample_bucket_ids[0])
    same_bucket = np.flatnonzero(np.asarray(store.sample_bucket_ids) == bucket_id)
    indices = same_bucket[:min(args.batch_size, len(same_bucket))].astype(np.int64)

    store.set_full_fidelity()
    x, _crop, _visible, _k, _bucket_id = store.serve_indices(indices, deterministic_val=True)
    print(
        f"full: x={tuple(x.shape)} min={float(x.min()):.4f} max={float(x.max()):.4f} "
        f"io={store.get_io_ratio():.3f} stats={store.get_read_stats()}"
    )

    store.reset_epoch_io()
    bucket = store.bucket_infos[bucket_id]
    patch_s = {bucket.bucket_id: np.ones(bucket.P, dtype=np.float32) / bucket.P}
    band_s = np.ones(store.num_bands, dtype=np.float32) / store.num_bands
    store.set_budget_ratio(args.budget, patch_sensitivity_by_bucket=patch_s, band_sensitivity=band_s)
    x, _crop, _visible, k, _bucket_id = store.serve_indices(indices, deterministic_val=False)
    print(
        f"budget: x={tuple(x.shape)} k=[{int(k.min())},{int(k.max())}] "
        f"io={store.get_io_ratio():.3f} stats={store.get_read_stats()}"
    )
    store.close()


if __name__ == "__main__":
    main()
