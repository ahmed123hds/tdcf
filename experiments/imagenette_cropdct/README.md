# Imagenette CropDCT Local Experiment

This folder is a small local testbed for the CropDCT storage question:

> Can we store variable-size natural images in a DCT-domain format near the
> original JPEG footprint while preserving reconstruction quality around
> 46-47 dB PSNR?

Use this before scaling ideas back to ImageNet-1K. Imagenette full-size is a
good fit because it is a real ImageNet subset with variable JPEG resolutions,
but it is small enough to run on a local machine.

You do not need TensorFlow. The downloader uses `torchvision.datasets.Imagenette`.

## Quick Smoke Run

Use the small 160px variant first to verify the local environment:

```bash
SIZE=160px MAX_TRAIN=20 MAX_VAL=10 FIDELITY_SAMPLES=10 ENTROPY_SAMPLES=10 \
bash experiments/imagenette_cropdct/run_local_imagenette_cropdct.sh
```

This is only a code-path smoke test. It is not the final storage conclusion.

## Recommended Local Run

From the repository root:

```bash
bash experiments/imagenette_cropdct/run_local_imagenette_cropdct.sh
```

The default run downloads Imagenette full-size if needed, builds a 1000-train /
200-val CropDCT subset, validates reconstruction fidelity, and writes a JSON
report.

Outputs stay inside:

```text
experiments/imagenette_cropdct/work/
```

## Full Imagenette Run

After the quick run looks good:

```bash
MAX_TRAIN=0 MAX_VAL=0 FIDELITY_SAMPLES=1000 ENTROPY_SAMPLES=1000 \
bash experiments/imagenette_cropdct/run_local_imagenette_cropdct.sh
```

`MAX_TRAIN=0` and `MAX_VAL=0` mean "use all examples".

## Useful Knobs

```bash
DATA_ROOT=experiments/imagenette_cropdct/work/data
OUT_ROOT=experiments/imagenette_cropdct/work/runs
SIZE=full
QUALITY=95
TILE_BLOCKS=32
RECORDS_PER_SHARD=1024
MAX_TRAIN=1000
MAX_VAL=200
FIDELITY_SAMPLES=200
ENTROPY_SAMPLES=200
DEVICE=cpu
```

## What The Report Means

The script reports:

- Original JPEG bytes for the selected Imagenette samples.
- CropDCT store bytes, including metadata and payload.
- Storage expansion ratio: `CropDCT bytes / original JPEG bytes`.
- Reconstruction quality: PSNR, MAE, max absolute difference.
- Per-band empirical entropy over stored int16 coefficients.
- Current compressed bytes versus an empirical entropy lower bound.

This gives us the practical information-theory view:

```text
How far is the current CropDCT implementation from the best possible
entropy-coded representation under the same quantization and access layout?
```

If the entropy lower bound is close to the original JPEG size, then the path to
equal-size storage is engineering: better variable-length coding and fewer
random-access overheads. If the lower bound is much larger, then equal-size
storage at the same PSNR is not realistic without changing the quantization or
access constraints.
