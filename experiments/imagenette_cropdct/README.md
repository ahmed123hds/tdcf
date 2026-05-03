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

## Native JPEG Coefficient Experiment

The CropDCT store above intentionally re-encodes decoded RGB pixels. The next,
more storage-faithful experiment stays inside the original JPEG domain:

```text
original JPEG bytes
-> parse original quantization tables / subsampling / Huffman scans
-> decode native quantized DCT coefficients
-> estimate per-band entropy lower bounds
-> reconstruct from native coefficients and compare to PIL/libjpeg decode
```

This tests whether the right long-term path is a native JPEG coefficient store
rather than a custom RGB-to-DCT re-encoding.

Quick smoke test:

```bash
SIZE=160px MAX_IMAGES=20 FIDELITY_SAMPLES=5 LOG_EVERY=5 \
bash experiments/imagenette_cropdct/run_native_jpeg_experiment.sh
```

Main 1000-image full-size test:

```bash
MAX_IMAGES=1000 FIDELITY_SAMPLES=50 LOG_EVERY=100 \
bash experiments/imagenette_cropdct/run_native_jpeg_experiment.sh
```

Full train split:

```bash
MAX_IMAGES=0 FIDELITY_SAMPLES=100 LOG_EVERY=250 \
bash experiments/imagenette_cropdct/run_native_jpeg_experiment.sh
```

Important: this first native JPEG implementation supports baseline sequential
JPEGs. It reports unsupported files separately. Imagenette samples tested so far
were baseline 4:2:0 JPEGs, which is exactly the common ImageNet-style case.

The key numbers are:

- `native_entropy_lb/original`: theoretical coefficient entropy versus original JPEG bytes.
- `entropy+headers/original`: coefficient entropy plus original JPEG header bytes.
- `raw_int16/original`: how bad raw coefficient storage would be.
- `reconstruct_vs_pil`: sanity check that native coefficients reconstruct close to PIL decode.

## Native JPEG On ImageNet-1K WebDataset

Once the Imagenette result looks promising, run the same native JPEG analysis on
ImageNet-1K WebDataset shards. This does not build a training store yet; it
measures the storage/fidelity limit of using original JPEG coefficients.

Quick 1000-sample train probe:

```bash
MAX_IMAGES=1000 FIDELITY_SAMPLES=50 LOG_EVERY=100 \
bash experiments/imagenette_cropdct/run_native_jpeg_imagenet1k.sh
```

10k train probe:

```bash
MAX_IMAGES=10000 FIDELITY_SAMPLES=100 LOG_EVERY=1000 \
bash experiments/imagenette_cropdct/run_native_jpeg_imagenet1k.sh
```

Validation split:

```bash
SPLIT=val MAX_IMAGES=5000 FIDELITY_SAMPLES=100 LOG_EVERY=500 \
bash experiments/imagenette_cropdct/run_native_jpeg_imagenet1k.sh
```

Override shard paths if needed:

```bash
TRAIN_SHARDS='/mnt/dataset_disk/imagenet_hf/imagenet1k-train-{0000..1023}.tar' \
VAL_SHARDS='/mnt/dataset_disk/imagenet_hf/imagenet1k-validation-{00..63}.tar' \
MAX_IMAGES=10000 \
bash experiments/imagenette_cropdct/run_native_jpeg_imagenet1k.sh
```
