# TDCF

Training-time DCT Fidelity Control for image classification.

This repository studies a simple question: during training, does a model really need the full spatial-frequency content of every image at every step?

TDCF represents images in block-DCT form and serves only part of that representation under an explicit budget. The training loop decides how much fidelity to serve at each epoch, and where to spend that budget across patches and frequency bands.

The project has two goals:

- reduce training-time input budget without giving away much accuracy
- make that reduction concrete enough to study as a systems problem, not only as a masking trick

## What is in the repo

There are three main paths in the codebase.

1. `CIFAR-100` ablations
   - Main entrypoint: [tdcf/train_cifar100.py](/home/filliones/Downloads/Documents/Work/Research/ICLR/tdcf/train_cifar100.py)
   - Convenience runner: [tdcf/scripts/run_cifar100_ablations.sh](/home/filliones/Downloads/Documents/Work/Research/ICLR/tdcf/scripts/run_cifar100_ablations.sh)

2. `Tiny-ImageNet` TPU training with a physical Block-DCT store
   - Main entrypoint: [tdcf/train_tpu_tiny_imagenet.py](/home/filliones/Downloads/Documents/Work/Research/ICLR/tdcf/train_tpu_tiny_imagenet.py)
   - Store preparation: [tdcf/prepare_tiny_imagenet.py](/home/filliones/Downloads/Documents/Work/Research/ICLR/tdcf/prepare_tiny_imagenet.py)

3. `ImageNet-1K` TPU training with online DCT
   - Main entrypoint: [tdcf/train_tpu_imagenet1k.py](/home/filliones/Downloads/Documents/Work/Research/ICLR/tdcf/train_tpu_imagenet1k.py)
   - TPU runners: [tdcf/scripts](/home/filliones/Downloads/Documents/Work/Research/ICLR/tdcf/scripts)

There is also an experimental bucketed physical-store path for ImageNet-1K:

- [tdcf/train_tpu_imagenet1k_store.py](/home/filliones/Downloads/Documents/Work/Research/ICLR/tdcf/train_tpu_imagenet1k_store.py)
- [tdcf/prepare_imagenet1k_crop_store.py](/home/filliones/Downloads/Documents/Work/Research/ICLR/tdcf/prepare_imagenet1k_crop_store.py)

That path is not the recommended main route right now because the full physical store is very storage-hungry at ImageNet scale.

## Current recommendation

For the paper-quality path today:

- use the physical Block-DCT store on `Tiny-ImageNet`
- use the online-DCT trainer on `ImageNet-1K`

That keeps the main ImageNet training recipe fair while still letting us study budgeted fidelity control at scale.

In other words:

- `Tiny-ImageNet` is where exact physical coefficient I/O is currently the cleanest
- `ImageNet-1K` is where the online method is the most practical and mature

## Method summary

The method has four moving parts.

1. `Pilot sensitivity measurement`
   - We run a short pilot phase and measure gradient sensitivity over block-DCT coefficients.
   - This gives us patch-wise and band-wise importance signals.

2. `Budget schedule`
   - Instead of letting fidelity drift upward without control, we use an explicit budget schedule over epochs.
   - The main knobs are `beta`, `max_beta`, and `gamma`.

3. `Allocation policy`
   - Given a budget, the allocator decides how many bands each patch receives.
   - The main policy in this repo is greedy allocation.

4. `Reconstruction and training`
   - We mask the coefficients according to the allocation.
   - Then we reconstruct the image and train the model as usual.

The core idea is not "DCT is special." The core idea is that training-time input fidelity can be scheduled and budgeted.

## Current status

### CIFAR-100

The CIFAR-100 ablation suite is in good shape. The main summary lives at:

- [results/cifar100_ablations/summary.md](/home/filliones/Downloads/Documents/Work/Research/ICLR/results/cifar100_ablations/summary.md)

The current results support three parts of the story:

- a useful accuracy vs I/O tradeoff exists
- dynamic budget scheduling beats a fixed matched budget
- greedy allocation beats a matched random allocation

### Tiny-ImageNet

Tiny-ImageNet has been the most encouraging medium-scale setting so far. It uses the physical BlockBandStore pipeline and is the cleanest place to study exact coefficient serving.

It has also shown the first hints of a regularization-like effect: capped runs can stay close to baseline, and in some settings can slightly outperform it.

### ImageNet-1K

ImageNet-1K currently runs through the online-DCT TPU trainer. This path has gone through several fixes:

- DCT/iDCT moved onto TPU
- input normalization fixed
- validation stabilized
- pilot loader cleanup fixed
- post-pilot policy schedule no longer freezes at the last pilot snapshot
- capped TPU runner scripts now default to milder schedules than the earlier harsh `0.4 -> cap` curriculum

This path is the one to use for the main large-scale experiments today.

## Repo layout

The files below are the ones most people will care about first.

- [tdcf/scheduler.py](/home/filliones/Downloads/Documents/Work/Research/ICLR/tdcf/scheduler.py): budget and fidelity schedulers
- [tdcf/sensitivity.py](/home/filliones/Downloads/Documents/Work/Research/ICLR/tdcf/sensitivity.py): pilot sensitivity estimation
- [tdcf/transforms.py](/home/filliones/Downloads/Documents/Work/Research/ICLR/tdcf/transforms.py): DCT and inverse DCT helpers
- [tdcf/io_dataloader.py](/home/filliones/Downloads/Documents/Work/Research/ICLR/tdcf/io_dataloader.py): physical-store dataloading utilities
- [tdcf/train_cifar100.py](/home/filliones/Downloads/Documents/Work/Research/ICLR/tdcf/train_cifar100.py): CIFAR-100 experiments
- [tdcf/train_tpu_tiny_imagenet.py](/home/filliones/Downloads/Documents/Work/Research/ICLR/tdcf/train_tpu_tiny_imagenet.py): Tiny-ImageNet TPU training
- [tdcf/train_tpu_imagenet1k.py](/home/filliones/Downloads/Documents/Work/Research/ICLR/tdcf/train_tpu_imagenet1k.py): online ImageNet-1K TPU training

## Running the main experiments

### CIFAR-100 ablations

Run the standard ablation suite with:

```bash
bash tdcf/scripts/run_cifar100_ablations.sh cap90
bash tdcf/scripts/run_cifar100_ablations.sh cap80
bash tdcf/scripts/run_cifar100_ablations.sh cap70
bash tdcf/scripts/run_cifar100_ablations.sh fixed90
bash tdcf/scripts/run_cifar100_ablations.sh random90
bash tdcf/scripts/run_cifar100_ablations.sh summary
```

### ImageNet-1K baseline

```bash
bash tdcf/scripts/run_imagenet1k_baseline.sh
```

### ImageNet-1K capped runs

The current TPU runners for the online path are:

```bash
bash tdcf/scripts/run_imagenet1k_cap90.sh
bash tdcf/scripts/run_imagenet1k_cap80.sh
bash tdcf/scripts/run_imagenet1k_cap70.sh
bash tdcf/scripts/run_imagenet1k_cap100.sh
```

The capped runners support environment overrides, for example:

```bash
BETA=0.7 MAX_BETA=0.95 GAMMA=0.75 SAVE_DIR=./results/imagenet1k_cap95_fast \
bash tdcf/scripts/run_imagenet1k_cap90.sh
```

### Deterministic checkpoint evaluation

For ImageNet-1K checkpoint evaluation:

```bash
python3 scripts/eval_imagenet1k_checkpoint.py \
  --checkpoint ./results/imagenet1k_baseline_full_fixed/best.pt \
  --val_shards '/mnt/dataset_disk/imagenet_hf/imagenet1k-validation-{00..63}.tar'
```

## Notes on storage and I/O claims

This repo currently supports two evaluation regimes.

### Physical coefficient-store regime

Used most cleanly on `Tiny-ImageNet`.

- coefficients are precomputed and stored
- the loader physically skips unread coefficient content
- I/O accounting is a true storage-level quantity

### Online transform regime

Used currently for `ImageNet-1K`.

- raw images are loaded first
- block-DCT is applied online on TPU
- budgeting controls the served coefficient content after transform

So on ImageNet-1K, the budget metric is best interpreted as a logical served-coefficient budget rather than a full disk-level coefficient-read measurement.

That distinction matters, and this repository keeps both paths visible so they can be studied honestly.

## Environment

This code assumes a PyTorch-based environment with the usual scientific stack. For TPU runs it also expects `torch_xla`. For ImageNet-1K online runs it expects WebDataset shards.

The key Python dependencies used directly in the code are:

- `torch`
- `torchvision`
- `numpy`
- `webdataset`
- `scikit-learn`
- `torch_xla` for TPU training

## What is still in motion

This is an active research repo. A few pieces are still evolving:

- the best ImageNet-1K budget schedule
- how far the regularization effect holds at larger scale
- whether the physical-store ImageNet path is worth the storage cost
- how to frame exact physical I/O versus logical online budget at ImageNet scale

That uncertainty is real, but the codebase is already useful for running controlled experiments and for reproducing the current ablation story.

## Practical advice

If you are new to the repo, start here:

1. Run the CIFAR-100 ablations and inspect the summary.
2. Read [tdcf/scheduler.py](/home/filliones/Downloads/Documents/Work/Research/ICLR/tdcf/scheduler.py) and [tdcf/sensitivity.py](/home/filliones/Downloads/Documents/Work/Research/ICLR/tdcf/sensitivity.py).
3. Use the online ImageNet-1K TPU runners for large-scale work.
4. Treat the ImageNet physical-store pipeline as a serious extension, not the default workflow.

That path will get you oriented without forcing you through every experimental branch at once.
