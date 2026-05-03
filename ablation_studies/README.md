# Ablation Studies

This folder contains ready-to-run ablation launch scripts for the compact
budget sweep used in the paper:

- cap100
- cap90
- cap80
- cap70
- cap60

The intended result root is:

```bash
./results/ablation_studies
```

## Tiny-ImageNet

Tiny-ImageNet uses the TPU/XLA trainer:

```bash
bash ablation_studies/tiny_imagenet/run_tpu_tinyimagenet_cap100.sh
bash ablation_studies/tiny_imagenet/run_tpu_tinyimagenet_cap90.sh
bash ablation_studies/tiny_imagenet/run_tpu_tinyimagenet_cap80.sh
bash ablation_studies/tiny_imagenet/run_tpu_tinyimagenet_cap70.sh
bash ablation_studies/tiny_imagenet/run_tpu_tinyimagenet_cap60.sh
```

Or run the full sweep in sequence:

```bash
bash ablation_studies/tiny_imagenet/run_tpu_tinyimagenet_all.sh
```

The default store path is:

```bash
./data/tiny_imagenet_block_store
```

Override it if your TPU VM uses a mounted disk:

```bash
DATA_DIR=/mnt/dataset_disk/tiny_imagenet_block_store \
bash ablation_studies/tiny_imagenet/run_tpu_tinyimagenet_cap90.sh
```

The cap100 run is the full-budget TDCF pipeline control. It is not the same as
a normal raw-image baseline. The capped runs use the same dynamic budget
schedule family, starting at `beta=0.4` and ramping to the requested cap.

## CIFAR-100

CIFAR-100 uses the existing paper ablation trainer:

```bash
bash ablation_studies/cifar100/run_cifar100_cap100.sh
bash ablation_studies/cifar100/run_cifar100_cap90.sh
bash ablation_studies/cifar100/run_cifar100_cap80.sh
bash ablation_studies/cifar100/run_cifar100_cap70.sh
bash ablation_studies/cifar100/run_cifar100_cap60.sh
```

Or run the full sweep in sequence:

```bash
bash ablation_studies/cifar100/run_cifar100_all.sh
```

Important: the current CIFAR-100 trainer in this repository is the established
PyTorch ablation trainer, not a TPU/XLA trainer. These scripts are meant to be
run on the same machine that runs the CIFAR ablations. I did not fake TPU
support here. If we want CIFAR-100 to execute on TPU cores too, the next clean
step is a dedicated `train_tpu_cifar100.py` trainer.

## Common Overrides

All scripts support simple environment overrides:

```bash
ROOT=./results/ablation_studies \
EPOCHS=100 \
SEED=42 \
bash ablation_studies/tiny_imagenet/run_tpu_tinyimagenet_cap90.sh
```

For TPU Tiny-ImageNet runs, the scripts set:

```bash
PJRT_DEVICE=TPU
PYTHONPATH=.
```

Run one script per terminal or per TPU allocation. Keep the generated
`train.log` files with the result folders; they contain the per-epoch budget,
I/O ratio, and accuracy traces needed for the ablation tables.
