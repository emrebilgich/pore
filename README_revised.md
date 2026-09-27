# Facial Pore Segmentation Benchmark

This repository contains the code used for the controlled facial pore segmentation benchmark described in the accompanying manuscript.

## Benchmark design

Three finalized dataset collections are evaluated:

- `FINAL_70`
- `FINAL_173`
- `FINAL_1775`

All experiments use the same 15-image locked test set. The remaining images form the development pool and are split into training and validation subsets with `random_state=42` and a 20% validation ratio.

Reported benchmark training/validation counts are:

| Dataset | Development pool | Train | Validation | Locked test |
|---|---:|---:|---:|---:|
| FINAL_70 | 55 | 44 | 11 | 15 |
| FINAL_173 | 157 | 125 | 32 | 15 |
| FINAL_1775 | 1583 | 1266 | 317 | 15 |

The dataset identifiers refer to the source collections; the counts above are the usable benchmark pools after dataset curation.

`FINAL_1775` contains offline augmented images, so it is not a pure sample-count experiment. Differences between the dataset scales reflect both dataset composition and the augmentation history of the source collection.

## Models

1. ResNet34 U-Net with ImageNet-pretrained encoder
2. ResNet34 U-Net with scSE decoder attention
3. Custom five-level Attention U-Net trained from scratch

## Training

- 512 × 512 input resolution
- Batch size: 4
- 40 epochs
- AdamW
- Learning rate: `1e-4`
- Weight decay: `1e-5`
- BCE + Dice loss
- Seed: 42
- Validation-selected threshold search from 0.05 to 0.95 with step 0.01

No random augmentation is applied online during the final benchmark. The offline augmentation is already part of the `FINAL_1775` source collection.

## Reproduction order

### 1. Prepare masks

Convert Roboflow YOLO polygon annotations to binary PNG masks:

```bash
python 01_prepare_dataset.py path/to/yolo_dataset --output-audit audit.json
```

### 2. Build finalized benchmark folders

Use the 15 test images from the 70-image source collection as the locked test set:

```bash
python 02_build_final_datasets.py \
  --source-70 path/to/70_source \
  --source-173 path/to/173_source \
  --source-1775 path/to/1775_source
```

### 3. Train the 9 benchmark configurations

```bash
python 03_train_benchmark.py
```

Each experiment writes its checkpoint, training history, split manifests and metadata under `FINAL_BENCHMARK_RESULTS/`.

### 4. Generate locked-test predictions and visualizations

```bash
python 04_evaluate_locked_test.py
```

This uses the threshold selected on the validation set during training. The locked test set is not used to select the reported threshold.

### 5. Generate the main comparison figure

```bash
python 05_generate_figures.py
```

### 6. Optional exploratory analysis

The training-subset threshold sweep is diagnostic only and is not used as a final performance estimate:

```bash
python 06_exploratory_training_subset.py
```

### 7. Verify the finalized benchmark datasets

```bash
python 07_verify_benchmark.py
```

The verification script checks image/mask counts, locked-test identity and hash-matched train/test overlaps. The hash check is a screening procedure, not a mathematical proof of zero leakage.

## Repository scope

The public benchmark code intentionally excludes earlier notebook cells, API credentials, absolute Windows paths, failed mask-recovery experiments, dummy model definitions, and historical prototype code that was not used to produce the reported benchmark results.

Earlier pseudo-labeling and dataset-development experiments can be retained in a separate `archive/` directory if their history is needed, but they should not be mixed with the final benchmark pipeline.
