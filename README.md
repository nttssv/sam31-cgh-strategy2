# SAM31 CGH Strategy 2 Training Package

This repo contains the Strategy 2 SAM3/SAM31 training package for CGH adrenal
pathology tiles. It has been retargeted from the old 24-tile package to the
current full 41-tile CellSeg1 dataset.

## Current Dataset

The expected source dataset is cloned separately from:

```bash
git clone --branch codex/add-second-batch-training-data --single-branch \
  https://github.com/nttssv/training_pa_he_annotation.git \
  ~/Desktop/1.Data/training_pa_he_annotation_full
```

Use this folder as the SAM3 source:

```text
~/Desktop/1.Data/training_pa_he_annotation_full/cellseg1_cgh_p2_combined_41_full
```

Expected source layout:

```text
cellseg1_cgh_p2_combined_41_full/
  train/images/*.png   # 41 images
  train/masks/*.png    # 41 instance masks
  auxiliary_masks/
  semantic_masks/
  previews/
  dataset_manifest.csv
  cell_instances.csv
  boundary_qc.csv
```

`prepare_sam31_dataset.py` stages this source into the legacy Strategy2 package
layout under `dataset/`, then regenerates COCO JSON for SAM3:

```text
dataset/images/
dataset/cell_instance_masks/
dataset/auxiliary_masks/
dataset/metadata/
dataset/coco_sam3/cgh_pathology_sam31/train/_annotations.coco.json
dataset/coco_sam3/cgh_pathology_sam31/test/_annotations.coco.json
```

Default split:

```text
train: 37 images
test:  p2_tile_05, p2_tile_10, p2_tile_15, p2_tile_20
```

Override the validation/test tiles with:

```bash
export SAM31_VAL_TILE_IDS="p2_tile_05,p2_tile_10,p2_tile_15,p2_tile_20"
```

## Clone On GPU Cluster

```bash
cd ~/Desktop
git clone git@github.com:nttssv/sam31-cgh-strategy2.git
cd sam31-cgh-strategy2
```

If SSH is not configured:

```bash
git clone https://github.com/nttssv/sam31-cgh-strategy2.git
```

Clone/update the 41-tile data:

```bash
mkdir -p ~/Desktop/1.Data
cd ~/Desktop/1.Data
git clone --branch codex/add-second-batch-training-data --single-branch \
  https://github.com/nttssv/training_pa_he_annotation.git \
  training_pa_he_annotation_full

find ~/Desktop/1.Data/training_pa_he_annotation_full/cellseg1_cgh_p2_combined_41_full/train/images -type f | wc -l
find ~/Desktop/1.Data/training_pa_he_annotation_full/cellseg1_cgh_p2_combined_41_full/train/masks -type f | wc -l
```

Both counts should be `41`.

## Install Dependencies

Install the PyTorch wheel for the assigned GPU first.

CUDA 12.8:

```bash
python -m pip install --user --force-reinstall torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

CUDA 11.8:

```bash
python -m pip install --user --force-reinstall torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118
```

Then install helper dependencies:

```bash
cd ~/Desktop/sam31-cgh-strategy2
python -m pip install --user -r requirements_sam31.txt
```

SAM3 itself is expected at `../sam3` by default, or set:

```bash
export SAM3_REPO=/path/to/facebookresearch/sam3
```

You also need Hugging Face access to `facebook/sam3`:

```bash
hf auth login
```

## Prepare Dataset

```bash
cd ~/Desktop/sam31-cgh-strategy2
export CGH_SAM31_SOURCE_ROOT="$HOME/Desktop/1.Data/training_pa_he_annotation_full/cellseg1_cgh_p2_combined_41_full"
python prepare_sam31_dataset.py
```

Expected summary:

```json
{
  "images": 41,
  "train_images": 37,
  "test_images": 4
}
```

## Run Notebook

```bash
cd ~/Desktop/sam31-cgh-strategy2
export CGH_SAM31_SOURCE_ROOT="$HOME/Desktop/1.Data/training_pa_he_annotation_full/cellseg1_cgh_p2_combined_41_full"
jupyter lab strategy2.ipynb
```

Before training, make sure:

- `SAM3_REPO` points to a working SAM3 checkout.
- Hugging Face access to `facebook/sam3` is configured.
- No previous `sam3/train/train.py` process is already using the GPU.

In the training cell, set:

```python
RUN_TRAINING = True
```

Outputs are written to:

```text
outputs/strategy2_41tiles_full_unfreeze_YYYYMMDD_HHMMSS/
```

The notebook streams SAM3 logs, parses training loss and segmentation metrics,
writes `strategy2_live_metrics.jsonl`, and draws live plots.

After training, run the notebook's qualitative comparison cell to save
`original / ground truth / SAM3 prediction` panels under the run folder. The
defaults compare the four test tiles with prompts for clear and compact cell
boundaries:

```bash
export SAM31_COMPARE_LIMIT=4
export SAM31_COMPARE_PROMPTS="clear cell boundary,compact cell boundary"
export SAM31_COMPARE_GT_CATEGORIES="clear_cell_boundary,compact_cell_boundary"
export SAM31_COMPARE_SCORE_THRESH=0.30
```

## Legacy 24-Tile Package

The old tarball `sam31_cgh_p2_24tiles_20260605.tar.gz` is retained for
reproducibility. The current notebook and prepare script default to the full
41-tile source dataset when `CGH_SAM31_SOURCE_ROOT` or `CGH_DATASET_ROOT` is
set, or when the default `~/Desktop/1.Data/.../cellseg1_cgh_p2_combined_41_full`
folder exists.
