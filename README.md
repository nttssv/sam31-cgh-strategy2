# SAM31 CGH Strategy 2 Training Package

This repo contains the Strategy 2 SAM3/SAM31 training package for the CGH pathology tiles.

## What Changed From Strategy 1

- Dataset increased from 20 to 24 canonical training tiles.
- The 4 new tiles are `yolo_tile_21` through `yolo_tile_24`.
- Duplicate/helper QuPath training tiles such as `cell_boundary_clean` and `nuclei_clean` were excluded.
- Export logic uses centroid-in-tile over the full QuPath hierarchy, so nested annotations are included.
- Validation/test split remains comparable: `p2_tile_05`, `p2_tile_10`, `p2_tile_15`, and `p2_tile_20`.
- Training notebook defaults to full unfreeze with `SAM3_CGH_FREEZE_BACKBONES=0`.
- Training output goes into a new timestamped `outputs/strategy2_*` folder and does not overwrite old runs.
- Live plots are shown inside `strategy2.ipynb`.

## Dataset Summary

After extracting the tarball and running `prepare_sam31_dataset.py`:

```json
{
  "images": 24,
  "train_images": 20,
  "test_images": 4,
  "annotations": 1343,
  "manifest_rows": 1571
}
```

## Clone On GPU Cluster

```bash
cd ~/Desktop
git clone git@github.com:nttssv/sam31-cgh-strategy2.git
cd sam31-cgh-strategy2

shasum -a 256 -c SHA256SUMS
tar -xzf sam31_cgh_p2_24tiles_20260605.tar.gz
cd sam31_cgh_p2_24tiles_20260605
```

If SSH is not configured on the cluster, use HTTPS instead:

```bash
git clone https://github.com/nttssv/sam31-cgh-strategy2.git
```

For a private repo, HTTPS cloning requires GitHub authentication or a token.

## Install Dependencies

Install the PyTorch wheel for the assigned GPU first.

Blackwell / CUDA 12.8:

```bash
python -m pip install --user --force-reinstall torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

V100 / CUDA 11.8:

```bash
python -m pip install --user --force-reinstall torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118
```

Then install helper dependencies:

```bash
python -m pip install --user -r requirements_sam31.txt
```

## Run Notebook

Start Jupyter from the extracted folder:

```bash
cd ~/Desktop/sam31-cgh-strategy2/sam31_cgh_p2_24tiles_20260605
jupyter lab
```

Open `strategy2.ipynb`.

Before training, make sure:

- `SAM3_REPO` points to the local SAM3 checkout, or SAM3 is cloned at `../sam3`.
- Hugging Face access to `facebook/sam3` is configured with `hf auth login`.
- No previous training process is already using the GPU.

In the training cell, set:

```python
RUN_TRAINING = True
```

The notebook writes outputs to:

```text
outputs/strategy2_24tiles_full_unfreeze_YYYYMMDD_HHMMSS/
```

Old Strategy 1 outputs are not touched.
