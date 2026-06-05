#!/usr/bin/env python3
"""Create the Strategy 2 SAM31/SAM3 training notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "strategy2.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip() + "\n"}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n") + "\n",
    }


cells = [
    md(
        """
# CGH Pathology SAM 3.1 Strategy 2

Run this notebook from the folder that contains `dataset/`, `prepare_sam31_dataset.py`, and `requirements_sam31.txt`.

Strategy 2 changes:

- Dataset changes from the first run: use the current QuPath export with 24 canonical tiles instead of the old 20-tile dataset.
- Export logic changes: collect annotations by centroid-in-tile over the full QuPath hierarchy, so nested `yolo_tile_23` and `yolo_tile_24` annotations are included.
- Duplicate helper tiles are excluded: `cell_boundary_clean` and `nuclei_clean` are not training samples.
- Split stays comparable: `p2_tile_05`, `p2_tile_10`, `p2_tile_15`, and `p2_tile_20` remain validation/test; the 4 new YOLO tiles go into train.
- Fine-tuning strategy changes: default to full unfreeze (`SAM3_CGH_FREEZE_BACKBONES=0`) with a lower LR scale.
- Monitoring changes: live plots are drawn inside this notebook from the training log, with optional TensorBoard below.
- Output changes: every run writes to a new `outputs/strategy2_*` folder and does not delete the old `outputs/sam31_runs` training result.
"""
    ),
    code(
        """
from pathlib import Path
from collections import deque
import ast
import json
import os
import re
import subprocess
import sys
import time

PACKAGE_ROOT = Path.cwd().resolve()
assert (PACKAGE_ROOT / "dataset").exists(), f"Start Jupyter from the dataset package root, got {PACKAGE_ROOT}"

SAM3_REPO = Path(os.environ.get("SAM3_REPO", PACKAGE_ROOT.parent / "sam3")).resolve()
HF_MODEL_ID = "facebook/sam3"
DATASET_ROOT = PACKAGE_ROOT / "dataset"
COCO_ROOT = DATASET_ROOT / "coco_sam3" / "cgh_pathology_sam31"
RUN_TAG = os.environ.get("SAM31_RUN_TAG") or time.strftime("strategy2_24tiles_full_unfreeze_%Y%m%d_%H%M%S")
OUTPUT_ROOT = PACKAGE_ROOT / "outputs" / RUN_TAG

print("PACKAGE_ROOT:", PACKAGE_ROOT)
print("SAM3_REPO:", SAM3_REPO)
print("COCO_ROOT:", COCO_ROOT)
print("OUTPUT_ROOT:", OUTPUT_ROOT)
print("Old run folders are not touched. Change SAM31_RUN_TAG or RUN_TAG before training if you want a specific folder name.")
"""
    ),
    md(
        """
## 1. Environment

Install PyTorch separately for the GPU assigned by the cluster. For the Blackwell RTX PRO 6000 node we used CUDA 12.8 wheels; for V100 use CUDA 11.8 wheels.

The dependency cell below intentionally defaults to dry-run mode. Enable it only if this Jupyter kernel is allowed to install packages.
"""
    ),
    code(
        """
INSTALL_DEPS = False
INSTALL_SAM3_EDITABLE = False

if INSTALL_DEPS:
    subprocess.run([sys.executable, "-m", "pip", "install", "--user", "-r", str(PACKAGE_ROOT / "requirements_sam31.txt")], check=True)

if INSTALL_SAM3_EDITABLE:
    if not SAM3_REPO.exists():
        subprocess.run(["git", "clone", "https://github.com/facebookresearch/sam3.git", str(SAM3_REPO)], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--user", "-e", ".[train]"], cwd=SAM3_REPO, check=True)

mods = ["numpy", "pandas", "PIL", "scipy", "pycocotools", "hydra", "submitit", "einops", "wandb", "torch"]
for name in mods:
    try:
        mod = __import__(name)
        version = getattr(mod, "__version__", "ok")
        print(f"{name}: {version}")
    except Exception as exc:
        print(f"{name}: MISSING - {type(exc).__name__}: {exc}")

try:
    import torch
    print("torch cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
except Exception:
    pass
"""
    ),
    md(
        """
## 2. Verify Dataset

This regenerates COCO JSON from the exported masks and checks that the Strategy 2 dataset is present.
"""
    ),
    code(
        """
subprocess.run([sys.executable, str(PACKAGE_ROOT / "prepare_sam31_dataset.py")], check=True)

summary_path = DATASET_ROOT / "sam31_dataset_summary.json"
summary = json.loads(summary_path.read_text())
print(json.dumps(summary, indent=2))

expected = {"images": 24, "train_images": 20, "test_images": 4}
for key, value in expected.items():
    assert summary.get(key) == value, f"Expected {key}={value}, got {summary.get(key)}"
"""
    ),
    code(
        """
import pandas as pd
from IPython.display import Markdown, display

manifest = pd.read_csv(DATASET_ROOT / "sam31_manifest.csv")
display(manifest.groupby(["split", "category", "include_for_training"]).size().rename("n").reset_index())

qc_path = DATASET_ROOT / "metadata" / "DATASET_QC.md"
if qc_path.exists():
    display(Markdown(qc_path.read_text()))
"""
    ),
    md(
        """
## 3. Visual Check

Use this before training. Blue is nucleus, green is trainable clear/compact cell boundary, red is stroma, orange is uncertain/ignore.
"""
    ),
    code(
        """
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

tile_id = "yolo_tile_23"

image = np.asarray(Image.open(DATASET_ROOT / "images" / f"{tile_id}.png").convert("RGB"))
cell_mask = np.asarray(Image.open(DATASET_ROOT / "cell_instance_masks" / f"{tile_id}.png"))
nucleus_mask = np.asarray(Image.open(DATASET_ROOT / "auxiliary_masks" / f"{tile_id}_gt_nucleus_instances.png"))
stroma_mask = np.asarray(Image.open(DATASET_ROOT / "auxiliary_masks" / f"{tile_id}_gt_stroma.png")) > 0
uncertain_mask = np.asarray(Image.open(DATASET_ROOT / "auxiliary_masks" / f"{tile_id}_gt_uncertain_ignore.png")) > 0

overlay = image.astype(float) / 255.0
overlay[nucleus_mask > 0] = overlay[nucleus_mask > 0] * 0.55 + np.array([0.05, 0.25, 1.0]) * 0.45
overlay[cell_mask > 0] = overlay[cell_mask > 0] * 0.55 + np.array([0.0, 0.8, 0.25]) * 0.45
overlay[stroma_mask] = overlay[stroma_mask] * 0.55 + np.array([1.0, 0.1, 0.05]) * 0.45
overlay[uncertain_mask] = overlay[uncertain_mask] * 0.55 + np.array([1.0, 0.6, 0.0]) * 0.45

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].imshow(image)
axes[0].set_title("Raw")
axes[1].imshow(cell_mask, cmap="nipy_spectral")
axes[1].set_title(f"Cell instances: {int(cell_mask.max())}")
axes[2].imshow(overlay)
axes[2].set_title(f"Overlay: {tile_id}")
for ax in axes:
    ax.axis("off")
plt.show()
"""
    ),
    md(
        """
## 4. Verify Hugging Face Access

You need accepted access to `facebook/sam3` and a Hugging Face token on the cluster.
"""
    ),
    code(
        """
try:
    from huggingface_hub import hf_hub_download, model_info
    info = model_info(HF_MODEL_ID)
    hf_hub_download(repo_id=HF_MODEL_ID, filename="config.json")
    print("HF model accessible:", info.modelId)
except Exception as exc:
    print("Could not verify HF access.")
    print("Run: hf auth login")
    print("For fine-grained tokens, enable read access to public gated repos.")
    print(type(exc).__name__, exc)
"""
    ),
    md(
        """
## 5. Create Strategy 2 SAM3 Config

Default Strategy 2 is full unfreeze with lower LR scale. If memory becomes a problem, change `FULL_UNFREEZE = False` and rerun this section.
"""
    ),
    code(
        """
assert SAM3_REPO.exists(), f"SAM3 repo not found: {SAM3_REPO}"

FULL_UNFREEZE = True
LR_SCALE = 0.03 if FULL_UNFREEZE else 0.10
MAX_DATA_EPOCHS = 80
TARGET_EPOCH_SIZE = 1000
VAL_EPOCH_FREQ = 5
TRAIN_BATCH_SIZE = 1
VAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 1

config_proc = subprocess.run(
    [sys.executable, str(PACKAGE_ROOT / "write_sam3_config.py"), "--sam3-repo", str(SAM3_REPO)],
    check=True,
    capture_output=True,
    text=True,
)
CONFIG_PATH = Path(config_proc.stdout.strip())
subprocess.run([sys.executable, str(PACKAGE_ROOT / "patch_sam3_cluster.py"), "--sam3-repo", str(SAM3_REPO)], check=True)

text = CONFIG_PATH.read_text()

def set_line(pattern, replacement, text):
    new_text, n = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if n == 0:
        print("WARNING: did not match", pattern)
    return new_text

text = set_line(r"^  roboflow_vl_100_root: .*$", f"  roboflow_vl_100_root: {json.dumps(str(DATASET_ROOT / 'coco_sam3'))}", text)
text = set_line(r"^  experiment_log_dir: .*$", f"  experiment_log_dir: {json.dumps(str(OUTPUT_ROOT))}", text)
text = set_line(r"^  lr_scale: .*$", f"  lr_scale: {LR_SCALE}", text)
text = set_line(r"^  max_data_epochs: .*$", f"  max_data_epochs: {MAX_DATA_EPOCHS}", text)
text = set_line(r"^  target_epoch_size: .*$", f"  target_epoch_size: {TARGET_EPOCH_SIZE}", text)
text = set_line(r"^  train_batch_size: .*$", f"  train_batch_size: {TRAIN_BATCH_SIZE}", text)
text = set_line(r"^  val_batch_size: .*$", f"  val_batch_size: {VAL_BATCH_SIZE}", text)
text = set_line(r"^  gradient_accumulation_steps: .*$", f"  gradient_accumulation_steps: {GRADIENT_ACCUMULATION_STEPS}", text)
text = set_line(r"^  val_epoch_freq: .*$", f"  val_epoch_freq: {VAL_EPOCH_FREQ}", text)
CONFIG_PATH.write_text(text)

CONFIG_NAME = "configs/cgh_pathology/cgh_pathology_sam31_seg.yaml"
print("Wrote:", CONFIG_PATH)
print("Use config name:", CONFIG_NAME)
print("FULL_UNFREEZE:", FULL_UNFREEZE)
print("LR_SCALE:", LR_SCALE)
print("MAX_DATA_EPOCHS:", MAX_DATA_EPOCHS)
"""
    ),
    code(
        """
config_text = CONFIG_PATH.read_text()
for token in [
    "roboflow_vl_100_root:",
    "experiment_log_dir:",
    "enable_segmentation:",
    "resolution:",
    "max_ann_per_img:",
    "max_train_queries:",
    "max_val_queries:",
    "max_data_epochs:",
    "target_epoch_size:",
    "train_batch_size:",
    "val_batch_size:",
    "val_epoch_freq:",
    "lr_scale:",
    "lr_transformer:",
    "lr_vision_backbone:",
    "lr_language_backbone:",
    "skip_saving_ckpts:",
    "amp_dtype:",
]:
    for line in config_text.splitlines():
        if token in line:
            print(line)
            break

loss_lines = [line.strip() for line in config_text.splitlines() if "SemanticSegCriterion" in line]
print("Loss target(s):")
for line in loss_lines:
    print(" ", line)
"""
    ),
    md(
        """
## 6. Check Running Training Processes

Use this to avoid accidentally running two training jobs on the same GPU.
"""
    ),
    code(
        """
subprocess.run(["bash", "-lc", "ps -ef | grep 'sam3/train/train.py' | grep -v grep || true"], check=True)
try:
    subprocess.run(["nvidia-smi"], check=False)
except FileNotFoundError:
    print("nvidia-smi not found in this environment")
"""
    ),
    md(
        """
## 7. Launch Training With Live Notebook Plots

Set `RUN_TRAINING = True` when ready. The cell streams SAM3 logs, parses key meters, saves `strategy2_live_metrics.jsonl`, and redraws plots in the notebook every few seconds.
"""
    ),
    code(
        """
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output, display

RUN_TRAINING = False
ALLOW_REUSE_OUTPUT_ROOT = False
USE_CLUSTER = 0
NUM_GPUS = 1
NUM_NODES = 1
PARTITION = None
ACCOUNT = None
QOS = None

WANDB_MODE = os.environ.get("WANDB_MODE", "offline")
WANDB_RUN_NAME = os.environ.get("WANDB_RUN_NAME", RUN_TAG)

cmd = [
    sys.executable,
    str(SAM3_REPO / "sam3" / "train" / "train.py"),
    "-c",
    CONFIG_NAME,
    "--use-cluster",
    str(USE_CLUSTER),
    "--num-gpus",
    str(NUM_GPUS),
    "--num-nodes",
    str(NUM_NODES),
]
if PARTITION:
    cmd += ["--partition", PARTITION]
if ACCOUNT:
    cmd += ["--account", ACCOUNT]
if QOS:
    cmd += ["--qos", QOS]

print(" ".join(cmd))
print("WANDB_MODE:", WANDB_MODE)
print("FULL_UNFREEZE env:", "0" if FULL_UNFREEZE else "1")

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
live_log_path = OUTPUT_ROOT / "strategy2_live_training.log"
metrics_jsonl = OUTPUT_ROOT / "strategy2_live_metrics.jsonl"

metric_rows = []
log_tail = deque(maxlen=24)

def gpu_snapshot():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out
    except Exception:
        return "nvidia-smi unavailable"

def parse_metric_payload(line):
    if "Losses and meters:" not in line and "Meters:" not in line:
        return None
    marker = "Losses and meters:" if "Losses and meters:" in line else "Meters:"
    payload = line.split(marker, 1)[1].strip()
    try:
        data = ast.literal_eval(payload)
    except Exception:
        return None
    row = {
        "time": time.time(),
        "epoch": data.get("Trainer/epoch"),
        "where": data.get("Trainer/where"),
        "steps_train": data.get("Trainer/steps_train"),
        "steps_val": data.get("Trainer/steps_val"),
        "train_loss": data.get("Losses/train_all_loss"),
        "train_core_loss": data.get("Losses/train_all_core_loss"),
        "train_semantic_miou": data.get("Losses/train_all_miou_semantic_seg"),
        "train_semantic_seg_loss": data.get("Losses/train_all_loss_semantic_seg"),
        "train_semantic_dice_loss": data.get("Losses/train_all_loss_semantic_dice"),
        "train_mask_loss": data.get("Losses/train_all_loss_mask"),
        "train_dice_loss": data.get("Losses/train_all_loss_dice"),
        "val_bbox_ap": data.get("Meters_train/val_roboflow100/detection/coco_eval_bbox_AP"),
        "val_bbox_ap50": data.get("Meters_train/val_roboflow100/detection/coco_eval_bbox_AP_50"),
        "val_bbox_ap75": data.get("Meters_train/val_roboflow100/detection/coco_eval_bbox_AP_75"),
    }
    return {k: v for k, v in row.items() if v is not None}

def parse_train_epoch_line(line):
    m = re.search(r"Train Epoch:\\s*\\[(\\d+)\\]\\[\\s*(\\d+)/(\\d+)\\].*?Losses/train_all_loss:\\s*([0-9.eE+-]+)", line)
    if not m:
        return None
    epoch, batch, total, loss = m.groups()
    return {
        "time": time.time(),
        "epoch": int(epoch),
        "batch": int(batch),
        "batches_per_epoch": int(total),
        "train_loss": float(loss),
    }

def append_metric(row):
    metric_rows.append(row)
    with metrics_jsonl.open("a") as f:
        f.write(json.dumps(row) + "\\n")

def redraw(returncode=None):
    clear_output(wait=True)
    print("Command:", " ".join(cmd))
    print("Output:", OUTPUT_ROOT)
    print("GPU:", gpu_snapshot())
    if returncode is not None:
        print("Process return code:", returncode)

    if metric_rows:
        df = pd.DataFrame(metric_rows)
        x = df["epoch"] if "epoch" in df else pd.Series(range(len(df)))
        fig, axes = plt.subplots(1, 3, figsize=(17, 4))

        if "train_loss" in df:
            axes[0].plot(x, df["train_loss"], marker="o", linewidth=1)
            axes[0].set_title("Train loss")
            axes[0].set_xlabel("epoch")
            axes[0].grid(True, alpha=0.3)

        for col, label in [
            ("train_semantic_miou", "semantic mIoU"),
            ("val_bbox_ap", "val bbox AP"),
            ("val_bbox_ap50", "val bbox AP50"),
        ]:
            if col in df and df[col].notna().any():
                axes[1].plot(df.loc[df[col].notna(), "epoch"], df.loc[df[col].notna(), col], marker="o", linewidth=1, label=label)
        axes[1].set_title("Quality metrics")
        axes[1].set_xlabel("epoch")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        for col, label in [
            ("train_semantic_dice_loss", "semantic dice loss"),
            ("train_mask_loss", "mask loss"),
            ("train_dice_loss", "dice loss"),
        ]:
            if col in df and df[col].notna().any():
                axes[2].plot(df.loc[df[col].notna(), "epoch"], df.loc[df[col].notna(), col], marker="o", linewidth=1, label=label)
        axes[2].set_title("Segmentation losses")
        axes[2].set_xlabel("epoch")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        plt.tight_layout()
        display(fig)
        plt.close(fig)
        display(df.tail(8))
    else:
        print("No parsed metrics yet. Waiting for SAM3 logger lines.")

    print("\\n--- log tail ---")
    print("\\n".join(log_tail))

if RUN_TRAINING:
    if not ALLOW_REUSE_OUTPUT_ROOT and OUTPUT_ROOT.exists():
        existing = [
            OUTPUT_ROOT / "checkpoints" / "checkpoint.pt",
            OUTPUT_ROOT / "strategy2_live_training.log",
            OUTPUT_ROOT / "strategy2_live_metrics.jsonl",
        ]
        existing = [path for path in existing if path.exists()]
        if existing:
            raise RuntimeError(
                "OUTPUT_ROOT already contains training artifacts. "
                "Change RUN_TAG/SAM31_RUN_TAG for a new folder, or set "
                "ALLOW_REUSE_OUTPUT_ROOT=True if you intentionally want to continue/reuse it. "
                f"Existing: {existing}"
            )

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["SAM3_CGH_FREEZE_BACKBONES"] = "0" if FULL_UNFREEZE else "1"
    env.setdefault("WANDB_PROJECT", "sam31-cgh")
    env["WANDB_RUN_NAME"] = WANDB_RUN_NAME
    env["WANDB_MODE"] = WANDB_MODE

    if metrics_jsonl.exists():
        metrics_jsonl.unlink()
    with live_log_path.open("w") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=SAM3_REPO,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        last_redraw = 0.0
        for line in proc.stdout:
            line = line.rstrip("\\n")
            log_file.write(line + "\\n")
            log_file.flush()
            log_tail.append(line)
            row = parse_metric_payload(line) or parse_train_epoch_line(line)
            if row:
                append_metric(row)
            now = time.time()
            if row or now - last_redraw > 5:
                redraw()
                last_redraw = now
        returncode = proc.wait()
    redraw(returncode=returncode)
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)
else:
    print("Dry run only. Set RUN_TRAINING=True when ready.")
"""
    ),
    md(
        """
## 8. Reload Live Metrics

Use this after interruption or after training finishes to redraw the notebook curves from `strategy2_live_metrics.jsonl`.
"""
    ),
    code(
        """
metrics_jsonl = OUTPUT_ROOT / "strategy2_live_metrics.jsonl"
if not metrics_jsonl.exists():
    print("No metrics JSONL yet:", metrics_jsonl)
else:
    rows = [json.loads(line) for line in metrics_jsonl.read_text().splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    display(df.tail(10))

    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    x = df["epoch"] if "epoch" in df else pd.Series(range(len(df)))
    if "train_loss" in df:
        axes[0].plot(x, df["train_loss"], marker="o")
        axes[0].set_title("Train loss")
    for col, label in [("train_semantic_miou", "semantic mIoU"), ("val_bbox_ap", "val bbox AP"), ("val_bbox_ap50", "val bbox AP50")]:
        if col in df and df[col].notna().any():
            axes[1].plot(df.loc[df[col].notna(), "epoch"], df.loc[df[col].notna(), col], marker="o", label=label)
    axes[1].set_title("Quality metrics")
    axes[1].legend()
    for col, label in [("train_semantic_dice_loss", "semantic dice loss"), ("train_mask_loss", "mask loss"), ("train_dice_loss", "dice loss")]:
        if col in df and df[col].notna().any():
            axes[2].plot(df.loc[df[col].notna(), "epoch"], df.loc[df[col].notna(), col], marker="o", label=label)
    axes[2].set_title("Segmentation losses")
    axes[2].legend()
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("epoch")
    plt.tight_layout()
    plt.show()
"""
    ),
    md(
        """
## 9. TensorBoard In Notebook

This is optional. The live plot above is usually enough, but TensorBoard shows every scalar written by SAM3.
"""
    ),
    code(
        """
TENSORBOARD_LOGDIR = str(OUTPUT_ROOT / "tensorboard")
print(TENSORBOARD_LOGDIR)
%load_ext tensorboard
%tensorboard --logdir $TENSORBOARD_LOGDIR
"""
    ),
    md(
        """
## 10. Checkpoint And Output Summary
"""
    ),
    code(
        """
print("Output root:", OUTPUT_ROOT)
for path in [
    OUTPUT_ROOT / "checkpoints" / "checkpoint.pt",
    OUTPUT_ROOT / "strategy2_live_training.log",
    OUTPUT_ROOT / "strategy2_live_metrics.jsonl",
]:
    print(path, "exists=" + str(path.exists()), "size=" + (str(path.stat().st_size) if path.exists() else "NA"))

if (OUTPUT_ROOT / "checkpoints").exists():
    subprocess.run(["bash", "-lc", f"ls -lh {str(OUTPUT_ROOT / 'checkpoints')!r}"], check=False)
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(notebook, indent=2) + "\n")
print(OUT)
