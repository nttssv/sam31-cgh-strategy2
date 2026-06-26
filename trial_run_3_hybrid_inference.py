#!/usr/bin/env python3
"""Trial Run 3 hybrid inference for CGH adrenal cell boundary segmentation.

Pipeline:
- YOLO segmentation predicts nuclei.
- SAM3.1 predicts clear-cell boundary candidates.
- CellSeg1 predicts dense cell-boundary candidates, used as residual compact
  candidates after removing overlap with SAM3 clear candidates.
- YOLO nuclei gate both SAM3 and CellSeg1 candidates.

This script is intentionally path-defaulted for the SUTD GPU cluster layout used
in this project, while every important path can still be overridden with env vars.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from IPython.display import Image as IPyImage, display
from PIL import Image
from pycocotools import mask as coco_mask
from ultralytics import YOLO


PACKAGE_ROOT = Path(__file__).resolve().parent

COCO_ROOT = Path(
    os.getenv(
        "TRIAL3_COCO_ROOT",
        str(PACKAGE_ROOT / "dataset" / "coco_sam3" / "cgh_pathology_sam31"),
    )
).expanduser()
SAM3_REPO = Path(
    os.getenv("SAM3_REPO", "/home/jovyan/Desktop/sam31-cgh-training-data/sam3")
).expanduser()
SAM31_OUTPUT_ROOT = Path(
    os.getenv(
        "SAM31_OUTPUT_ROOT",
        str(PACKAGE_ROOT / "outputs" / "strategy2_41tiles_full_unfreeze_20260625_160623"),
    )
).expanduser()
SAM31_CHECKPOINT = Path(
    os.getenv("SAM31_CHECKPOINT", str(SAM31_OUTPUT_ROOT / "checkpoints" / "checkpoint.pt"))
).expanduser()
CELLSEG1_REPO = Path(
    os.getenv(
        "CELLSEG1_REPO",
        "/home/jovyan/Desktop/1.Data/training_pa_he_annotation_full/outputs/cellseg1_cluster_live/cellseg1_repo",
    )
).expanduser()
CELLSEG1_RUN_DIR = Path(
    os.getenv(
        "CELLSEG1_RUN_DIR",
        "/home/jovyan/Desktop/1.Data/training_pa_he_annotation_full/outputs/cellseg1_cluster_live/cellseg1_cgh_p2_41full_20260625_124306",
    )
).expanduser()
CELLSEG1_CONFIG_PATH = Path(
    os.getenv("CELLSEG1_CONFIG_PATH", str(CELLSEG1_RUN_DIR / "cellseg1_cgh_p2_runtime_config.yaml"))
).expanduser()
CELLSEG1_LORA = Path(
    os.getenv("CELLSEG1_LORA", str(CELLSEG1_RUN_DIR / "sam_lora_cgh_p2_cell_boundary.pth"))
).expanduser()
YOLO_MODEL_PATH = Path(
    os.getenv(
        "YOLO_MODEL_PATH",
        "/home/jovyan/Desktop/sam31-cgh-training-data/training_data/reference_models/cellseg1_cgh_p2_yolo_best.pt",
    )
).expanduser()

TRIAL3_NAME = os.getenv("TRIAL3_NAME", "trial_run_3_hybrid_selected_tiles")
OUT_DIR = Path(os.getenv("TRIAL3_OUT_DIR", str(SAM31_OUTPUT_ROOT / TRIAL3_NAME))).expanduser()
PRED_MASK_DIR = OUT_DIR / "pred_masks"
COMPARE_DIR = OUT_DIR / "comparison_images"

SPLITS = [token.strip() for token in os.getenv("TRIAL3_SPLITS", "train,test").split(",") if token.strip()]
TRIAL3_TILE_KEYS = [
    token.strip()
    for token in os.getenv(
        "TRIAL3_TILE_KEYS",
        "train_human_compact_tile_001,train_human_compact_tile_002,train_human_compact_tile_003,train_human_compact_tile_004",
    ).split(",")
    if token.strip()
]

SAM31_CLEAR_PROMPTS = [
    token.strip()
    for token in os.getenv(
        "TRIAL3_SAM31_CLEAR_PROMPTS",
        "clear cell boundary,adrenal cortical clear cell boundary",
    ).split(",")
    if token.strip()
]
SAM31_SCORE_THRESH = float(os.getenv("TRIAL3_SAM31_SCORE_THRESH", "0.30"))
SAM31_MIN_AREA = int(os.getenv("TRIAL3_SAM31_MIN_AREA", "80"))

CELLSEG1_IOU_THRESH = float(os.getenv("TRIAL3_CELLSEG1_IOU_THRESH", "0.80"))
CELLSEG1_STABILITY_THRESH = float(os.getenv("TRIAL3_CELLSEG1_STABILITY_THRESH", "0.60"))
CELLSEG1_MIN_AREA = int(os.getenv("TRIAL3_CELLSEG1_MIN_AREA", "80"))
CELLSEG1_MAX_AREA = int(os.getenv("TRIAL3_CELLSEG1_MAX_AREA", "8000"))
MAX_OVERLAP_WITH_SAM_CLEAR = float(os.getenv("TRIAL3_MAX_OVERLAP_WITH_SAM_CLEAR", "0.20"))

YOLO_NUCLEUS_CLASS_ID = int(os.getenv("TRIAL3_YOLO_NUCLEUS_CLASS_ID", "0"))
YOLO_CONF = float(os.getenv("TRIAL3_YOLO_CONF", "0.25"))
YOLO_IOU = float(os.getenv("TRIAL3_YOLO_IOU", "0.50"))
YOLO_MIN_NUCLEUS_AREA = int(os.getenv("TRIAL3_YOLO_MIN_NUCLEUS_AREA", "10"))

MIN_NUCLEUS_OVERLAP_PX = int(os.getenv("TRIAL3_MIN_NUCLEUS_OVERLAP_PX", "5"))
USE_NUCLEUS_CENTROID_GATE = os.getenv("TRIAL3_USE_NUCLEUS_CENTROID_GATE", "1") != "0"
DISPLAY_IMAGES = os.getenv("TRIAL3_DISPLAY_IMAGES", "0") == "1"

GT_COLOR = np.array([0, 180, 120], dtype=np.uint8)
NUCLEUS_COLOR = np.array([255, 0, 0], dtype=np.uint8)
SAM_CLEAR_COLOR = np.array([0, 120, 255], dtype=np.uint8)
CELLSEG_COMPACT_COLOR = np.array([255, 180, 0], dtype=np.uint8)
HYBRID_COLOR = np.array([0, 255, 255], dtype=np.uint8)


def assert_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def load_coco() -> Tuple[Dict[Tuple[str, int], str], Dict[Tuple[str, int], List[dict]], List[dict]]:
    cat_by_split_and_id: Dict[Tuple[str, int], str] = {}
    anns_by_split_and_image: Dict[Tuple[str, int], List[dict]] = {}
    image_infos: List[dict] = []

    for split in SPLITS:
        coco_path = COCO_ROOT / split / "_annotations.coco.json"
        assert_path(coco_path, f"{split} COCO annotations")
        coco = json.loads(coco_path.read_text())
        for cat in coco["categories"]:
            cat_by_split_and_id[(split, int(cat["id"]))] = cat["name"]
        for ann in coco["annotations"]:
            anns_by_split_and_image.setdefault((split, int(ann["image_id"])), []).append(ann)
        for image_info in coco["images"]:
            row = dict(image_info)
            row["_split"] = split
            image_infos.append(row)

    image_infos.sort(key=lambda x: (x["_split"], x["file_name"]))
    return cat_by_split_and_id, anns_by_split_and_image, image_infos


def select_images(image_infos: Sequence[dict]) -> List[dict]:
    wanted = set(TRIAL3_TILE_KEYS)
    selected = []
    for info in image_infos:
        split = info["_split"]
        stem = Path(info["file_name"]).stem
        key = f"{split}_{stem}"
        if key in wanted or stem in wanted:
            selected.append(info)
    if not selected:
        raise RuntimeError(f"No selected trial tiles found: {TRIAL3_TILE_KEYS}")
    return selected


def decode_coco_segmentation(segmentation, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, dict):
        if isinstance(segmentation.get("counts"), list):
            rle = coco_mask.frPyObjects(segmentation, height, width)
        else:
            rle = segmentation
        return coco_mask.decode(rle).astype(bool)
    if isinstance(segmentation, list):
        rles = coco_mask.frPyObjects(segmentation, height, width)
        decoded = coco_mask.decode(rles)
        if decoded.ndim == 3:
            decoded = decoded.any(axis=2)
        return decoded.astype(bool)
    raise TypeError(f"Unsupported COCO segmentation type: {type(segmentation)!r}")


def gt_mask_for_categories(
    image_info: dict,
    categories: set[str],
    cat_by_split_and_id: Dict[Tuple[str, int], str],
    anns_by_split_and_image: Dict[Tuple[str, int], List[dict]],
) -> np.ndarray:
    split = image_info["_split"]
    height, width = int(image_info["height"]), int(image_info["width"])
    out = np.zeros((height, width), dtype=np.uint16)
    label = 1

    for ann in anns_by_split_and_image.get((split, int(image_info["id"])), []):
        cat_name = cat_by_split_and_id.get((split, int(ann["category_id"])), "")
        if cat_name not in categories:
            continue
        mask = decode_coco_segmentation(ann["segmentation"], height, width)
        pixels = mask & (out == 0)
        if int(pixels.sum()) < 20:
            continue
        out[pixels] = label
        label += 1
    return out


def resize_mask_to_image(mask: np.ndarray, img_rgb: np.ndarray) -> np.ndarray:
    height, width = img_rgb.shape[:2]
    if mask.shape[:2] == (height, width):
        return mask
    return cv2.resize(mask.astype(np.uint16), (width, height), interpolation=cv2.INTER_NEAREST)


def mask_boundaries(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    boundary = np.zeros(mask.shape, dtype=bool)
    boundary[1:, :] |= mask[1:, :] != mask[:-1, :]
    boundary[:-1, :] |= mask[1:, :] != mask[:-1, :]
    boundary[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    boundary[:, :-1] |= mask[:, 1:] != mask[:, :-1]
    return boundary & (mask > 0)


def draw_instance_mask(img_rgb: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float = 0.30) -> np.ndarray:
    mask = resize_mask_to_image(mask, img_rgb)
    out = img_rgb.copy()
    overlay = img_rgb.copy()
    overlay[mask > 0] = color
    out[mask_boundaries(mask)] = color

    for label in sorted(int(v) for v in np.unique(mask) if int(v) != 0):
        mask_bin = (mask == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, tuple(map(int, color)), 2)
    return cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)


def binary_metrics(gt_mask: np.ndarray, pred_mask: np.ndarray) -> dict:
    gt = gt_mask > 0
    pred = pred_mask > 0
    tp = int((gt & pred).sum())
    fp = int((~gt & pred).sum())
    fn = int((gt & ~pred).sum())
    tn = int((~gt & ~pred).sum())
    return {
        "tp_px": tp,
        "fp_px": fp,
        "fn_px": fn,
        "tn_px": tn,
        "dice": (2 * tp) / (2 * tp + fp + fn + 1e-8),
        "iou": tp / (tp + fp + fn + 1e-8),
        "precision": tp / (tp + fp + 1e-8),
        "recall_sensitivity": tp / (tp + fn + 1e-8),
        "specificity": tn / (tn + fp + 1e-8),
        "gt_area_px": int(gt.sum()),
        "pred_area_px": int(pred.sum()),
        "gt_instances": int(gt_mask.max()),
        "pred_instances": int(pred_mask.max()),
    }


def relabel_instance_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=np.uint16)
    label = 1
    for old in sorted(int(v) for v in np.unique(mask) if int(v) != 0):
        pixels = mask == old
        if int(pixels.sum()) == 0:
            continue
        out[pixels] = label
        label += 1
    return out


def centroid_of_mask(mask_bool: np.ndarray) -> Tuple[int, int] | None:
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return None
    return int(np.mean(xs)), int(np.mean(ys))


def cell_has_nucleus(cell_bool: np.ndarray, nucleus_mask: np.ndarray) -> Tuple[bool, int, str]:
    nucleus_binary = nucleus_mask > 0
    overlap_px = int((cell_bool & nucleus_binary).sum())
    if overlap_px >= MIN_NUCLEUS_OVERLAP_PX:
        return True, overlap_px, "overlap"

    if USE_NUCLEUS_CENTROID_GATE:
        for nuc_label in sorted(int(v) for v in np.unique(nucleus_mask) if int(v) != 0):
            nucleus = nucleus_mask == nuc_label
            centroid = centroid_of_mask(nucleus)
            if centroid is None:
                continue
            x, y = centroid
            if 0 <= y < cell_bool.shape[0] and 0 <= x < cell_bool.shape[1] and cell_bool[y, x]:
                return True, overlap_px, "centroid"
    return False, overlap_px, "none"


def yolo_nucleus_mask(yolo_model: YOLO, image_path: Path, image_rgb: np.ndarray, tile_key: str = "") -> Tuple[np.ndarray, List[dict]]:
    result = yolo_model.predict(
        source=str(image_path),
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        verbose=False,
        retina_masks=True,
    )[0]

    height, width = image_rgb.shape[:2]
    nucleus_mask = np.zeros((height, width), dtype=np.uint16)
    rows = []
    next_label = 1

    if result.masks is None:
        print(f"{tile_key} | YOLO nucleus: none")
        return nucleus_mask, rows

    masks = result.masks.data.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    confs = result.boxes.conf.detach().cpu().numpy()

    for idx, raw_mask in enumerate(masks):
        cls = int(classes[idx])
        conf = float(confs[idx])
        if cls != YOLO_NUCLEUS_CLASS_ID:
            continue
        binary = raw_mask > 0.5
        binary = resize_mask_to_image(binary.astype(np.uint8), image_rgb) > 0
        area = int(binary.sum())
        if area < YOLO_MIN_NUCLEUS_AREA:
            continue
        pixels = binary & (nucleus_mask == 0)
        if int(pixels.sum()) < YOLO_MIN_NUCLEUS_AREA:
            continue
        nucleus_mask[pixels] = next_label
        rows.append(
            {
                "source_model": "YOLO",
                "class_name": "nucleus",
                "label": next_label,
                "score": conf,
                "area_px": int(pixels.sum()),
            }
        )
        next_label += 1

    print(f"{tile_key} | YOLO nucleus kept={int(nucleus_mask.max())}")
    return nucleus_mask, rows


def sam31_autocast_context():
    if not torch.cuda.is_available():
        return nullcontext()
    raw_dtype = os.getenv("TRIAL3_SAM31_AMP_DTYPE", "float16").lower()
    dtype = torch.float16 if raw_dtype in {"fp16", "float16", "half"} else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def masks_to_numpy_stack(masks) -> np.ndarray:
    if hasattr(masks, "detach"):
        arr = masks.detach().float().cpu().numpy()
    else:
        arr = np.asarray(masks)
    arr = np.squeeze(arr)
    if arr.size == 0:
        return np.zeros((0, 1, 1), dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim > 3:
        arr = arr.reshape((-1,) + arr.shape[-2:])
    return arr


def scores_to_numpy(scores, n: int) -> np.ndarray:
    if scores is None:
        return np.ones(n)
    if hasattr(scores, "detach"):
        arr = scores.detach().float().cpu().numpy()
    else:
        arr = np.asarray(scores, dtype=float)
    arr = np.squeeze(arr).reshape(-1)
    if arr.size < n:
        arr = np.pad(arr, (0, n - arr.size), constant_values=1.0)
    return arr[:n]


def sam3_clear_mask(processor, pil_image: Image.Image, nucleus_mask: np.ndarray, tile_key: str = "") -> Tuple[np.ndarray, List[dict]]:
    with torch.inference_mode(), sam31_autocast_context():
        state = processor.set_image(pil_image)

    height, width = pil_image.height, pil_image.width
    pred = np.zeros((height, width), dtype=np.uint16)
    rows = []
    next_label = 1

    for prompt in SAM31_CLEAR_PROMPTS:
        with torch.inference_mode(), sam31_autocast_context():
            output = processor.set_text_prompt(state=state, prompt=prompt)

        mask_stack = masks_to_numpy_stack(output.get("masks", []))
        scores = scores_to_numpy(output.get("scores"), len(mask_stack))
        kept = 0
        removed_no_nucleus = 0

        for idx, (raw_mask, score) in enumerate(zip(mask_stack, scores)):
            if float(score) < SAM31_SCORE_THRESH:
                continue
            binary = raw_mask > 0
            binary = resize_mask_to_image(binary.astype(np.uint8), np.asarray(pil_image)) > 0
            pixels = binary & (pred == 0)
            area = int(pixels.sum())
            if area < SAM31_MIN_AREA:
                continue
            has_nuc, nuc_overlap, gate_type = cell_has_nucleus(pixels, nucleus_mask)
            if not has_nuc:
                removed_no_nucleus += 1
                continue
            pred[pixels] = next_label
            rows.append(
                {
                    "source_model": "SAM3.1",
                    "class_name": "clear_cell_boundary",
                    "label": next_label,
                    "prompt": prompt,
                    "score": float(score),
                    "area_px": area,
                    "nucleus_overlap_px": nuc_overlap,
                    "nucleus_gate_type": gate_type,
                }
            )
            next_label += 1
            kept += 1

        print(
            f"{tile_key} | SAM3.1 clear | prompt={prompt!r} raw={len(mask_stack)} "
            f"kept={kept} removed_no_nucleus={removed_no_nucleus}"
        )
    return pred, rows


def cellseg1_predict_one(image_path: Path, cellseg_config: dict, read_image_to_numpy, resize_image, predict_images) -> Tuple[np.ndarray, np.ndarray]:
    image = resize_image(read_image_to_numpy(image_path), cellseg_config["resize_size"])
    pred = predict_images(cellseg_config, [image])[0]
    return image, pred.astype(np.uint16)


def filter_cellseg_compact_candidates(
    cellseg_mask: np.ndarray,
    sam_clear_mask: np.ndarray,
    nucleus_mask: np.ndarray,
    image_rgb: np.ndarray,
) -> Tuple[np.ndarray, List[dict]]:
    cellseg_mask = resize_mask_to_image(cellseg_mask, image_rgb)
    sam_clear_mask = resize_mask_to_image(sam_clear_mask, image_rgb)
    nucleus_mask = resize_mask_to_image(nucleus_mask, image_rgb)

    compact = np.zeros(cellseg_mask.shape, dtype=np.uint16)
    rows = []
    next_label = 1
    sam_clear_binary = sam_clear_mask > 0
    removed_no_nucleus = 0
    removed_overlap_clear = 0
    removed_area = 0

    for old_label in sorted(int(v) for v in np.unique(cellseg_mask) if int(v) != 0):
        candidate = cellseg_mask == old_label
        area = int(candidate.sum())
        if area < CELLSEG1_MIN_AREA or area > CELLSEG1_MAX_AREA:
            removed_area += 1
            continue
        overlap = int((candidate & sam_clear_binary).sum())
        overlap_ratio = overlap / (area + 1e-8)
        if overlap_ratio > MAX_OVERLAP_WITH_SAM_CLEAR:
            removed_overlap_clear += 1
            continue
        has_nuc, nuc_overlap, gate_type = cell_has_nucleus(candidate, nucleus_mask)
        if not has_nuc:
            removed_no_nucleus += 1
            continue
        compact[candidate] = next_label
        rows.append(
            {
                "source_model": "CellSeg1",
                "class_name": "compact_cell_boundary",
                "label": next_label,
                "prompt": "cellseg1_residual_mask",
                "score": np.nan,
                "area_px": area,
                "overlap_with_sam_clear_ratio": overlap_ratio,
                "nucleus_overlap_px": nuc_overlap,
                "nucleus_gate_type": gate_type,
            }
        )
        next_label += 1

    print(
        f"CellSeg1 residual compact kept={int(compact.max())} "
        f"removed_no_nucleus={removed_no_nucleus} "
        f"removed_overlap_clear={removed_overlap_clear} removed_area={removed_area}"
    )
    return compact, rows


def combine_clear_and_compact(sam_clear: np.ndarray, compact_candidate: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    sam_clear = relabel_instance_mask(sam_clear)
    compact_candidate = relabel_instance_mask(compact_candidate)
    final = np.zeros(sam_clear.shape, dtype=np.uint16)
    class_mask = np.zeros(sam_clear.shape, dtype=np.uint8)
    rows = []
    next_label = 1

    for old in sorted(int(v) for v in np.unique(sam_clear) if int(v) != 0):
        pixels = sam_clear == old
        final[pixels] = next_label
        class_mask[pixels] = 1
        rows.append(
            {
                "final_label": next_label,
                "final_class_id": 1,
                "final_class_name": "clear_cell_boundary",
                "source_model": "SAM3.1",
                "source_label": old,
                "area_px": int(pixels.sum()),
            }
        )
        next_label += 1

    for old in sorted(int(v) for v in np.unique(compact_candidate) if int(v) != 0):
        pixels = (compact_candidate == old) & (final == 0)
        if int(pixels.sum()) < CELLSEG1_MIN_AREA:
            continue
        final[pixels] = next_label
        class_mask[pixels] = 2
        rows.append(
            {
                "final_label": next_label,
                "final_class_id": 2,
                "final_class_name": "compact_cell_boundary",
                "source_model": "CellSeg1",
                "source_label": old,
                "area_px": int(pixels.sum()),
            }
        )
        next_label += 1
    return final, class_mask, rows


def load_sam3_processor():
    if str(SAM3_REPO) not in sys.path:
        sys.path.insert(0, str(SAM3_REPO))
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(
        checkpoint_path=None,
        load_from_HF=False,
        eval_mode=True,
        enable_segmentation=True,
    )
    checkpoint = torch.load(SAM31_CHECKPOINT, map_location="cpu")
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    print("SAM3 missing:", len(missing))
    print("SAM3 unexpected:", len(unexpected))
    if missing or unexpected:
        raise RuntimeError(f"SAM3 checkpoint mismatch. missing={missing[:10]} unexpected={unexpected[:10]}")
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return Sam3Processor(model)


def load_cellseg1():
    if str(CELLSEG1_REPO) not in sys.path:
        sys.path.insert(0, str(CELLSEG1_REPO))
    from data.utils import read_image_to_numpy, resize_image
    from predict import predict_images

    with CELLSEG1_CONFIG_PATH.open("r") as handle:
        cellseg_config = yaml.safe_load(handle)
    cellseg_config["result_pth_path"] = str(CELLSEG1_LORA)
    cellseg_config["pred_iou_thresh"] = CELLSEG1_IOU_THRESH
    cellseg_config["stability_score_thresh"] = CELLSEG1_STABILITY_THRESH
    cellseg_config["deterministic"] = False
    return cellseg_config, read_image_to_numpy, resize_image, predict_images


def save_csv(path: Path, rows: List[dict]) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)
    else:
        with path.open("w", newline="") as handle:
            csv.writer(handle).writerow(["empty"])


def main() -> None:
    for path, label in [
        (COCO_ROOT, "COCO root"),
        (SAM3_REPO, "SAM3 repo"),
        (SAM31_CHECKPOINT, "SAM3 checkpoint"),
        (CELLSEG1_REPO, "CellSeg1 repo"),
        (CELLSEG1_CONFIG_PATH, "CellSeg1 config"),
        (CELLSEG1_LORA, "CellSeg1 LoRA checkpoint"),
        (YOLO_MODEL_PATH, "YOLO model"),
    ]:
        assert_path(path, label)

    PRED_MASK_DIR.mkdir(parents=True, exist_ok=True)
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    cat_by_split_and_id, anns_by_split_and_image, all_images = load_coco()
    trial_images = select_images(all_images)

    print("Trial Run 3 output:", OUT_DIR)
    print("Selected images:", len(trial_images), [f"{x['_split']}_{Path(x['file_name']).stem}" for x in trial_images])
    print("SAM3 prompts:", SAM31_CLEAR_PROMPTS)
    print("SAM3 score threshold:", SAM31_SCORE_THRESH)

    yolo_model = YOLO(str(YOLO_MODEL_PATH))
    print("YOLO names:", yolo_model.names)
    processor = load_sam3_processor()
    cellseg_config, read_image_to_numpy, resize_image, predict_images = load_cellseg1()
    print("CellSeg1 LoRA:", cellseg_config["result_pth_path"])

    metrics_rows: List[dict] = []
    source_instance_rows: List[dict] = []
    final_instance_rows: List[dict] = []
    saved_images: List[Path] = []

    for image_info in trial_images:
        split = image_info["_split"]
        tile_id = Path(image_info["file_name"]).stem
        tile_key = f"{split}_{tile_id}"
        image_path = COCO_ROOT / split / image_info["file_name"]
        print("\n" + "=" * 100)
        print("Trial 3 running:", tile_key)
        started = time.time()

        pil_image = Image.open(image_path).convert("RGB")
        original_rgb = np.asarray(pil_image)
        gt_clear = gt_mask_for_categories(
            image_info, {"clear_cell_boundary"}, cat_by_split_and_id, anns_by_split_and_image
        )
        gt_compact = gt_mask_for_categories(
            image_info, {"compact_cell_boundary"}, cat_by_split_and_id, anns_by_split_and_image
        )
        gt_all = gt_mask_for_categories(
            image_info,
            {"clear_cell_boundary", "compact_cell_boundary"},
            cat_by_split_and_id,
            anns_by_split_and_image,
        )

        nucleus_mask, nucleus_rows = yolo_nucleus_mask(yolo_model, image_path, original_rgb, tile_key=tile_key)
        sam_clear, sam_rows = sam3_clear_mask(processor, pil_image, nucleus_mask, tile_key=tile_key)
        _, cellseg_raw = cellseg1_predict_one(
            image_path, cellseg_config, read_image_to_numpy, resize_image, predict_images
        )
        cellseg_raw = resize_mask_to_image(cellseg_raw, original_rgb)
        compact_candidate, compact_rows = filter_cellseg_compact_candidates(
            cellseg_raw, sam_clear, nucleus_mask, original_rgb
        )
        hybrid_mask, hybrid_class_mask, final_rows = combine_clear_and_compact(sam_clear, compact_candidate)
        elapsed = time.time() - started

        paths = {
            "hybrid": PRED_MASK_DIR / f"{tile_key}_hybrid_instance_mask.png",
            "class": PRED_MASK_DIR / f"{tile_key}_hybrid_class_mask.png",
            "nucleus": PRED_MASK_DIR / f"{tile_key}_yolo_nucleus_mask.png",
            "sam_clear": PRED_MASK_DIR / f"{tile_key}_sam31_clear_mask.png",
            "compact": PRED_MASK_DIR / f"{tile_key}_cellseg1_residual_compact_mask.png",
        }
        Image.fromarray(hybrid_mask.astype(np.uint16)).save(paths["hybrid"])
        Image.fromarray(hybrid_class_mask.astype(np.uint8)).save(paths["class"])
        Image.fromarray(nucleus_mask.astype(np.uint16)).save(paths["nucleus"])
        Image.fromarray(sam_clear.astype(np.uint16)).save(paths["sam_clear"])
        Image.fromarray(compact_candidate.astype(np.uint16)).save(paths["compact"])

        for model_name, pred_mask, gt_mask in [
            ("SAM3_clear_vs_GT_clear", sam_clear, gt_clear),
            ("CellSeg1_residual_compact_vs_GT_compact", compact_candidate, gt_compact),
            ("Hybrid_vs_GT_all_boundaries", hybrid_mask, gt_all),
        ]:
            row = binary_metrics(gt_mask, pred_mask)
            row.update(
                {
                    "trial": TRIAL3_NAME,
                    "model": model_name,
                    "split": split,
                    "tile_id": tile_id,
                    "file_name": image_info["file_name"],
                    "inference_time_sec": elapsed,
                    "yolo_nucleus_instances": int(nucleus_mask.max()),
                    "sam31_clear_instances": int(sam_clear.max()),
                    "cellseg1_residual_compact_instances": int(compact_candidate.max()),
                    "hybrid_instances": int(hybrid_mask.max()),
                    "sam31_score_thresh": SAM31_SCORE_THRESH,
                    "sam31_min_area": SAM31_MIN_AREA,
                    "cellseg1_min_area": CELLSEG1_MIN_AREA,
                    "cellseg1_max_area": CELLSEG1_MAX_AREA,
                    "max_overlap_with_sam_clear": MAX_OVERLAP_WITH_SAM_CLEAR,
                    "yolo_conf": YOLO_CONF,
                    "yolo_iou": YOLO_IOU,
                }
            )
            metrics_rows.append(row)

        for row in nucleus_rows + sam_rows + compact_rows:
            row.update({"trial": TRIAL3_NAME, "split": split, "tile_id": tile_id, "file_name": image_info["file_name"]})
            source_instance_rows.append(row)
        for row in final_rows:
            row.update({"trial": TRIAL3_NAME, "split": split, "tile_id": tile_id, "file_name": image_info["file_name"]})
            final_instance_rows.append(row)

        panels = [
            original_rgb,
            draw_instance_mask(original_rgb, gt_all, GT_COLOR),
            draw_instance_mask(original_rgb, nucleus_mask, NUCLEUS_COLOR),
            draw_instance_mask(original_rgb, sam_clear, SAM_CLEAR_COLOR),
            draw_instance_mask(original_rgb, compact_candidate, CELLSEG_COMPACT_COLOR),
            draw_instance_mask(original_rgb, hybrid_mask, HYBRID_COLOR),
        ]
        titles = [
            f"{tile_key}: original",
            f"GT all n={int(gt_all.max())}",
            f"YOLO nucleus n={int(nucleus_mask.max())}",
            f"SAM3 clear n={int(sam_clear.max())}",
            f"CellSeg1 residual compact n={int(compact_candidate.max())}",
            f"Hybrid n={int(hybrid_mask.max())}",
        ]
        fig, axes = plt.subplots(1, 6, figsize=(30, 5))
        for ax, panel, title in zip(axes, panels, titles):
            ax.imshow(panel)
            ax.set_title(title)
            ax.axis("off")
        fig.tight_layout()
        out_img = COMPARE_DIR / f"{tile_key}_trial3_compare.png"
        fig.savefig(out_img, dpi=180)
        plt.close(fig)
        saved_images.append(out_img)
        if DISPLAY_IMAGES:
            display(IPyImage(filename=str(out_img)))

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv = OUT_DIR / "trial_run_3_metrics.csv"
    source_csv = OUT_DIR / "trial_run_3_source_instances.csv"
    final_csv = OUT_DIR / "trial_run_3_final_instances.csv"
    summary_csv = OUT_DIR / "trial_run_3_summary.csv"

    metrics_df.to_csv(metrics_csv, index=False)
    save_csv(source_csv, source_instance_rows)
    save_csv(final_csv, final_instance_rows)

    summary_df = metrics_df.groupby("model").agg(
        n_tiles=("tile_id", "count"),
        dice_mean=("dice", "mean"),
        dice_std=("dice", "std"),
        iou_mean=("iou", "mean"),
        precision_mean=("precision", "mean"),
        recall_mean=("recall_sensitivity", "mean"),
        pred_instances_mean=("pred_instances", "mean"),
        gt_instances_mean=("gt_instances", "mean"),
        inference_time_sec_mean=("inference_time_sec", "mean"),
    ).reset_index()
    summary_df.to_csv(summary_csv, index=False)

    print("\nTrial run 3 complete.")
    print("Output:", OUT_DIR)
    print("Metrics:", metrics_csv)
    print("Summary:", summary_csv)
    print("Source instances:", source_csv)
    print("Final instances:", final_csv)
    print("Comparison images:", COMPARE_DIR)
    print(summary_df)


if __name__ == "__main__":
    main()
