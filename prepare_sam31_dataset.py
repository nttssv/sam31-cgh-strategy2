#!/usr/bin/env python3
"""Prepare CGH pathology masks for SAM 3.1 fine-tuning.

The source export contains 512x512 tile images, per-cell instance masks,
per-nucleus instance masks, and auxiliary tissue masks. This script creates:

- dataset/sam31_manifest.csv
- dataset/coco_sam3/cgh_pathology_sam31/train/_annotations.coco.json
- dataset/coco_sam3/cgh_pathology_sam31/test/_annotations.coco.json

The generated COCO JSON uses 1-based category ids for compatibility while
preserving the zero-based class ids used by the YOLO project in metadata.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage


PACKAGE_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PACKAGE_ROOT / "dataset"
IMAGE_DIR = DATASET_ROOT / "images"
CELL_MASK_DIR = DATASET_ROOT / "cell_instance_masks"
AUX_MASK_DIR = DATASET_ROOT / "auxiliary_masks"
METADATA_DIR = DATASET_ROOT / "metadata"
COCO_ROOT = DATASET_ROOT / "coco_sam3" / "cgh_pathology_sam31"
DEFAULT_FULL41_SOURCE_ROOT = (
    Path.home() / "Desktop/1.Data/training_pa_he_annotation_full/cellseg1_cgh_p2_combined_41_full"
)

INCLUDE_UNCERTAIN_IN_COCO = os.getenv("INCLUDE_UNCERTAIN_IN_COCO", "0") == "1"
MIN_COMPONENT_AREA_PX = int(os.getenv("MIN_COMPONENT_AREA_PX", "10"))
DEFAULT_VAL_TILE_IDS = "p2_tile_05,p2_tile_10,p2_tile_15,p2_tile_20"
VAL_TILE_IDS = {
    token.strip()
    for token in os.getenv("SAM31_VAL_TILE_IDS", DEFAULT_VAL_TILE_IDS).split(",")
    if token.strip()
}

CATEGORIES = [
    {
        "id": 1,
        "name": "nucleus",
        "class_id_zero_based": 0,
        "prompt": "cell nucleus in H and E pathology",
    },
    {
        "id": 2,
        "name": "clear_cell_boundary",
        "class_id_zero_based": 1,
        "prompt": "adrenal cortical clear cell boundary",
    },
    {
        "id": 3,
        "name": "compact_cell_boundary",
        "class_id_zero_based": 2,
        "prompt": "adrenal cortical compact cell boundary",
    },
    {
        "id": 4,
        "name": "connective_tissue_candidate",
        "class_id_zero_based": 3,
        "prompt": "connective tissue or stromal candidate region",
    },
    {
        "id": 5,
        "name": "uncertain_cell_boundary",
        "class_id_zero_based": 4,
        "prompt": "uncertain adrenal cortical cell boundary",
    },
]

CAT_BY_NAME = {row["name"]: row for row in CATEGORIES}


def read_png_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path))


def image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def is_real_png(path: Path) -> bool:
    return path.suffix.lower() == ".png" and not path.name.startswith("._")


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_files(src: Path, dst: Path, suffixes: Tuple[str, ...] = (".png",)) -> int:
    if not src.exists():
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in sorted(src.iterdir()):
        if path.name.startswith("._") or path.name == ".DS_Store":
            continue
        if path.is_file() and path.suffix.lower() in suffixes:
            shutil.copy2(path, dst / path.name)
            copied += 1
    return copied


def resolve_source_root() -> Optional[Path]:
    for env_name in ("CGH_SAM31_SOURCE_ROOT", "CGH_DATASET_ROOT"):
        raw = os.getenv(env_name, "").strip()
        if raw:
            path = Path(raw).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"{env_name} points to a missing path: {path}")
            return path
    if DEFAULT_FULL41_SOURCE_ROOT.exists():
        return DEFAULT_FULL41_SOURCE_ROOT.resolve()
    return None


def resolve_source_image_mask_dirs(source_root: Path) -> Tuple[Path, Path]:
    candidates = [
        (source_root / "train" / "images", source_root / "train" / "masks"),
        (source_root / "images", source_root / "masks"),
    ]
    for image_dir, mask_dir in candidates:
        if image_dir.exists() and mask_dir.exists():
            return image_dir, mask_dir
    raise FileNotFoundError(
        "Could not locate image/mask folders under source root. Expected "
        f"train/images + train/masks or images + masks under: {source_root}"
    )


def stage_dataset_from_source(source_root: Path) -> Dict[str, object]:
    """Copy the full CellSeg1 export into the legacy Strategy2 dataset layout."""
    image_src, mask_src = resolve_source_image_mask_dirs(source_root)
    image_files = sorted(p for p in image_src.glob("*.png") if is_real_png(p))
    mask_files = sorted(p for p in mask_src.glob("*.png") if is_real_png(p))
    if not image_files or not mask_files:
        raise FileNotFoundError(f"No PNG image/mask pairs found in {image_src} and {mask_src}")
    if len(image_files) != len(mask_files):
        raise RuntimeError(f"Image/mask count mismatch: {len(image_files)} images, {len(mask_files)} masks")
    if [p.stem for p in image_files] != [p.stem for p in mask_files]:
        raise RuntimeError("Image/mask stems do not match in the source dataset")

    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    for path in [
        IMAGE_DIR,
        CELL_MASK_DIR,
        AUX_MASK_DIR,
        METADATA_DIR,
        DATASET_ROOT / "semantic_masks",
        DATASET_ROOT / "previews",
        COCO_ROOT,
    ]:
        clean_dir(path)

    copied_images = copy_files(image_src, IMAGE_DIR, suffixes=(".png",))
    copied_masks = copy_files(mask_src, CELL_MASK_DIR, suffixes=(".png",))
    copied_aux = copy_files(source_root / "auxiliary_masks", AUX_MASK_DIR, suffixes=(".png",))
    copied_semantic = copy_files(source_root / "semantic_masks", DATASET_ROOT / "semantic_masks", suffixes=(".png",))
    copied_previews = copy_files(source_root / "previews", DATASET_ROOT / "previews", suffixes=(".png", ".jpg", ".jpeg"))

    metadata_names = [
        "dataset_manifest.csv",
        "cell_instances.csv",
        "boundary_qc.csv",
        "batch_membership.csv",
        "conversion_summary.json",
        "README.md",
    ]
    copied_metadata = 0
    for name in metadata_names:
        src = source_root / name
        if src.exists() and src.is_file():
            shutil.copy2(src, METADATA_DIR / src.name)
            copied_metadata += 1

    summary = {
        "source_root": str(source_root),
        "source_image_dir": str(image_src),
        "source_mask_dir": str(mask_src),
        "copied_images": copied_images,
        "copied_cell_masks": copied_masks,
        "copied_auxiliary_masks": copied_aux,
        "copied_semantic_masks": copied_semantic,
        "copied_previews": copied_previews,
        "copied_metadata_files": copied_metadata,
        "val_tile_ids": sorted(VAL_TILE_IDS),
    }
    (DATASET_ROOT / "staged_source_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_yolo_split() -> Dict[str, str]:
    """Return tile id -> split using the existing YOLO split if available."""
    split_by_tile: Dict[str, str] = {}
    yolo_images = DATASET_ROOT / "yolo_seg_dataset" / "images"
    for split_name in ("train", "val"):
        split_dir = yolo_images / split_name
        if not split_dir.exists():
            continue
        for path in sorted(p for p in split_dir.glob("*.png") if is_real_png(p)):
            split_by_tile[path.stem] = "test" if split_name == "val" else "train"
    return split_by_tile


def default_split(tile_id: str) -> str:
    return "test" if tile_id in VAL_TILE_IDS else "train"


def load_cell_class_map() -> Dict[Tuple[str, int], str]:
    """Map (tile_id, instance_label) to clear/compact category name."""
    csv_path = METADATA_DIR / "cell_instances.csv"
    mapping: Dict[Tuple[str, int], str] = {}
    if not csv_path.exists():
        return mapping

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tile_id = row.get("tile_id", "").strip().strip('"')
            label_raw = row.get("instance_label", "").strip().strip('"')
            klass = row.get("boundary_class", "").lower()
            if not tile_id or not label_raw:
                continue
            try:
                label = int(float(label_raw))
            except ValueError:
                continue
            if "compact" in klass:
                mapping[(tile_id, label)] = "compact_cell_boundary"
            elif "clear" in klass:
                mapping[(tile_id, label)] = "clear_cell_boundary"
    return mapping


def connected_components(mask: np.ndarray) -> Iterable[Tuple[int, np.ndarray]]:
    labeled, n = ndimage.label(mask.astype(bool))
    for label in range(1, n + 1):
        component = labeled == label
        if int(component.sum()) >= MIN_COMPONENT_AREA_PX:
            yield label, component


def instance_components(mask: np.ndarray) -> Iterable[Tuple[int, np.ndarray]]:
    labels = np.unique(mask)
    labels = labels[labels != 0]
    for label in labels:
        component = mask == label
        if int(component.sum()) >= MIN_COMPONENT_AREA_PX:
            yield int(label), component


def mask_to_bbox(mask: np.ndarray) -> Optional[List[float]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)]


def mask_to_uncompressed_rle(mask: np.ndarray) -> Dict[str, object]:
    """COCO uncompressed RLE, in Fortran/column-major order."""
    pixels = np.asfortranarray(mask.astype(np.uint8)).reshape(-1, order="F")
    counts: List[int] = []
    last_val = 0
    run_len = 0
    for val in pixels:
        val = int(val)
        if val == last_val:
            run_len += 1
        else:
            counts.append(run_len)
            run_len = 1
            last_val = val
    counts.append(run_len)
    h, w = mask.shape[:2]
    return {"size": [int(h), int(w)], "counts": counts}


def add_annotation(
    annotations: List[Dict[str, object]],
    image_id: int,
    category_name: str,
    mask: np.ndarray,
    source_mask: str,
    source_instance_label: int,
    ann_id: int,
) -> Optional[Dict[str, object]]:
    bbox = mask_to_bbox(mask)
    if bbox is None:
        return None
    category = CAT_BY_NAME[category_name]
    ann = {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category["id"],
        "bbox": bbox,
        "area": int(mask.sum()),
        "segmentation": mask_to_uncompressed_rle(mask),
        "iscrowd": 0,
        "source_mask": source_mask,
        "source_instance_label": int(source_instance_label),
        "class_id_zero_based": category["class_id_zero_based"],
    }
    annotations.append(ann)
    return ann


def copy_image_to_split(image_path: Path, split: str) -> str:
    split_dir = COCO_ROOT / split
    split_dir.mkdir(parents=True, exist_ok=True)
    dst = split_dir / image_path.name
    shutil.copy2(image_path, dst)
    return image_path.name


def manifest_row(
    tile_id: str,
    split: str,
    image_file: str,
    category_name: str,
    mask_file: str,
    instance_label: int,
    area_px: int,
    include_for_training: bool,
) -> Dict[str, object]:
    category = CAT_BY_NAME[category_name]
    return {
        "tile_id": tile_id,
        "split": split,
        "image_file": image_file,
        "category": category_name,
        "coco_category_id": category["id"],
        "class_id_zero_based": category["class_id_zero_based"],
        "mask_file": mask_file,
        "source_instance_label": instance_label,
        "area_px": area_px,
        "include_for_training": str(include_for_training).lower(),
    }


def build() -> None:
    source_root = resolve_source_root()
    staged_summary: Dict[str, object] = {}
    if source_root is not None:
        staged_summary = stage_dataset_from_source(source_root)
    elif not IMAGE_DIR.exists() or not CELL_MASK_DIR.exists():
        raise FileNotFoundError(
            "No staged dataset found. Set CGH_SAM31_SOURCE_ROOT or CGH_DATASET_ROOT "
            "to the full CellSeg1 dataset folder, for example "
            "~/Desktop/1.Data/training_pa_he_annotation_full/cellseg1_cgh_p2_combined_41_full."
        )
    else:
        clean_dir(COCO_ROOT)

    split_by_tile = load_yolo_split()
    cell_class_map = load_cell_class_map()

    images_by_split: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    anns_by_split: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    manifest_rows: List[Dict[str, object]] = []
    next_image_id = 1
    next_ann_id = 1

    for image_path in sorted(p for p in IMAGE_DIR.glob("*.png") if is_real_png(p)):
        tile_id = image_path.stem
        split = split_by_tile.get(tile_id, default_split(tile_id))
        width, height = image_size(image_path)
        file_name = copy_image_to_split(image_path, split)
        image_id = next_image_id
        next_image_id += 1
        images_by_split[split].append(
            {"id": image_id, "file_name": file_name, "width": width, "height": height}
        )

        cell_mask_path = CELL_MASK_DIR / image_path.name
        if cell_mask_path.exists():
            cell_mask = read_png_mask(cell_mask_path)
            for label, mask in instance_components(cell_mask):
                category_name = cell_class_map.get((tile_id, label), "clear_cell_boundary")
                ann = add_annotation(
                    anns_by_split[split],
                    image_id,
                    category_name,
                    mask,
                    str(cell_mask_path.relative_to(PACKAGE_ROOT)),
                    label,
                    next_ann_id,
                )
                if ann is not None:
                    next_ann_id += 1
                    manifest_rows.append(
                        manifest_row(
                            tile_id,
                            split,
                            str((COCO_ROOT / split / file_name).relative_to(PACKAGE_ROOT)),
                            category_name,
                            str(cell_mask_path.relative_to(PACKAGE_ROOT)),
                            label,
                            int(mask.sum()),
                            True,
                        )
                    )

        nucleus_mask_path = AUX_MASK_DIR / f"{tile_id}_gt_nucleus_instances.png"
        if nucleus_mask_path.exists():
            nucleus_mask = read_png_mask(nucleus_mask_path)
            for label, mask in instance_components(nucleus_mask):
                ann = add_annotation(
                    anns_by_split[split],
                    image_id,
                    "nucleus",
                    mask,
                    str(nucleus_mask_path.relative_to(PACKAGE_ROOT)),
                    label,
                    next_ann_id,
                )
                if ann is not None:
                    next_ann_id += 1
                    manifest_rows.append(
                        manifest_row(
                            tile_id,
                            split,
                            str((COCO_ROOT / split / file_name).relative_to(PACKAGE_ROOT)),
                            "nucleus",
                            str(nucleus_mask_path.relative_to(PACKAGE_ROOT)),
                            label,
                            int(mask.sum()),
                            True,
                        )
                    )

        stroma_mask_path = AUX_MASK_DIR / f"{tile_id}_gt_stroma.png"
        if stroma_mask_path.exists():
            stroma_mask = read_png_mask(stroma_mask_path) > 0
            for label, mask in connected_components(stroma_mask):
                ann = add_annotation(
                    anns_by_split[split],
                    image_id,
                    "connective_tissue_candidate",
                    mask,
                    str(stroma_mask_path.relative_to(PACKAGE_ROOT)),
                    label,
                    next_ann_id,
                )
                if ann is not None:
                    next_ann_id += 1
                    manifest_rows.append(
                        manifest_row(
                            tile_id,
                            split,
                            str((COCO_ROOT / split / file_name).relative_to(PACKAGE_ROOT)),
                            "connective_tissue_candidate",
                            str(stroma_mask_path.relative_to(PACKAGE_ROOT)),
                            label,
                            int(mask.sum()),
                            True,
                        )
                    )

        uncertain_mask_path = AUX_MASK_DIR / f"{tile_id}_gt_uncertain_ignore.png"
        if uncertain_mask_path.exists():
            uncertain_mask = read_png_mask(uncertain_mask_path) > 0
            for label, mask in connected_components(uncertain_mask):
                include = INCLUDE_UNCERTAIN_IN_COCO
                if include:
                    ann = add_annotation(
                        anns_by_split[split],
                        image_id,
                        "uncertain_cell_boundary",
                        mask,
                        str(uncertain_mask_path.relative_to(PACKAGE_ROOT)),
                        label,
                        next_ann_id,
                    )
                    if ann is not None:
                        next_ann_id += 1
                manifest_rows.append(
                    manifest_row(
                        tile_id,
                        split,
                        str((COCO_ROOT / split / file_name).relative_to(PACKAGE_ROOT)),
                        "uncertain_cell_boundary",
                        str(uncertain_mask_path.relative_to(PACKAGE_ROOT)),
                        label,
                        int(mask.sum()),
                        include,
                    )
                )

    categories_for_coco = CATEGORIES if INCLUDE_UNCERTAIN_IN_COCO else CATEGORIES[:4]
    for split in ("train", "test"):
        split_dir = COCO_ROOT / split
        split_dir.mkdir(parents=True, exist_ok=True)
        coco = {
            "images": images_by_split.get(split, []),
            "annotations": anns_by_split.get(split, []),
            "categories": categories_for_coco,
            "info": {
                "description": "CGH adrenal pathology SAM 3.1 training dataset",
                "source": "QuPath annotations exported from cellseg1_cgh_p2",
            },
        }
        with (split_dir / "_annotations.coco.json").open("w") as f:
            json.dump(coco, f)

    manifest_path = DATASET_ROOT / "sam31_manifest.csv"
    fieldnames = [
        "tile_id",
        "split",
        "image_file",
        "category",
        "coco_category_id",
        "class_id_zero_based",
        "mask_file",
        "source_instance_label",
        "area_px",
        "include_for_training",
    ]
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "images": sum(len(v) for v in images_by_split.values()),
        "train_images": len(images_by_split.get("train", [])),
        "test_images": len(images_by_split.get("test", [])),
        "annotations": sum(len(v) for v in anns_by_split.values()),
        "manifest_rows": len(manifest_rows),
        "categories": [c["name"] for c in categories_for_coco],
        "include_uncertain_in_coco": INCLUDE_UNCERTAIN_IN_COCO,
        "val_tile_ids": sorted(VAL_TILE_IDS),
        "staged_source": staged_summary,
    }
    with (DATASET_ROOT / "sam31_dataset_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    build()
