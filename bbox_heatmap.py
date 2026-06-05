#!/usr/bin/env python3
"""Create a dataset-level heatmap from YOLO bounding boxes and export it as JPG."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageFilter


DEFAULT_LABELS_ROOT = Path("Splitted dataset") / "labels"
DEFAULT_IMAGES_ROOT = Path("Splitted dataset") / "images"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a bbox heatmap for the dataset.")
    parser.add_argument(
        "--labels-root",
        type=Path,
        default=DEFAULT_LABELS_ROOT,
        help="Root directory containing train/valid/test label folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset_bbox_heatmap.jpg"),
        help="Output JPG path.",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=DEFAULT_IMAGES_ROOT,
        help="Root directory containing train/valid/test image folders used to pick a sample background.",
    )
    parser.add_argument(
        "--background-image",
        type=Path,
        default=None,
        help="Optional explicit image to use as the background. If omitted, the first dataset image is used.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1024,
        help="Heatmap canvas size in pixels (square).",
    )
    parser.add_argument(
        "--blur-radius",
        type=float,
        default=18.0,
        help="Gaussian blur radius for smoothing the density map.",
    )
    return parser.parse_args()


def iter_label_files(labels_root: Path) -> Iterable[Path]:
    for split in ("train", "valid", "test"):
        split_dir = labels_root / split
        if not split_dir.exists():
            continue
        yield from sorted(split_dir.glob("*.txt"))


def iter_image_files(images_root: Path) -> Iterable[Path]:
    for split in ("train", "valid", "test"):
        split_dir = images_root / split
        if not split_dir.exists():
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
            yield from sorted(split_dir.glob(ext))


def read_yolo_bboxes(label_file: Path) -> List[Tuple[int, float, float, float, float]]:
    bboxes: List[Tuple[int, float, float, float, float]] = []
    text = label_file.read_text(encoding="utf-8").strip()
    if not text:
        return bboxes

    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            continue
        bboxes.append((class_id, x_center, y_center, width, height))
    return bboxes


def accumulate_heatmap(label_files: Iterable[Path], size: int) -> Tuple[np.ndarray, int, int]:
    heatmap = np.zeros((size, size), dtype=np.float32)
    label_count = 0
    bbox_count = 0

    for label_file in label_files:
        label_count += 1
        for _, x_center, y_center, width, height in read_yolo_bboxes(label_file):
            bbox_count += 1

            x1 = max(0, int((x_center - width / 2.0) * size))
            y1 = max(0, int((y_center - height / 2.0) * size))
            x2 = min(size, int(np.ceil((x_center + width / 2.0) * size)))
            y2 = min(size, int(np.ceil((y_center + height / 2.0) * size)))

            if x2 <= x1 or y2 <= y1:
                continue

            heatmap[y1:y2, x1:x2] += 1.0

    return heatmap, label_count, bbox_count


def normalize_to_uint8(values: np.ndarray) -> np.ndarray:
    maximum = float(values.max())
    if maximum <= 0.0:
        return np.zeros_like(values, dtype=np.uint8)
    normalized = values / maximum
    normalized = np.clip(normalized * 255.0, 0, 255)
    return normalized.astype(np.uint8)


def select_background_image(image_files: List[Path], background_image: Path | None) -> Tuple[Path, int]:
    if background_image is not None:
        if not background_image.exists():
            raise FileNotFoundError(f"Background image not found: {background_image}")
        return background_image, len(image_files)

    if not image_files:
        raise FileNotFoundError("No readable dataset images found to use as background.")

    return image_files[0], len(image_files)


def apply_colormap(gray: np.ndarray) -> Image.Image:
    """Map grayscale density to an RGB heatmap without extra dependencies."""
    normalized = gray.astype(np.float32) / 255.0

    # Blue -> Cyan -> Yellow -> Red
    r = np.zeros_like(normalized)
    g = np.zeros_like(normalized)
    b = np.zeros_like(normalized)

    first = normalized <= 0.33
    second = (normalized > 0.33) & (normalized <= 0.66)
    third = normalized > 0.66

    # 0.00-0.33: dark blue to cyan
    t1 = np.clip(normalized[first] / 0.33, 0.0, 1.0)
    r[first] = 0.0
    g[first] = 255.0 * t1
    b[first] = 128.0 + 127.0 * t1

    # 0.33-0.66: cyan to yellow
    t2 = np.clip((normalized[second] - 0.33) / 0.33, 0.0, 1.0)
    r[second] = 255.0 * t2
    g[second] = 255.0
    b[second] = 255.0 * (1.0 - t2)

    # 0.66-1.00: yellow to red
    t3 = np.clip((normalized[third] - 0.66) / 0.34, 0.0, 1.0)
    r[third] = 255.0
    g[third] = 255.0 * (1.0 - t3)
    b[third] = 0.0

    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def prepare_background(image_path: Path, size: int) -> Image.Image:
    with Image.open(image_path) as img:
        background = img.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)

    background = background.filter(ImageFilter.GaussianBlur(radius=0.8))
    gray = background.convert("L")
    gray = Image.eval(gray, lambda p: int(max(0, min(255, p * 0.88))))
    return gray.convert("RGB")


def main() -> None:
    args = parse_args()

    label_files = list(iter_label_files(args.labels_root))
    if not label_files:
        raise FileNotFoundError(f"No label files found under: {args.labels_root}")

    image_files = list(iter_image_files(args.images_root))
    if not image_files:
        raise FileNotFoundError(f"No image files found under: {args.images_root}")

    background_path, image_count = select_background_image(image_files, args.background_image)

    heatmap, label_count, bbox_count = accumulate_heatmap(label_files, args.size)
    heatmap_image = Image.fromarray(normalize_to_uint8(heatmap), mode="L")
    if args.blur_radius > 0:
        heatmap_image = heatmap_image.filter(ImageFilter.GaussianBlur(radius=args.blur_radius))

    colored_heatmap = apply_colormap(np.array(heatmap_image))
    background = prepare_background(background_path, args.size)

    # Keep the X-ray visible while letting the heatmap dominate the hotspots.
    final_image = Image.blend(background, colored_heatmap, alpha=0.50)
    final_image = Image.blend(final_image, background, alpha=0.08)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    final_image.save(args.output, quality=95)

    print(f"Scanned {label_count} label files")
    print(f"Used sample background image: {background_path}")
    print(f"Dataset images available: {image_count}")
    print(f"Accumulated {bbox_count} bounding boxes")
    print(f"Saved heatmap to: {args.output}")


if __name__ == "__main__":
    main()