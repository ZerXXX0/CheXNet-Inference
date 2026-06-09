#!/usr/bin/env python3
"""Convert YOLO object detection dataset to a classification dataset format.

Supports three strategies:
- 'all_classes' (default): Copies each image to the folder of every class it contains.
- 'first_class': Copies each image to the folder of the first class listed in its labels.
- 'dominant_class': Copies each image to the folder of the class with the largest bounding box area.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert YOLO dataset to classification format.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("Splitted dataset"),
        help="Path to the original YOLO dataset directory.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["all_classes", "first_class", "dominant_class"],
        default="all_classes",
        help="Strategy for handling images with multiple labels.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        default=True,
        help="Keep a backup of the original dataset directory (default: True).",
    )
    parser.add_argument(
        "--no-backup",
        action="store_false",
        dest="backup",
        help="Do not keep a backup of the original dataset directory.",
    )
    return parser.parse_args()


def read_class_names(data_yaml: Path) -> dict[int, str]:
    """Read class names from data.yaml without external dependencies."""
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    text = data_yaml.read_text(encoding="utf-8")

    # Simple manual parser for data.yaml
    names: dict[int, str] = {}
    in_names = False
    for line in text.splitlines():
        raw = line.rstrip()
        stripped = raw.strip()

        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("names:"):
            in_names = True
            continue

        if in_names:
            if raw.startswith(" ") or raw.startswith("\t"):
                if ":" in stripped:
                    k, v = stripped.split(":", 1)
                    try:
                        idx = int(k.strip())
                        names[idx] = v.strip().strip('"\'')
                    except ValueError:
                        continue
            else:
                break

    return names


def parse_yolo_labels(label_file: Path) -> list[tuple[int, float]]:
    """Parse label file and return list of (class_id, box_area)."""
    if not label_file.exists():
        return []

    labels: list[tuple[int, float]] = []
    text = label_file.read_text(encoding="utf-8").strip()
    if not text:
        return labels

    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            try:
                class_id = int(parts[0])
                width = float(parts[3])
                height = float(parts[4])
                area = width * height
                labels.append((class_id, area))
            except ValueError:
                continue
    return labels


def determine_classes(labels: list[tuple[int, float]], strategy: str) -> list[int]:
    """Determine class IDs for an image based on the selected strategy."""
    if not labels:
        return []

    if strategy == "all_classes":
        # Return unique class IDs present
        return list(set(cid for cid, _ in labels))

    if strategy == "first_class":
        # Return the class of the first bounding box
        return [labels[0][0]]

    if strategy == "dominant_class":
        # Sum areas per class and find the class with the largest area
        areas: dict[int, float] = {}
        for cid, area in labels:
            areas[cid] = areas.get(cid, 0.0) + area
        dominant_class = max(areas.items(), key=lambda x: x[1])[0]
        return [dominant_class]

    return []


def main() -> None:
    args = parse_args()

    if not args.dataset_dir.exists():
        print(f"Error: Dataset directory {args.dataset_dir} does not exist.", file=sys.stderr)
        sys.exit(1)

    data_yaml = args.dataset_dir / "data.yaml"
    try:
        class_mapping = read_class_names(data_yaml)
    except Exception as e:
        print(f"Error reading class names: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Class mapping: {class_mapping}")
    print(f"Strategy: {args.strategy}")

    # Prepare temp output directory
    temp_dir = args.dataset_dir.parent / f"{args.dataset_dir.name}_new"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    # Copy data.yaml to the new dataset folder
    if data_yaml.exists():
        shutil.copy2(data_yaml, temp_dir / "data.yaml")

    splits = ["train", "valid", "test"]
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    copied_counts = Counter()

    for split in splits:
        img_src_dir = args.dataset_dir / "images" / split
        lbl_src_dir = args.dataset_dir / "labels" / split

        if not img_src_dir.exists():
            continue

        # Get all images
        images = [
            p for p in img_src_dir.iterdir()
            if p.is_file() and p.suffix.lower() in image_extensions
        ]

        for img_path in images:
            lbl_path = lbl_src_dir / f"{img_path.stem}.txt"
            labels = parse_yolo_labels(lbl_path)

            target_class_ids = determine_classes(labels, args.strategy)

            if not target_class_ids:
                # No annotations -> 'normal' folder
                target_dirs = [temp_dir / split / "normal"]
            else:
                target_dirs = []
                for cid in target_class_ids:
                    class_name = class_mapping.get(cid, f"class_{cid}")
                    target_dirs.append(temp_dir / split / class_name)

            for dest_dir in target_dirs:
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_path, dest_dir / img_path.name)
                copied_counts[f"{split}/{dest_dir.name}"] += 1

    print("Conversion finished successfully. Statistics per split/class folder:")
    for folder, count in sorted(copied_counts.items()):
        print(f"  {folder}: {count} images")

    # Rename / Backup directories safely
    if args.backup:
        backup_dir = args.dataset_dir.parent / f"{args.dataset_dir.name}_backup"
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        print(f"Backing up original dataset to {backup_dir.name}...")
        args.dataset_dir.rename(backup_dir)
    else:
        print("Removing original dataset...")
        shutil.rmtree(args.dataset_dir)

    print(f"Renaming new classification dataset to {args.dataset_dir.name}...")
    temp_dir.rename(args.dataset_dir)
    print("Done!")


if __name__ == "__main__":
    main()
