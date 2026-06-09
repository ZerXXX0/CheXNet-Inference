#!/usr/bin/env python3
"""Run CheXNet-style inference on a test image folder.

Defaults are aligned with this workspace:
- Checkpoint: model.pth.tar
- Dataset config: Splitted dataset/data.yaml
- Test images: Splitted dataset/images/test
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torchvision import models, transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

DEFAULT_CHEXNET14_CLASS_NAMES = [
    "atelectasis",
    "cardiomegaly",
    "effusion",
    "infiltration",
    "mass",
    "nodule",
    "pneumonia",
    "pneumothorax",
    "consolidation",
    "edema",
    "emphysema",
    "fibrosis",
    "pleural_thickening",
    "hernia",
]

DATASET_NAME_TO_CANONICAL = {
    "kavitas": "cavity",
    "infiltrat": "infiltration",
    "limfadenopati": "lymphadenopathy",
    "tuberkuloma": "tuberculoma",
    "bronkiektasis": "bronchiectasis",
    "pneumothorax": "pneumothorax",
    "efusi pleura": "effusion",
    "atelektasis": "atelectasis",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference on test set using model.pth.tar")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("model.pth.tar"),
        help="Path to checkpoint (.pth/.pth.tar).",
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=Path("Splitted dataset") / "data.yaml",
        help="Path to dataset YAML with class names.",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("Splitted dataset") / "test",
        help="Path to test directory (containing class subfolders).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("inference_test_results.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="Deprecated: labels are now read directly from class subdirectories.",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=Path("inference_test_metrics.json"),
        help="Output JSON path for evaluation metrics.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Input image size (square).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Sigmoid threshold for multi-label predictions.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference.",
    )
    return parser.parse_args()


def read_class_names(data_yaml: Path) -> List[str]:
    """Read class names from a YOLO-style YAML file without hard dependency on PyYAML."""
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    text = data_yaml.read_text(encoding="utf-8")

    # Try PyYAML first when available.
    try:
        import yaml  # type: ignore

        parsed = yaml.safe_load(text)
        names_obj = parsed.get("names", {}) if isinstance(parsed, dict) else {}
        if isinstance(names_obj, dict):
            items = sorted((int(k), str(v)) for k, v in names_obj.items())
            return [name for _, name in items]
        if isinstance(names_obj, list):
            return [str(x) for x in names_obj]
    except Exception:
        pass

    # Fallback parser for simple `names:` mappings.
    names: Dict[int, str] = {}
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
                    except ValueError:
                        continue
                    names[idx] = v.strip().strip('"\'')
            else:
                break

    if not names:
        raise ValueError(f"Could not parse class names from: {data_yaml}")

    return [name for _, name in sorted(names.items(), key=lambda kv: kv[0])]


def build_model(num_classes: int) -> torch.nn.Module:
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = torch.nn.Linear(in_features, num_classes)
    return model


def pick_state_dict(checkpoint_obj: object) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint_obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            state = checkpoint_obj.get(key)
            if isinstance(state, dict):
                return state
        # Sometimes checkpoint is already a raw state_dict.
        if all(isinstance(k, str) for k in checkpoint_obj.keys()):
            if any("weight" in k or "bias" in k for k in checkpoint_obj.keys()):
                return checkpoint_obj  # type: ignore[return-value]
    raise ValueError("Unsupported checkpoint format; expected state_dict-like object.")


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())
    if not has_module_prefix:
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def normalize_densenet_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Normalize common legacy CheXNet/DenseNet checkpoint key formats."""
    normalized: Dict[str, torch.Tensor] = {}

    replacements: Tuple[Tuple[str, str], ...] = (
        (".norm.1.", ".norm1."),
        (".relu.1.", ".relu1."),
        (".conv.1.", ".conv1."),
        (".norm.2.", ".norm2."),
        (".relu.2.", ".relu2."),
        (".conv.2.", ".conv2."),
    )

    for key, value in state_dict.items():
        new_key = key

        if new_key.startswith("densenet121."):
            new_key = new_key.replace("densenet121.", "", 1)

        for old, new in replacements:
            new_key = new_key.replace(old, new)

        if new_key.startswith("classifier.0."):
            new_key = new_key.replace("classifier.0.", "classifier.", 1)

        normalized[new_key] = value

    return normalized


def infer_num_classes_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> int | None:
    weight = state_dict.get("classifier.weight")
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        return int(weight.shape[0])
    bias = state_dict.get("classifier.bias")
    if isinstance(bias, torch.Tensor) and bias.ndim == 1:
        return int(bias.shape[0])
    return None


def default_model_class_names(num_classes: int) -> List[str]:
    if num_classes == len(DEFAULT_CHEXNET14_CLASS_NAMES):
        return list(DEFAULT_CHEXNET14_CLASS_NAMES)
    return [f"class_{i}" for i in range(num_classes)]


def normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", " ").replace("_", " ")


def canonicalize_dataset_name(name: str) -> str:
    normalized = normalize_name(name)
    mapped = DATASET_NAME_TO_CANONICAL.get(normalized, normalized)
    return mapped.replace(" ", "_")


def canonicalize_model_name(name: str) -> str:
    return normalize_name(name).replace(" ", "_")


def build_intersection_map(
    dataset_class_names: List[str],
    model_class_names: List[str],
) -> List[Tuple[int, str, str]]:
    """Return tuples of (model_index, dataset_display_name, canonical_name)."""
    model_index_by_canonical = {
        canonicalize_model_name(model_name): idx for idx, model_name in enumerate(model_class_names)
    }

    intersection: List[Tuple[int, str, str]] = []
    for dataset_name in dataset_class_names:
        canonical = canonicalize_dataset_name(dataset_name)
        if canonical in model_index_by_canonical:
            intersection.append((model_index_by_canonical[canonical], dataset_name, canonical))

    return intersection


def get_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def collect_images(test_dir: Path) -> List[Path]:
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    unique_images: Dict[str, Path] = {}
    for p in sorted(test_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            if p.name not in unique_images:
                unique_images[p.name] = p
                
    if not unique_images:
        raise ValueError(f"No images found in: {test_dir}")
    
    return list(unique_images.values())


def load_ground_truth_vectors(
    test_dir: Path,
    intersect_map: List[Tuple[int, str, str]],
) -> Dict[str, List[int]]:
    """Build ground truth vectors for all unique images in the test set by scanning the class directories."""
    dataset_name_to_local_idx = {
        dataset_display_name: local_idx
        for local_idx, (_, dataset_display_name, _) in enumerate(intersect_map)
    }

    class_count = len(intersect_map)
    gt_vectors: Dict[str, List[int]] = {}

    for class_dir in test_dir.iterdir():
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        local_idx = dataset_name_to_local_idx.get(class_name)

        for img_path in class_dir.iterdir():
            if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                img_name = img_path.name
                if img_name not in gt_vectors:
                    gt_vectors[img_name] = [0] * class_count
                
                if local_idx is not None:
                    gt_vectors[img_name][local_idx] = 1

    return gt_vectors


def compute_metrics(
    pred_vectors: Dict[str, List[int]],
    gt_vectors: Dict[str, List[int]],
    class_names: List[str],
) -> Dict[str, object]:
    image_names = [name for name in pred_vectors.keys() if name in gt_vectors]
    n = len(image_names)
    if n == 0:
        raise ValueError("No overlapping images between predictions and ground-truth labels.")

    class_metrics: Dict[str, Dict[str, float | int]] = {}
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0

    micro_tp = 0
    micro_fp = 0
    micro_fn = 0

    exact_match = 0

    for image_name in image_names:
        if pred_vectors[image_name] == gt_vectors[image_name]:
            exact_match += 1

    for class_idx, class_name in enumerate(class_names):
        tp = fp = tn = fn = 0
        for image_name in image_names:
            p = pred_vectors[image_name][class_idx]
            t = gt_vectors[image_name][class_idx]
            if p == 1 and t == 1:
                tp += 1
            elif p == 1 and t == 0:
                fp += 1
            elif p == 0 and t == 1:
                fn += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        support = tp + fn

        class_metrics[class_name] = {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "support": support,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }

        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

    class_count = len(class_names)
    macro_precision /= class_count
    macro_recall /= class_count
    macro_f1 /= class_count

    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    return {
        "num_images": n,
        "exact_match_accuracy": round(exact_match / n, 6),
        "micro": {
            "precision": round(micro_precision, 6),
            "recall": round(micro_recall, 6),
            "f1": round(micro_f1, 6),
        },
        "macro": {
            "precision": round(macro_precision, 6),
            "recall": round(macro_recall, 6),
            "f1": round(macro_f1, 6),
        },
        "per_class": class_metrics,
    }


def save_metrics(metrics: Dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


@torch.inference_mode()
def run_inference(
    model: torch.nn.Module,
    image_paths: List[Path],
    intersect_map: List[Tuple[int, str, str]],
    transform: transforms.Compose,
    device: torch.device,
    threshold: float,
) -> Tuple[List[Dict[str, str]], Dict[str, List[int]]]:
    if not intersect_map:
        raise ValueError("No intersected classes found between dataset names and model classes.")

    results: List[Dict[str, str]] = []
    pred_vectors: Dict[str, List[int]] = {}
    model.eval()

    for image_path in image_paths:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)

        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze(0).detach().cpu()

        intersect_scores = [float(probs[model_idx].item()) for model_idx, _, _ in intersect_map]
        top_local_index = int(torch.tensor(intersect_scores).argmax().item())
        top_label = intersect_map[top_local_index][1]
        top_score = intersect_scores[top_local_index]

        positive_labels = [
            dataset_display_name
            for (model_idx, dataset_display_name, _) in intersect_map
            if float(probs[model_idx].item()) >= threshold
        ]

        pred_vector = [
            1 if float(probs[model_idx].item()) >= threshold else 0
            for (model_idx, _, _) in intersect_map
        ]

        row: Dict[str, str] = {
            "image": image_path.name,
            "top1_label": top_label,
            "top1_score": f"{top_score:.6f}",
            "predicted_labels": "|".join(positive_labels),
        }

        for model_idx, dataset_display_name, _ in intersect_map:
            row[f"prob_{dataset_display_name}"] = f"{float(probs[model_idx]):.6f}"

        results.append(row)
        pred_vectors[image_path.name] = pred_vector

    return results, pred_vectors


def save_csv(rows: List[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["image", "top1_label", "top1_score", "predicted_labels"]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    dataset_class_names = read_class_names(args.data_yaml)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    checkpoint_obj = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = strip_module_prefix(pick_state_dict(checkpoint_obj))
    state_dict = normalize_densenet_keys(state_dict)

    inferred_num_classes = infer_num_classes_from_state_dict(state_dict)
    if inferred_num_classes is None:
        raise ValueError("Could not infer classifier output size from checkpoint.")

    num_classes = inferred_num_classes
    model_class_names = default_model_class_names(num_classes)
    intersect_map = build_intersection_map(dataset_class_names, model_class_names)

    if not intersect_map:
        raise ValueError(
            "No class name intersection between dataset YAML and model classes. "
            "Update DATASET_NAME_TO_CANONICAL mapping in inference.py."
        )

    kept_dataset_names = [dataset_name for _, dataset_name, _ in intersect_map]
    print(f"Using intersect classes ({len(kept_dataset_names)}): {kept_dataset_names}")

    model = build_model(num_classes=num_classes).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"Warning: missing keys in checkpoint load ({len(missing)}): {missing[:6]}{'...' if len(missing) > 6 else ''}")
    if unexpected:
        print(
            f"Warning: unexpected keys in checkpoint load ({len(unexpected)}): "
            f"{unexpected[:6]}{'...' if len(unexpected) > 6 else ''}"
        )

    image_paths = collect_images(args.test_dir)
    transform = get_transform(args.image_size)

    rows, pred_vectors = run_inference(
        model=model,
        image_paths=image_paths,
        intersect_map=intersect_map,
        transform=transform,
        device=device,
        threshold=args.threshold,
    )

    gt_vectors = load_ground_truth_vectors(
        test_dir=args.test_dir,
        intersect_map=intersect_map,
    )

    metric_class_names = [dataset_display_name for _, dataset_display_name, _ in intersect_map]
    metrics = compute_metrics(pred_vectors=pred_vectors, gt_vectors=gt_vectors, class_names=metric_class_names)

    save_csv(rows, args.output_csv)
    save_metrics(metrics, args.metrics_json)

    print(f"Processed {len(rows)} images from: {args.test_dir}")
    print(f"Saved predictions to: {args.output_csv}")
    print(f"Saved evaluation metrics to: {args.metrics_json}")
    print(
        "Metrics summary - "
        f"ExactMatch: {metrics['exact_match_accuracy']}, "
        f"Micro-F1: {metrics['micro']['f1']}, "
        f"Macro-F1: {metrics['macro']['f1']}"
    )


if __name__ == "__main__":
    main()
