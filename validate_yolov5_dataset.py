"""Validate an engineer-reviewed YOLOv5 P&ID dataset before training.

The validator deliberately blocks training when review metadata, a held-out test
split, negative examples, or licence confirmation are missing. It also checks
that no identical image bytes occur in more than one split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
REVIEW_REQUIRED = {
    "status", "approved_by", "drawing_count", "split_by_drawing",
    "license_confirmed", "negative_examples",
}


def _resolve(root: Path, value: str | list[str]) -> list[Path]:
    values = value if isinstance(value, list) else [value]
    output: list[Path] = []
    for item in values:
        candidate = Path(str(item))
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.is_file() and candidate.suffix.lower() == ".txt":
            for line in candidate.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    image = Path(line)
                    output.append(image if image.is_absolute() else root / image)
        elif candidate.is_dir():
            output.extend(p for p in candidate.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        elif candidate.suffix.lower() in IMAGE_SUFFIXES:
            output.append(candidate)
    return sorted({p.resolve() for p in output})


def _label_path(image: Path) -> Path:
    parts = list(image.parts)
    if "images" in parts:
        index = len(parts) - 1 - parts[::-1].index("images")
        parts[index] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image.with_suffix(".txt")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _class_names(value) -> list[str]:
    if isinstance(value, dict):
        return [str(value[key]) for key in sorted(value, key=lambda x: int(x))]
    if isinstance(value, list):
        return [str(item) for item in value]
    raise ValueError("data.yaml 'names' must be a list or integer-keyed mapping")


def validate_dataset(data_yaml: str | Path) -> dict:
    cfg_path = Path(data_yaml).resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("data.yaml must contain a mapping")
    missing = {"train", "val", "test", "names"} - set(cfg)
    if missing:
        raise ValueError("data.yaml is missing: " + ", ".join(sorted(missing)))

    names = _class_names(cfg["names"])
    if not names or any(not name.strip() for name in names):
        raise ValueError("At least one non-empty P&ID symbol class is required")

    review_path = cfg_path.parent / "dataset_review.json"
    if not review_path.is_file():
        raise ValueError("dataset_review.json is required beside data.yaml")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    absent = REVIEW_REQUIRED - set(review)
    if absent:
        raise ValueError("dataset_review.json is missing: " + ", ".join(sorted(absent)))
    if str(review["status"]).upper() != "APPROVED" or not str(review["approved_by"]).strip():
        raise ValueError("The dataset must be APPROVED by an identified engineering reviewer")
    if int(review["drawing_count"]) < 3 or review["split_by_drawing"] is not True:
        raise ValueError("Use at least three drawings and split by complete drawing, never random tiles")
    if review["license_confirmed"] is not True:
        raise ValueError("Training data and symbol-library usage rights must be confirmed")
    if review["negative_examples"] is not True:
        raise ValueError("Negative/background examples must be reviewed and included")

    root = Path(str(cfg.get("path", cfg_path.parent)))
    if not root.is_absolute():
        root = (cfg_path.parent / root).resolve()
    splits = {name: _resolve(root, cfg[name]) for name in ("train", "val", "test")}
    for name, images in splits.items():
        if not images:
            raise ValueError(f"{name} split contains no readable image paths")
        missing_images = [str(path) for path in images if not path.is_file()]
        if missing_images:
            raise ValueError(f"{name} split contains missing images: {missing_images[:3]}")

    seen_hashes: dict[str, str] = {}
    class_counts = {name: 0 for name in names}
    backgrounds = 0
    label_rows = 0
    for split, images in splits.items():
        for image in images:
            digest = _hash(image)
            if digest in seen_hashes and seen_hashes[digest] != split:
                raise ValueError(f"Identical image bytes occur in {seen_hashes[digest]} and {split}: {image.name}")
            seen_hashes[digest] = split
            label = _label_path(image)
            if not label.is_file() or not label.read_text(encoding="utf-8").strip():
                backgrounds += 1
                continue
            for number, raw in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
                parts = raw.split()
                if len(parts) != 5:
                    raise ValueError(f"{label}:{number} must contain class x_center y_center width height")
                class_id = int(parts[0])
                values = [float(value) for value in parts[1:]]
                if not 0 <= class_id < len(names):
                    raise ValueError(f"{label}:{number} class {class_id} is outside names[]")
                if not all(0.0 <= value <= 1.0 for value in values) or values[2] <= 0 or values[3] <= 0:
                    raise ValueError(f"{label}:{number} contains an invalid normalized box")
                class_counts[names[class_id]] += 1
                label_rows += 1

    if label_rows == 0:
        raise ValueError("No reviewed symbol boxes were found")
    empty_classes = [name for name, count in class_counts.items() if count == 0]
    if empty_classes:
        raise ValueError("Classes with no examples: " + ", ".join(empty_classes))
    if backgrounds == 0:
        raise ValueError("No negative/background image was found despite review declaration")

    return {
        "data_yaml": str(cfg_path), "root": str(root), "classes": names,
        "images": {key: len(value) for key, value in splits.items()},
        "labelled_boxes": label_rows, "background_images": backgrounds,
        "class_counts": class_counts, "review": review,
        "status": "TRAINING_GATE_PASSED",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_yaml")
    parser.add_argument("--report", default="dataset_validation.json")
    args = parser.parse_args()
    result = validate_dataset(args.data_yaml)
    Path(args.report).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
