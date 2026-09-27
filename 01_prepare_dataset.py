from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(path)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"Could not encode: {path}")
    encoded.tofile(str(path))


def convert_split(root: Path, split: str) -> dict:
    image_dir = root / split / "images"
    label_dir = root / split / "labels"
    mask_dir = root / split / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.exists():
        return {"split": split, "images": 0, "missing_labels": 0, "invalid_lines": 0, "nonempty_masks": 0, "empty_masks": 0}

    image_paths = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    audit = {
        "split": split,
        "images": len(image_paths),
        "missing_labels": 0,
        "invalid_lines": 0,
        "nonempty_masks": 0,
        "empty_masks": 0,
    }

    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        mask_path = mask_dir / f"{image_path.stem}.png"
        image = read_image(image_path)
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        if not label_path.exists():
            audit["missing_labels"] += 1
        else:
            with label_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    parts = line.strip().split()
                    if not parts:
                        continue

                    try:
                        class_id = int(float(parts[0]))
                    except ValueError:
                        audit["invalid_lines"] += 1
                        continue

                    if class_id != 0:
                        continue

                    coordinates = parts[1:]
                    if len(coordinates) < 6 or len(coordinates) % 2 != 0:
                        audit["invalid_lines"] += 1
                        continue

                    try:
                        values = np.asarray([float(v) for v in coordinates], dtype=np.float32).reshape(-1, 2)
                    except ValueError:
                        audit["invalid_lines"] += 1
                        continue

                    values[:, 0] = np.clip(values[:, 0] * width, 0, width - 1)
                    values[:, 1] = np.clip(values[:, 1] * height, 0, height - 1)
                    polygon = np.rint(values).astype(np.int32)

                    if len(polygon) >= 3:
                        cv2.fillPoly(mask, [polygon], 255)
                    else:
                        audit["invalid_lines"] += 1

        if int(mask.sum()) > 0:
            audit["nonempty_masks"] += 1
        else:
            audit["empty_masks"] += 1

        write_png(mask_path, mask)

    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert YOLO polygon labels to binary pore masks and audit the dataset.")
    parser.add_argument("dataset", type=Path, help="Dataset root containing train/valid/test splits.")
    parser.add_argument("--output-audit", type=Path, default=None, help="Optional JSON audit output path.")
    args = parser.parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset.resolve()}")

    audit = [convert_split(args.dataset, split) for split in ("train", "valid", "test")]
    summary = {"dataset": str(args.dataset), "splits": audit}

    print(json.dumps(summary, indent=2))

    if args.output_audit:
        args.output_audit.parent.mkdir(parents=True, exist_ok=True)
        args.output_audit.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
