from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


def read_gray(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"Could not encode: {path}")
    encoded.tofile(str(path))


def average_hash(path: Path) -> str | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    small = cv2.resize(image, (8, 8), interpolation=cv2.INTER_AREA)
    mean_value = small.mean()
    return "".join("1" if pixel > mean_value else "0" for pixel in small.flatten())


def pair_files(root: Path, split: str) -> list[tuple[Path, Path]]:
    image_dir = root / split / "images"
    mask_dir = root / split / "masks"
    if not image_dir.exists() or not mask_dir.exists():
        return []
    images = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    pairs = []
    for image in images:
        mask = mask_dir / f"{image.stem}.png"
        if mask.exists():
            pairs.append((image, mask))
    return pairs


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    for split in ("train", "test"):
        (path / split / "images").mkdir(parents=True, exist_ok=True)
        (path / split / "masks").mkdir(parents=True, exist_ok=True)


def copy_pairs(pairs: list[tuple[Path, Path]], output_root: Path, split: str) -> int:
    count = 0
    for image, mask in pairs:
        shutil.copy2(image, output_root / split / "images" / image.name)
        shutil.copy2(mask, output_root / split / "masks" / f"{image.stem}.png")
        count += 1
    return count


def find_yolo_label(image_path: Path) -> Path | None:
    candidates = [
        image_path.parent.parent / "labels" / f"{image_path.stem}.txt",
        image_path.with_suffix(".txt"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def yolo_to_mask(image_path: Path, label_path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {image_path}")
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            if int(float(parts[0])) != 0:
                continue
            values = np.asarray([float(v) for v in parts[1:]], dtype=np.float32).reshape(-1, 2)
            values[:, 0] = np.clip(values[:, 0] * width, 0, width - 1)
            values[:, 1] = np.clip(values[:, 1] * height, 0, height - 1)
            polygon = np.rint(values).astype(np.int32)
            if len(polygon) >= 3:
                cv2.fillPoly(mask, [polygon], 255)
    return mask


def build_final_173(source: Path, locked_test: list[tuple[Path, Path]], output: Path, test_hashes: set[str]) -> None:
    reset_dir(output)
    copy_pairs(locked_test, output, "test")

    candidates = []
    for image, mask in pair_files(source, "train"):
        image_hash = average_hash(image)
        if image_hash and image_hash in test_hashes:
            continue
        candidates.append((image, mask))
    copy_pairs(candidates, output, "train")


def build_final_70(source: Path, locked_test: list[tuple[Path, Path]], output: Path, test_hashes: set[str]) -> None:
    reset_dir(output)
    source_pairs = pair_files(source, "train")
    train_pairs = []
    for image, mask in source_pairs:
        image_hash = average_hash(image)
        if image_hash and image_hash in test_hashes:
            continue
        train_pairs.append((image, mask))
    copy_pairs(train_pairs, output, "train")
    copy_pairs(locked_test, output, "test")


def build_final_1775(source: Path, locked_test: list[tuple[Path, Path]], output: Path, test_hashes: set[str]) -> None:
    reset_dir(output)
    copy_pairs(locked_test, output, "test")

    image_paths = sorted(source.rglob("*.jpg"))
    copied = 0
    for image_path in image_paths:
        image_hash = average_hash(image_path)
        if image_hash and image_hash in test_hashes:
            continue
        label_path = find_yolo_label(image_path)
        if label_path is None:
            continue
        mask = yolo_to_mask(image_path, label_path)
        target_image = output / "train" / "images" / image_path.name
        target_mask = output / "train" / "masks" / f"{image_path.stem}.png"
        if target_image.exists():
            raise RuntimeError(f"Filename collision detected: {target_image.name}")
        shutil.copy2(image_path, target_image)
        write_png(target_mask, mask)
        copied += 1
    print(f"FINAL_1775 development images written: {copied}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the finalized benchmark datasets around one locked 15-image test set.")
    parser.add_argument("--source-70", type=Path, required=True)
    parser.add_argument("--source-173", type=Path, required=True)
    parser.add_argument("--source-1775", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("."))
    args = parser.parse_args()

    locked_test = pair_files(args.source_70, "test")
    if len(locked_test) != 15:
        raise RuntimeError(f"Expected 15 locked test images in source-70, found {len(locked_test)}")

    test_hashes = {h for image, _ in locked_test if (h := average_hash(image))}

    build_final_70(args.source_70, locked_test, args.output_root / "FINAL_70", test_hashes)
    build_final_173(args.source_173, locked_test, args.output_root / "FINAL_173", test_hashes)
    build_final_1775(args.source_1775, locked_test, args.output_root / "FINAL_1775", test_hashes)

    print("Final benchmark datasets created.")


if __name__ == "__main__":
    main()
