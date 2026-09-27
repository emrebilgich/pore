from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

DATASETS = ["FINAL_70", "FINAL_173", "FINAL_1775"]


def average_hash(path: Path) -> str | None:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    image = cv2.resize(image, (8, 8), interpolation=cv2.INTER_AREA)
    mean_value = image.mean()
    return "".join("1" if pixel > mean_value else "0" for pixel in image.flatten())


def main() -> None:
    test_names = None
    test_hashes = set()

    for dataset_name in DATASETS:
        root = Path(dataset_name)
        train_images = sorted((root / "train" / "images").glob("*.jpg"))
        train_masks = sorted((root / "train" / "masks").glob("*.png"))
        test_images = sorted((root / "test" / "images").glob("*.jpg"))
        test_masks = sorted((root / "test" / "masks").glob("*.png"))

        if len(test_images) != 15:
            raise RuntimeError(f"{dataset_name}: expected 15 test images, found {len(test_images)}")
        if len(train_images) != len(train_masks):
            raise RuntimeError(f"{dataset_name}: train image/mask count mismatch")
        if len(test_images) != len(test_masks):
            raise RuntimeError(f"{dataset_name}: test image/mask count mismatch")

        names = sorted(p.stem for p in test_images)
        if test_names is None:
            test_names = names
        elif names != test_names:
            raise RuntimeError(f"{dataset_name}: locked test filenames do not match FINAL_70")

        hashes = {h for p in test_images if (h := average_hash(p))}
        if dataset_name == DATASETS[0]:
            test_hashes = hashes

        leaking_hashes = []
        for image in train_images:
            image_hash = average_hash(image)
            if image_hash and image_hash in test_hashes:
                leaking_hashes.append(image.name)

        print(f"{dataset_name}: train={len(train_images)}, test={len(test_images)}, hash-matched test/train overlaps={len(leaking_hashes)}")
        if leaking_hashes:
            print("  Matches:", leaking_hashes[:20])

    print("Benchmark dataset verification completed.")
    print("Note: average hashing is a screening method; it is not a mathematical proof of zero leakage.")


if __name__ == "__main__":
    main()
