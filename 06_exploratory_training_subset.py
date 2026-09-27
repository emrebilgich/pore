from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from pore_models import create_model

SEED = 42
IMAGE_SIZE = 512
BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = ["ResNet34_UNet_Baseline", "ResNet34_UNet_scSE", "Custom_Attention_UNet"]


class TrainingSubset(Dataset):
    def __init__(self, pairs: list[tuple[Path, Path]]) -> None:
        self.pairs = pairs
        self.transform = A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image = cv2.cvtColor(cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask = (cv2.imdecode(np.fromfile(str(mask_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
        transformed = self.transform(image=image, mask=mask)
        return transformed["image"].float(), transformed["mask"].float().unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exploratory threshold curves on a subset of seen training images.")
    parser.add_argument("--dataset", type=Path, default=Path("FINAL_1775"))
    parser.add_argument("--num-images", type=int, default=150)
    parser.add_argument("--output", type=Path, default=Path("EXPLORATORY_RANDOM_TEST"))
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    images = sorted((args.dataset / "train" / "images").glob("*.jpg"))
    random.shuffle(images)
    images = images[: min(args.num_images, len(images))]
    pairs = [(p, args.dataset / "train" / "masks" / f"{p.stem}.png") for p in images]
    pairs = [pair for pair in pairs if pair[1].exists()]
    loader = DataLoader(TrainingSubset(pairs), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    thresholds = np.arange(0.05, 0.951, 0.05)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for model_name in MODELS:
        checkpoint = Path("FINAL_BENCHMARK_RESULTS") / f"{args.dataset.name}__{model_name}__seed{SEED}" / "best_model.pth"
        model = create_model(model_name, pretrained=False).to(DEVICE)
        model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True), strict=True)
        model.eval()

        probabilities, truths = [], []
        with torch.no_grad():
            for images_batch, masks_batch in loader:
                probabilities.append(torch.sigmoid(model(images_batch.to(DEVICE))).cpu().numpy())
                truths.append(masks_batch.numpy())

        probabilities = np.concatenate(probabilities).ravel()
        truths = np.concatenate(truths).ravel() > 0.5
        f1_scores, iou_scores = [], []

        for threshold in thresholds:
            predictions = probabilities > threshold
            tp = np.sum(truths & predictions)
            fp = np.sum(~truths & predictions)
            fn = np.sum(truths & ~predictions)
            f1 = 2 * tp / (2 * tp + fp + fn + 1e-7)
            iou = tp / (tp + fp + fn + 1e-7)
            f1_scores.append(f1)
            iou_scores.append(iou)

        label = model_name.replace("_", " ")
        axes[0].plot(thresholds, f1_scores, marker="o", label=label)
        axes[1].plot(thresholds, iou_scores, marker="o", label=label)
        del model

    axes[0].set_title("Threshold sensitivity on seen training images")
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("F1 / Dice")
    axes[1].set_title("Threshold sensitivity on seen training images")
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("IoU")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()

    fig.tight_layout()
    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / "training_subset_threshold_sweep.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
