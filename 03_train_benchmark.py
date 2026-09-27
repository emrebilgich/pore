from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

from pore_models import count_parameters, create_model

SEED = 42
IMAGE_SIZE = 512
BATCH_SIZE = 4
EPOCHS = 40
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
VALIDATION_RATIO = 0.20
THRESHOLD_MIN = 0.05
THRESHOLD_MAX = 0.95
THRESHOLD_STEP = 0.01
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASETS = {
    "FINAL_70": Path("FINAL_70"),
    "FINAL_173": Path("FINAL_173"),
    "FINAL_1775": Path("FINAL_1775"),
}
MODELS = [
    "ResNet34_UNet_Baseline",
    "ResNet34_UNet_scSE",
    "Custom_Attention_UNet",
]
OUTPUT_ROOT = Path("FINAL_BENCHMARK_RESULTS")


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def read_image(path: Path, flags: int) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def find_pairs(dataset_root: Path, split: str) -> list[tuple[str, str]]:
    image_dir = dataset_root / split / "images"
    mask_dir = dataset_root / split / "masks"
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f"Missing split directories: {dataset_root / split}")
    images = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    pairs = []
    for image in images:
        mask = mask_dir / f"{image.stem}.png"
        if not mask.exists():
            raise FileNotFoundError(f"Missing mask: {mask}")
        pairs.append((str(image), str(mask)))
    return pairs


class PoreDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str]]) -> None:
        self.pairs = pairs
        self.transform = A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        image_path, mask_path = self.pairs[index]
        image = cv2.cvtColor(read_image(Path(image_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask = (read_image(Path(mask_path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
        transformed = self.transform(image=image, mask=mask)
        return transformed["image"].float(), transformed["mask"].float().unsqueeze(0)


class BCEDiceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        probabilities = torch.sigmoid(logits).reshape(logits.size(0), -1)
        targets = targets.reshape(targets.size(0), -1)
        intersection = (probabilities * targets).sum(dim=1)
        dice = (2.0 * intersection + 1.0) / (probabilities.sum(dim=1) + targets.sum(dim=1) + 1.0)
        return bce + (1.0 - dice.mean())


@torch.no_grad()
def validation_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> float:
    model.eval()
    total = 0.0
    for images, masks in loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        total += criterion(model(images), masks).item()
    return total / max(len(loader), 1)


@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities, truths = [], []
    for images, masks in loader:
        probabilities.append(torch.sigmoid(model(images.to(DEVICE))).cpu().numpy())
        truths.append(masks.numpy())
    return np.concatenate(probabilities), np.concatenate(truths)


def f1_from_masks(probabilities: np.ndarray, truths: np.ndarray, threshold: float) -> float:
    predictions = probabilities > threshold
    tp = np.sum((truths == 1) & predictions)
    fp = np.sum((truths == 0) & predictions)
    fn = np.sum((truths == 1) & ~predictions)
    return float((2 * tp) / (2 * tp + fp + fn + 1e-7))


def select_threshold(probabilities: np.ndarray, truths: np.ndarray) -> tuple[float, float]:
    best_threshold, best_f1 = 0.50, -1.0
    thresholds = np.arange(THRESHOLD_MIN, THRESHOLD_MAX + 0.0001, THRESHOLD_STEP)
    for threshold in thresholds:
        score = f1_from_masks(probabilities, truths, float(threshold))
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold, best_f1


@torch.no_grad()
def evaluate_test(model: nn.Module, loader: DataLoader, threshold: float) -> dict:
    model.eval()
    tp = fp = fn = 0
    for images, masks in loader:
        predictions = (torch.sigmoid(model(images.to(DEVICE))).cpu().numpy() > threshold)
        truths = masks.numpy()
        tp += int(np.sum((truths == 1) & predictions))
        fp += int(np.sum((truths == 0) & predictions))
        fn += int(np.sum((truths == 1) & ~predictions))

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = (2 * tp) / (2 * tp + fp + fn + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)
    return {"Precision": precision, "Recall": recall, "F1_Dice": f1, "IoU": iou, "TP": tp, "FP": fp, "FN": fn}


def save_manifest(pairs: list[tuple[str, str]], path: Path) -> None:
    pd.DataFrame({"image": [p[0] for p in pairs], "mask": [p[1] for p in pairs]}).to_csv(path, index=False)


def run(output_root: Path) -> pd.DataFrame:
    seed_everything(SEED)
    output_root.mkdir(parents=True, exist_ok=True)
    all_results = []
    criterion = BCEDiceLoss()

    for dataset_name, dataset_root in DATASETS.items():
        train_pool = find_pairs(dataset_root, "train")
        test_pairs = find_pairs(dataset_root, "test")
        if len(test_pairs) != 15:
            raise RuntimeError(f"{dataset_name}: expected 15 locked test images, found {len(test_pairs)}")

        train_pairs, validation_pairs = train_test_split(
            train_pool,
            test_size=VALIDATION_RATIO,
            random_state=SEED,
            shuffle=True,
        )

        dataset_output = output_root / dataset_name
        dataset_output.mkdir(parents=True, exist_ok=True)
        save_manifest(train_pairs, dataset_output / "train_manifest.csv")
        save_manifest(validation_pairs, dataset_output / "validation_manifest.csv")
        save_manifest(test_pairs, dataset_output / "test_manifest.csv")

        train_dataset = PoreDataset(train_pairs)
        validation_dataset = PoreDataset(validation_pairs)
        test_dataset = PoreDataset(test_pairs)

        # One generator is intentionally reused across the fixed model order,
        # matching the reported benchmark run.
        generator = torch.Generator().manual_seed(SEED)

        for model_name in MODELS:
            seed_everything(SEED)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, generator=generator)
            validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

            model = create_model(model_name, pretrained=True).to(DEVICE)
            parameter_count = count_parameters(model)
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

            experiment_dir = output_root / f"{dataset_name}__{model_name}__seed{SEED}"
            experiment_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = experiment_dir / "best_model.pth"

            best_valid_loss = float("inf")
            best_epoch = 0
            history = []

            for epoch in range(1, EPOCHS + 1):
                model.train()
                train_loss_sum = 0.0
                for images, masks in train_loader:
                    images, masks = images.to(DEVICE), masks.to(DEVICE)
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(model(images), masks)
                    loss.backward()
                    optimizer.step()
                    train_loss_sum += loss.item()

                train_loss = train_loss_sum / max(len(train_loader), 1)
                val_loss = validation_loss(model, validation_loader, criterion)
                scheduler.step(val_loss)
                history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "learning_rate": optimizer.param_groups[0]["lr"]})

                if val_loss < best_valid_loss:
                    best_valid_loss = val_loss
                    best_epoch = epoch
                    torch.save(model.state_dict(), checkpoint_path)

                if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
                    print(f"{dataset_name} | {model_name} | epoch {epoch:02d}/{EPOCHS} | train {train_loss:.4f} | val {val_loss:.4f}")

            pd.DataFrame(history).to_csv(experiment_dir / "training_history.csv", index=False)

            model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))
            validation_probabilities, validation_truths = collect_predictions(model, validation_loader)
            threshold, validation_f1 = select_threshold(validation_probabilities, validation_truths)
            test_metrics = evaluate_test(model, test_loader, threshold)

            metadata = {
                "dataset": dataset_name,
                "model": model_name,
                "seed": SEED,
                "image_size": IMAGE_SIZE,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "validation_ratio": VALIDATION_RATIO,
                "train_images": len(train_pairs),
                "validation_images": len(validation_pairs),
                "test_images": len(test_pairs),
                "parameter_count": parameter_count,
                "best_epoch": best_epoch,
                "best_validation_loss": best_valid_loss,
                "validation_selected_threshold": threshold,
                "validation_f1_at_threshold": validation_f1,
                "test_metrics": test_metrics,
                "test_used_for_threshold_selection": False,
            }
            (experiment_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            all_results.append({
                "Dataset": dataset_name,
                "Model": model_name,
                "Seed": SEED,
                "Train": len(train_pairs),
                "Validation": len(validation_pairs),
                "Test": len(test_pairs),
                "Parameters": parameter_count,
                "Best Epoch": best_epoch,
                "Best Val Loss": round(best_valid_loss, 6),
                "Threshold": round(threshold, 2),
                "Validation F1": round(validation_f1, 6),
                "Test F1_Dice": round(test_metrics["F1_Dice"], 6),
                "Test IoU": round(test_metrics["IoU"], 6),
                "Precision": round(test_metrics["Precision"], 6),
                "Recall": round(test_metrics["Recall"], 6),
                "TP": test_metrics["TP"],
                "FP": test_metrics["FP"],
                "FN": test_metrics["FN"],
            })

            pd.DataFrame(all_results).to_csv(output_root / "FINAL_BENCHMARK_RESULTS.csv", index=False)
            del model, optimizer, scheduler
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return pd.DataFrame(all_results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the three segmentation models on all three benchmark datasets.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    results = run(args.output_root)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
