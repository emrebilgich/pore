from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from pore_models import create_model

SEED = 42
IMAGE_SIZE = 512
BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASETS = ["FINAL_70", "FINAL_173", "FINAL_1775"]
MODELS = ["ResNet34_UNet_Baseline", "ResNet34_UNet_scSE", "Custom_Attention_UNet"]
RESULTS_ROOT = Path("FINAL_BENCHMARK_RESULTS")
OUTPUT_ROOT = RESULTS_ROOT / "FINAL_POST_EVALUATION"


def read_image(path: Path, flags: int) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)
    if image is None:
        raise ValueError(f"Could not decode: {path}")
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Could not encode: {path}")
    encoded.tofile(str(path))


class LockedTestDataset(Dataset):
    def __init__(self, image_paths: list[Path], mask_dir: Path) -> None:
        self.items = []
        self.transform = A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        for image_path in image_paths:
            mask_path = mask_dir / f"{image_path.stem}.png"
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask: {mask_path}")
            self.items.append((image_path, mask_path))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        image_path, mask_path = self.items[index]
        image = cv2.cvtColor(read_image(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask = (read_image(mask_path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
        transformed = self.transform(image=image, mask=mask)
        return transformed["image"].float(), transformed["mask"].float().unsqueeze(0), str(image_path)


def metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    pred = pred.astype(bool)
    truth = truth.astype(bool)
    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "Precision": precision, "Recall": recall, "F1_Dice": f1, "IoU": iou}


def save_error_map(gt: np.ndarray, pred: np.ndarray, path: Path) -> None:
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    error = np.zeros((*gt.shape, 3), dtype=np.uint8)
    error[gt & pred] = [255, 255, 0]
    error[(~gt) & pred] = [255, 0, 0]
    error[gt & (~pred)] = [0, 0, 255]
    save_image(path, cv2.cvtColor(error, cv2.COLOR_RGB2BGR))


@torch.no_grad()
def predict(model: torch.nn.Module, dataset: LockedTestDataset):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    probabilities, masks, paths = [], [], []
    for images, ground_truth, image_paths in loader:
        probabilities.extend(torch.sigmoid(model(images.to(DEVICE))).squeeze(1).cpu().numpy())
        masks.extend(ground_truth.squeeze(1).numpy())
        paths.extend(image_paths)
    return probabilities, masks, paths


def read_results_table() -> pd.DataFrame:
    path = RESULTS_ROOT / "FINAL_BENCHMARK_RESULTS.csv"
    if not path.exists():
        raise FileNotFoundError(f"Benchmark results not found: {path}")
    return pd.read_csv(path)


def main() -> None:
    results = read_results_table()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    reference_stems = None
    for dataset_name in DATASETS:
        image_dir = Path(dataset_name) / "test" / "images"
        image_paths = sorted(image_dir.glob("*.jpg"))
        stems = sorted(p.stem for p in image_paths)
        reference_stems = stems if reference_stems is None else reference_stems
        if stems != reference_stems:
            raise RuntimeError(f"Locked test set mismatch: {dataset_name}")
        if len(stems) != 15:
            raise RuntimeError(f"Expected 15 test images in {dataset_name}, found {len(stems)}")

        dataset = LockedTestDataset(image_paths, Path(dataset_name) / "test" / "masks")

        for model_name in MODELS:
            row = results[(results["Dataset"] == dataset_name) & (results["Model"] == model_name) & (results["Seed"] == SEED)]
            if len(row) != 1:
                raise RuntimeError(f"Could not find unique benchmark row for {dataset_name} / {model_name}")
            row = row.iloc[0]
            threshold = float(row["Threshold"])

            experiment_dir = RESULTS_ROOT / f"{dataset_name}__{model_name}__seed{SEED}"
            checkpoint = experiment_dir / "best_model.pth"
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)

            model = create_model(model_name, pretrained=False).to(DEVICE)
            state = torch.load(checkpoint, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state, strict=True)
            model.eval()

            probabilities, masks, paths = predict(model, dataset)

            all_metrics = {"TP": 0, "FP": 0, "FN": 0}
            output_dir = OUTPUT_ROOT / f"{dataset_name}__{model_name}__seed{SEED}"
            for folder in ("original", "ground_truth", "prediction", "overlay", "error_map", "probability"):
                (output_dir / folder).mkdir(parents=True, exist_ok=True)

            per_image_rows = []
            for probability, gt, path_string in zip(probabilities, masks, paths):
                gt = cv2.resize(gt.astype(np.uint8), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
                pred = (probability >= threshold).astype(np.uint8)
                if pred.shape != gt.shape:
                    pred = cv2.resize(pred, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
                image_path = Path(path_string)
                original = cv2.cvtColor(read_image(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                original = cv2.resize(original, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

                overlay = original.copy()
                overlay[gt == 1] = ((0.5 * overlay[gt == 1]) + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
                overlay[pred == 1] = ((0.5 * overlay[pred == 1]) + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
                probability_map = cv2.applyColorMap((np.clip(probability, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
                probability_map = cv2.resize(probability_map, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

                file_stem = image_path.stem
                save_image(output_dir / "original" / f"{file_stem}.png", cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
                save_image(output_dir / "ground_truth" / f"{file_stem}.png", gt * 255)
                save_image(output_dir / "prediction" / f"{file_stem}.png", pred * 255)
                save_image(output_dir / "overlay" / f"{file_stem}.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                save_error_map(gt, pred, output_dir / "error_map" / f"{file_stem}.png")
                save_image(output_dir / "probability" / f"{file_stem}.png", probability_map)

                image_metrics = metrics(pred, gt)
                image_metrics.update({"Image": file_stem, "Threshold": threshold})
                per_image_rows.append(image_metrics)
                for key in all_metrics:
                    all_metrics[key] += image_metrics[key]

            per_image_df = pd.DataFrame(per_image_rows)
            per_image_df.to_csv(output_dir / "per_image_metrics.csv", index=False)

            test_precision = all_metrics["TP"] / (all_metrics["TP"] + all_metrics["FP"]) if all_metrics["TP"] + all_metrics["FP"] else 0.0
            test_recall = all_metrics["TP"] / (all_metrics["TP"] + all_metrics["FN"]) if all_metrics["TP"] + all_metrics["FN"] else 0.0
            test_f1 = 2 * test_precision * test_recall / (test_precision + test_recall) if test_precision + test_recall else 0.0
            test_iou = all_metrics["TP"] / (all_metrics["TP"] + all_metrics["FP"] + all_metrics["FN"]) if all_metrics["TP"] + all_metrics["FP"] + all_metrics["FN"] else 0.0

            summary_rows.append({
                "Dataset": dataset_name,
                "Model": model_name,
                "Test_Images": 15,
                "Validation_Selected_Threshold": threshold,
                "Test_F1_Dice": test_f1,
                "Test_IoU": test_iou,
                "Test_Precision": test_precision,
                "Test_Recall": test_recall,
                "TP": all_metrics["TP"],
                "FP": all_metrics["FP"],
                "FN": all_metrics["FN"],
            })
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_ROOT / "FINAL_LOCKED_15_RESULTS.csv", index=False)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
