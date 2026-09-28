from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

from pore_models import create_model

SEED = 42
IMAGE_SIZE = 512
BATCH_SIZE = 4
EPOCHS = 20 
LEARNING_RATE = 1e-4
N_SPLITS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_DIR = Path("FINAL_173")
MODELS = ["ResNet34_UNet_Baseline", "ResNet34_UNet_scSE", "Custom_Attention_UNet"]

def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def find_pairs(dataset_root: Path, split: str) -> list[tuple[str, str]]:
    image_dir = dataset_root / split / "images"
    mask_dir = dataset_root / split / "masks"
    images = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".png"})
    return [(str(img), str(mask_dir / f"{img.stem}.png")) for img in images if (mask_dir / f"{img.stem}.png").exists()]

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
        img_path, mask_path = self.pairs[index]
        image = cv2.cvtColor(cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask = (cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
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
def evaluate_fold(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    tp = fp = fn = 0
    for images, masks in loader:
        preds = (torch.sigmoid(model(images.to(DEVICE))).cpu().numpy() > 0.50)
        truths = masks.numpy()
        tp += int(np.sum((truths == 1) & preds))
        fp += int(np.sum((truths == 0) & preds))
        fn += int(np.sum((truths == 1) & ~preds))
    return (2 * tp) / (2 * tp + fp + fn + 1e-7)

def main() -> None:
    seed_everything(SEED)
    print(f"--- Starting {N_SPLITS}-Fold Cross Validation on {DATASET_DIR.name} ---")
    
    train_pool = find_pairs(DATASET_DIR, "train")
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    criterion = BCEDiceLoss()
    
    final_results = []

    for model_name in MODELS:
        print(f"\n[Evaluating Architecture: {model_name}]")
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_pool), start=1):
            train_pairs = [train_pool[i] for i in train_idx]
            val_pairs = [train_pool[i] for i in val_idx]
            
            train_loader = DataLoader(PoreDataset(train_pairs), batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(PoreDataset(val_pairs), batch_size=BATCH_SIZE, shuffle=False)
            
            model = create_model(model_name, pretrained=True).to(DEVICE)
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
            
            for epoch in range(EPOCHS):
                model.train()
                for images, masks in train_loader:
                    optimizer.zero_grad()
                    loss = criterion(model(images.to(DEVICE)), masks.to(DEVICE))
                    loss.backward()
                    optimizer.step()
                    
            fold_f1 = evaluate_fold(model, val_loader)
            fold_scores.append(fold_f1)
            print(f"  -> Fold {fold}: Validation F1 = {fold_f1:.4f}")
            
            del model, optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        mean_f1 = np.mean(fold_scores)
        std_f1 = np.std(fold_scores)
        print(f"Result for {model_name}: {mean_f1:.4f} ± {std_f1:.4f}")
        
        final_results.append({
            "Architecture": model_name,
            "Mean_F1": round(mean_f1, 4),
            "Std_F1": round(std_f1, 4)
        })

    pd.DataFrame(final_results).to_csv("KFold_Results.csv", index=False)
    print("\n[SUCCESS] K-Fold Cross Validation completed.")

if __name__ == "__main__":
    main()
