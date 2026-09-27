from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_FILE = Path("FINAL_BENCHMARK_RESULTS/FINAL_BENCHMARK_RESULTS.csv")
OUTPUT_FILE = Path("FINAL_BENCHMARK_RESULTS/F1_Score_Comparison_Chart.png")


def main() -> None:
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(RESULTS_FILE)

    df = pd.read_csv(RESULTS_FILE)
    model_labels = {
        "ResNet34_UNet_Baseline": "ResNet34-UNet",
        "ResNet34_UNet_scSE": "ResNet34-UNet (scSE)",
        "Custom_Attention_UNet": "Attention-UNet",
    }
    dataset_labels = {
        "FINAL_70": "FINAL_70\n55 development images",
        "FINAL_173": "FINAL_173\n157 development images",
        "FINAL_1775": "FINAL_1775\n1583 development images",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name, group in df.groupby("Model"):
        group = group.copy()
        x = [dataset_labels[d] for d in group["Dataset"]]
        ax.plot(x, group["Test F1_Dice"], marker="o", label=model_labels.get(model_name, model_name))

    ax.set_xlabel("Benchmark dataset")
    ax.set_ylabel("Test F1 / Dice")
    ax.set_ylim(0.30, 0.58)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
