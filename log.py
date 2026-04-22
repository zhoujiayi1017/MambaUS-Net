from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


class MetricLogger:
    """
    Utility class for logging, plotting, and saving training metrics.
    """

    def __init__(self, save_dir: str = "logs"):
        """
        Initialize the metric logger.

        Args:
            save_dir (str): Directory for saving plots and metric files.
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = defaultdict(list)

    def log(self, key: str, value: float) -> None:
        """
        Append a metric value.

        Args:
            key (str): Metric name.
            value (float): Metric value.
        """
        self.metrics[key].append(value)

    def plot_metrics(self) -> None:
        """
        Plot and save training curves.
        """
        if "train_loss" not in self.metrics or len(self.metrics["train_loss"]) == 0:
            return

        epochs = range(1, len(self.metrics["train_loss"]) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Loss curves
        axes[0].plot(epochs, self.metrics["train_loss"], label="Train Loss")
        if "val_loss" in self.metrics:
            axes[0].plot(epochs, self.metrics["val_loss"], label="Val Loss")
        axes[0].set_title("Loss Curves")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Dice / IoU curves
        if "train_dice" in self.metrics:
            axes[1].plot(epochs, self.metrics["train_dice"], label="Train Dice")
        if "val_dice" in self.metrics:
            axes[1].plot(epochs, self.metrics["val_dice"], label="Val Dice")
        if "train_iou" in self.metrics:
            axes[1].plot(epochs, self.metrics["train_iou"], label="Train IoU", linestyle="--")
        if "val_iou" in self.metrics:
            axes[1].plot(epochs, self.metrics["val_iou"], label="Val IoU", linestyle="--")

        axes[1].set_title("Dice and IoU Curves")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Score")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(self.save_dir / "training_curves.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    def save_to_file(self) -> None:
        """
        Save recorded metrics to a CSV file.
        """
        df = pd.DataFrame({k: pd.Series(v) for k, v in self.metrics.items()})
        df.to_csv(self.save_dir / "metrics.csv", index=False)