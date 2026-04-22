import os
import math
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.data import DataLoader

from dataset import ImageMaskDataset
from metrics import (
    DiceLossMulti,
    per_class_dice_from_logits,
    mean_iou_from_logits,
)
from models.mambaus_net import MambaSeg
from log import MetricLogger


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cudnn.deterministic = True
    cudnn.benchmark = False


def get_device() -> torch.device:
    """Return the available device."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_dataloaders(args):
    """Build training and validation dataloaders."""
    train_dataset = ImageMaskDataset(
        args.train_path,
        transform=None,
        augment=True,
        image_size=(args.image_size, args.image_size),
    )
    val_dataset = ImageMaskDataset(
        args.val_path,
        transform=None,
        augment=False,
        image_size=(args.image_size, args.image_size),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


def build_model(args, device: torch.device) -> nn.Module:
    """Build model and optionally load checkpoint."""
    model = MambaSeg(
        in_channels=args.in_channels,
        num_classes=args.num_classes,
    ).to(device)

    if args.resume_ckpt:
        state = torch.load(args.resume_ckpt, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[INFO] Loaded checkpoint from: {args.resume_ckpt}")
        print(f"[INFO] Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    else:
        print("[INFO] Training from scratch.")

    return model


def build_optimizer(args, model: nn.Module):
    """Build optimizer."""
    return AdamW(
        model.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay,
    )


def build_scheduler(args, optimizer):
    """Build warmup + cosine annealing scheduler."""
    warmup = LinearLR(
        optimizer,
        start_factor=args.warmup_start_factor,
        end_factor=1.0,
        total_iters=args.warmup_epochs,
    )

    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs - args.warmup_epochs),
        eta_min=args.min_lr,
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[args.warmup_epochs],
    )
    return scheduler


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer,
    device: torch.device,
    num_classes: int,
    ce_loss,
    dice_loss,
    ce_w: float,
    dice_w: float,
):
    """Run one training epoch."""
    model.train()

    running_loss = 0.0
    sum_dices = np.zeros(num_classes, dtype=np.float64)
    sum_counts = np.zeros(num_classes, dtype=np.int64)
    sum_iou = 0.0
    n_batches = 0

    for imgs, masks in dataloader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(imgs)

        loss_ce = ce_loss(logits, masks)
        loss_dice = dice_loss(logits, masks)
        loss = ce_w * loss_ce + dice_w * loss_dice

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

        dices = per_class_dice_from_logits(logits, masks, num_classes)
        for i, d in enumerate(dices):
            if not math.isnan(d):
                sum_dices[i] += d
                sum_counts[i] += 1

        _, mean_iou = mean_iou_from_logits(logits, masks, num_classes)
        sum_iou += mean_iou
        n_batches += 1

    epoch_loss = running_loss / len(dataloader.dataset)
    avg_dice_per_class = [
        (sum_dices[i] / sum_counts[i]) if sum_counts[i] > 0 else 0.0
        for i in range(num_classes)
    ]
    mean_dice = float(np.mean(avg_dice_per_class))
    mean_iou = float(sum_iou / max(1, n_batches))

    return avg_dice_per_class, mean_dice, mean_iou, epoch_loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    ce_loss,
    dice_loss,
    ce_w: float,
    dice_w: float,
):
    """Run validation."""
    model.eval()

    running_loss = 0.0
    sum_dices = np.zeros(num_classes, dtype=np.float64)
    sum_counts = np.zeros(num_classes, dtype=np.int64)
    sum_iou = 0.0
    n_batches = 0

    for imgs, masks in dataloader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(imgs)

        loss_ce = ce_loss(logits, masks)
        loss_dice = dice_loss(logits, masks)
        loss = ce_w * loss_ce + dice_w * loss_dice

        running_loss += loss.item() * imgs.size(0)

        dices = per_class_dice_from_logits(logits, masks, num_classes)
        for i, d in enumerate(dices):
            if not math.isnan(d):
                sum_dices[i] += d
                sum_counts[i] += 1

        _, mean_iou = mean_iou_from_logits(logits, masks, num_classes)
        sum_iou += mean_iou
        n_batches += 1

    epoch_loss = running_loss / len(dataloader.dataset)
    avg_dice_per_class = [
        (sum_dices[i] / sum_counts[i]) if sum_counts[i] > 0 else 0.0
        for i in range(num_classes)
    ]
    mean_dice = float(np.mean(avg_dice_per_class))
    mean_iou = float(sum_iou / max(1, n_batches))

    return avg_dice_per_class, mean_dice, mean_iou, epoch_loss


def save_checkpoint(model: nn.Module, save_dir: Path, filename: str) -> None:
    """Save model checkpoint."""
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / filename)


def train(args):
    """Main training function."""
    device = get_device()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logger = MetricLogger(save_dir=str(save_dir / "logs"))

    train_loader, val_loader = build_dataloaders(args)
    model = build_model(args, device)

    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLossMulti(num_classes=args.num_classes)

    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)

    best_val_dice = 0.0
    eps = 1e-4

    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n[Epoch {epoch}/{args.epochs}] lr={current_lr:.6e}")

        train_dice_cls, train_mean_dice, train_mean_iou, train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            num_classes=args.num_classes,
            ce_loss=ce_loss,
            dice_loss=dice_loss,
            ce_w=args.ce_w,
            dice_w=args.dice_w,
        )

        val_dice_cls, val_mean_dice, val_mean_iou, val_loss = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            num_classes=args.num_classes,
            ce_loss=ce_loss,
            dice_loss=dice_loss,
            ce_w=args.ce_w,
            dice_w=args.dice_w,
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train mDice: {train_mean_dice:.4f} | "
            f"Train mIoU: {train_mean_iou:.4f}"
        )
        print(
            f"Val   Loss: {val_loss:.4f} | "
            f"Val   mDice: {val_mean_dice:.4f} | "
            f"Val   mIoU: {val_mean_iou:.4f}"
        )

        for i, d in enumerate(val_dice_cls):
            print(f"  Val Dice (class {i}): {d:.4f}")

        logger.log("train_loss", train_loss)
        logger.log("val_loss", val_loss)
        logger.log("train_dice", train_mean_dice)
        logger.log("val_dice", val_mean_dice)
        logger.log("train_iou", train_mean_iou)
        logger.log("val_iou", val_mean_iou)
        logger.plot_metrics()

        if val_mean_dice > best_val_dice + eps:
            best_val_dice = val_mean_dice
            ckpt_name = f"best_mDice_{val_mean_dice:.4f}.pth"
            save_checkpoint(model, save_dir, ckpt_name)
            print(f"[INFO] New best checkpoint saved: {ckpt_name}")

        scheduler.step()

    logger.save_to_file()
    print(f"\n[INFO] Training finished. Best val mDice: {best_val_dice:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for MambaUS-Net.")

    parser.add_argument("--image_size", type=int, default=256, help="Input image size.")
    parser.add_argument("--train_path", type=str, required=True, help="Path to the training dataset.")
    parser.add_argument("--val_path", type=str, required=True, help="Path to the validation dataset.")
    parser.add_argument("--save_dir", type=str, default="./outputs/default", help="Directory to save checkpoints and logs.")

    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--base_lr", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate for cosine annealing.")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for AdamW.")

    parser.add_argument("--warmup_epochs", type=int, default=10, help="Number of warmup epochs.")
    parser.add_argument("--warmup_start_factor", type=float, default=0.1, help="Starting LR factor for warmup.")

    parser.add_argument("--in_channels", type=int, default=3, help="Number of input image channels.")
    parser.add_argument("--num_classes", type=int, default=3, help="Number of segmentation classes.")

    parser.add_argument("--ce_w", type=float, default=0.5, help="Weight for CrossEntropy loss.")
    parser.add_argument("--dice_w", type=float, default=0.5, help="Weight for Dice loss.")

    parser.add_argument("--resume_ckpt", type=str, default=None, help="Path to a checkpoint for finetuning.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    train(args)