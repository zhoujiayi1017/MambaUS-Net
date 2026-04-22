import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLossMulti(nn.Module):
    """
    Multi-class Dice loss computed from softmax probabilities.
    """

    def __init__(self, num_classes: int, smooth: float = 1e-6, ignore_index: int = None):
        """
        Initialize the Dice loss module.

        Args:
            num_classes (int): Number of segmentation classes.
            smooth (float): Smoothing constant to avoid division by zero.
            ignore_index (int, optional): Class index to ignore during loss computation.
        """
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-class Dice loss.

        Args:
            logits (torch.Tensor): Raw model outputs of shape [B, C, H, W].
            target (torch.Tensor): Ground-truth labels of shape [B, H, W].

        Returns:
            torch.Tensor: Mean Dice loss across valid classes.
        """
        prob = F.softmax(logits, dim=1)
        target_onehot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        loss = 0.0
        count = 0

        for c in range(self.num_classes):
            if self.ignore_index is not None and c == self.ignore_index:
                continue

            pred_c = prob[:, c]
            target_c = target_onehot[:, c]

            intersection = (pred_c * target_c).sum(dim=(1, 2))
            denominator = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))

            dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
            loss += (1.0 - dice).mean()
            count += 1

        return loss / max(count, 1)


@torch.no_grad()
def per_class_dice_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    eps: float = 1e-6,
):
    """
    Compute Dice score for each class from raw logits.

    Args:
        logits (torch.Tensor): Raw model outputs of shape [B, C, H, W].
        target (torch.Tensor): Ground-truth labels of shape [B, H, W].
        num_classes (int): Number of segmentation classes.
        eps (float): Small constant for numerical stability.

    Returns:
        list[float]: Dice score for each class.
    """
    preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
    dices = []

    for c in range(num_classes):
        pred_c = (preds == c).float()
        target_c = (target == c).float()

        intersection = (pred_c * target_c).sum()
        denominator = pred_c.sum() + target_c.sum()

        dice = (2.0 * intersection + eps) / (denominator + eps)
        dices.append(dice.item())

    return dices


@torch.no_grad()
def mean_iou_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    eps: float = 1e-6,
):
    """
    Compute per-class IoU and mean IoU from raw logits.

    Args:
        logits (torch.Tensor): Raw model outputs of shape [B, C, H, W].
        target (torch.Tensor): Ground-truth labels of shape [B, H, W].
        num_classes (int): Number of segmentation classes.
        eps (float): Small constant for numerical stability.

    Returns:
        tuple[list[float | None], float]:
            - Per-class IoU values
            - Mean IoU over valid classes
    """
    preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
    ious = []

    for c in range(num_classes):
        pred_c = (preds == c).float()
        target_c = (target == c).float()

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum() - intersection

        if union == 0:
            iou_value = None
        else:
            iou = (intersection + eps) / (union + eps)
            iou_value = iou.item()

        ious.append(iou_value)

    valid_ious = [v for v in ious if v is not None]
    mean_iou = float(np.mean(valid_ious)) if len(valid_ious) > 0 else 0.0

    return ious, mean_iou