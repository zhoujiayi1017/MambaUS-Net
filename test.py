import os
import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from models.mambaus_net import MambaSeg


def get_device() -> torch.device:
    """Return the available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(
    ckpt_path: str,
    in_channels: int,
    num_classes: int,
    device: torch.device,
) -> torch.nn.Module:
    """
    Build the model and load checkpoint.

    Args:
        ckpt_path (str): Path to the checkpoint file.
        in_channels (int): Number of input channels.
        num_classes (int): Number of segmentation classes.
        device (torch.device): Target device.

    Returns:
        torch.nn.Module: Loaded model in evaluation mode.
    """
    model = MambaSeg(in_channels=in_channels, num_classes=num_classes).to(device)

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)
    state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state

    clean_state_dict = {}
    for k, v in state_dict.items():
        clean_key = k[7:] if k.startswith("module.") else k
        clean_state_dict[clean_key] = v

    model.load_state_dict(clean_state_dict, strict=True)
    model.eval()
    return model


def preprocess_image(
    img_path: Path,
    image_size: Tuple[int, int],
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Preprocess one image for inference.

    Args:
        img_path (Path): Path to the input image.
        image_size (Tuple[int, int]): Target size as (width, height).

    Returns:
        Tuple[torch.Tensor, Tuple[int, int]]:
            - Input tensor of shape [1, C, H, W]
            - Original image size as (width, height)
    """
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size

    img = img.resize(image_size, Image.Resampling.BILINEAR)
    img = np.asarray(img, dtype=np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))

    img_tensor = torch.from_numpy(img).float().unsqueeze(0)
    return img_tensor, (orig_w, orig_h)


def save_prediction(
    pred_mask: np.ndarray,
    save_path: Path,
    original_size: Tuple[int, int],
) -> None:
    """
    Resize prediction back to the original image size and save it.

    Args:
        pred_mask (np.ndarray): Predicted mask of shape [H, W].
        save_path (Path): Output path.
        original_size (Tuple[int, int]): Original image size as (width, height).
    """
    pred_mask = cv2.resize(
        pred_mask,
        original_size,
        interpolation=cv2.INTER_NEAREST,
    )
    cv2.imwrite(str(save_path), pred_mask)


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    image_dir: str,
    save_dir: str,
    image_size: Tuple[int, int],
    device: torch.device,
) -> None:
    """
    Run inference on all images in a directory and save predicted masks.

    Args:
        model (torch.nn.Module): Trained segmentation model.
        image_dir (str): Directory containing test images.
        save_dir (str): Directory to save predicted masks.
        image_size (Tuple[int, int]): Input image size as (width, height).
        device (torch.device): Target device.
    """
    image_dir = Path(image_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_names = sorted(
        [
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"))
            and not f.startswith("._")
        ]
    )

    if len(image_names) == 0:
        print(f"[WARN] No images found in: {image_dir}")
        return

    print(f"[INFO] Found {len(image_names)} images in: {image_dir}")
    print(f"[INFO] Saving predictions to: {save_dir}")

    for i, img_name in enumerate(image_names, start=1):
        img_path = image_dir / img_name
        img_tensor, original_size = preprocess_image(img_path, image_size)
        img_tensor = img_tensor.to(device, non_blocking=True)

        logits = model(img_tensor)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        save_path = save_dir / f"{Path(img_name).stem}.png"
        save_prediction(pred, save_path, original_size)

        if i % 20 == 0 or i == len(image_names):
            print(f"[INFO] Processed {i}/{len(image_names)} images.")

    print("[INFO] Inference completed.")


def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for MambaUS-Net.")

    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing test images.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save predicted masks.")

    parser.add_argument("--in_channels", type=int, default=3, help="Number of input image channels.")
    parser.add_argument("--num_classes", type=int, default=3, help="Number of segmentation classes.")
    parser.add_argument("--image_size", type=int, default=256, help="Input image size.")

    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    image_size = (args.image_size, args.image_size)

    model = build_model(
        ckpt_path=args.ckpt_path,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        device=device,
    )

    run_inference(
        model=model,
        image_dir=args.image_dir,
        save_dir=args.save_dir,
        image_size=image_size,
        device=device,
    )


if __name__ == "__main__":
    main()