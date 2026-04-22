import os
import random
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageMaskDataset(Dataset):
    """
    Dataset for paired image-mask segmentation data.

    Expected directory structure:
        data_path/
        ├── images/
        └── masks/
    """

    def __init__(
        self,
        data_path: str,
        transform: Optional[callable] = None,
        augment: bool = True,
        image_size: Tuple[int, int] = (256, 256),
    ):
        """
        Initialize the dataset.

        Args:
            data_path (str): Root directory containing 'images' and 'masks'.
            transform (callable, optional): Optional transform applied to the image tensor.
            augment (bool): Whether to apply data augmentation.
            image_size (tuple): Target image size as (width, height).
        """
        self.data_path = Path(data_path)
        self.image_dir = self.data_path / "images"
        self.mask_dir = self.data_path / "masks"
        self.transform = transform
        self.augment = augment
        self.image_size = image_size

        self._check_directories()

        self.image_names = sorted(
            [
                f.name for f in self.image_dir.iterdir()
                if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
            ]
        )

        if len(self.image_names) == 0:
            raise RuntimeError(f"No images found in: {self.image_dir}")

        missing_masks = [name for name in self.image_names if not (self.mask_dir / name).exists()]
        if missing_masks:
            preview = ", ".join(missing_masks[:5])
            raise FileNotFoundError(
                f"Missing mask files for {len(missing_masks)} image(s). Examples: {preview}"
            )

    def _check_directories(self) -> None:
        """Check whether the required dataset folders exist."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.data_path}")
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory does not exist: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory does not exist: {self.mask_dir}")

    def __len__(self) -> int:
        return len(self.image_names)

    def _apply_geometric_augment(
        self,
        img: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply geometric augmentations to image-mask pairs.

        Args:
            img (np.ndarray): Image array in HWC format, float32, range [0, 1].
            mask (np.ndarray): Mask array in HW format, int64.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Augmented image and mask.
        """
        h, w = mask.shape

        # Horizontal flip
        if random.random() < 0.5:
            img = np.flip(img, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()

        # Rotation, scaling, and translation
        if random.random() < 0.8:
            angle = random.uniform(-25.0, 25.0)
            scale = random.uniform(0.6, 1.4)

            max_tx = 0.3 * w
            max_ty = 0.3 * h
            tx = random.uniform(-max_tx, max_tx)
            ty = random.uniform(-max_ty, max_ty)

            center = (w / 2.0, h / 2.0)
            affine_matrix = cv2.getRotationMatrix2D(center, angle, scale)
            affine_matrix[0, 2] += tx
            affine_matrix[1, 2] += ty

            img = cv2.warpAffine(
                img,
                affine_matrix,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            mask = cv2.warpAffine(
                mask.astype(np.float32),
                affine_matrix,
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_REFLECT_101,
            ).astype(np.int64)

        # Elastic deformation
        if random.random() < 0.3:
            strength = random.uniform(0.0, 0.3)
            if strength > 0.0:
                alpha = strength * 40.0
                sigma = 6.0

                dx = np.random.randn(h, w).astype(np.float32)
                dy = np.random.randn(h, w).astype(np.float32)
                dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
                dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha

                x, y = np.meshgrid(np.arange(w), np.arange(h))
                map_x = (x + dx).astype(np.float32)
                map_y = (y + dy).astype(np.float32)

                img = cv2.remap(
                    img,
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
                mask = cv2.remap(
                    mask.astype(np.float32),
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_REFLECT_101,
                ).astype(np.int64)

        return img, mask

    def _apply_intensity_augment(self, img: np.ndarray) -> np.ndarray:
        """
        Apply intensity augmentations to the image only.

        Args:
            img (np.ndarray): Image array in HWC format, float32, range [0, 1].

        Returns:
            np.ndarray: Augmented image.
        """
        # Brightness shift
        if random.random() < 0.5:
            shift = random.uniform(-20.0, 20.0) / 255.0
            img = np.clip(img + shift, 0.0, 1.0)

        # Contrast adjustment
        if random.random() < 0.5:
            contrast = random.uniform(0.4, 1.6)
            img = np.clip((img - 0.5) * contrast + 0.5, 0.0, 1.0)

        # Gamma correction
        if random.random() < 0.3:
            gamma = random.uniform(0.5, 1.5)
            img = np.clip(np.power(img, 1.0 / max(gamma, 1e-6)), 0.0, 1.0)

        # Gaussian blur
        if random.random() < 0.2:
            img = cv2.GaussianBlur(img, (5, 5), 0)

        # Motion blur
        if random.random() < 0.15:
            ksize = int(random.uniform(3, 13))
            if ksize % 2 == 0:
                ksize += 1
            kernel = np.zeros((1, ksize), np.float32)
            kernel[0, :] = 1.0 / ksize
            img = cv2.filter2D(img, -1, kernel)

        # Gaussian noise
        if random.random() < 0.3:
            sigma = random.uniform(5.0, 20.0) / 255.0
            noise = np.random.randn(*img.shape).astype(np.float32) * sigma
            img = np.clip(img + noise, 0.0, 1.0)

        # Speckle noise
        if random.random() < 0.3:
            sigma_s = random.uniform(0.02, 0.07)
            noise = np.random.randn(*img.shape).astype(np.float32) * sigma_s
            img = np.clip(img + img * noise, 0.0, 1.0)

        return img

    def __getitem__(self, idx: int):
        """
        Load one image-mask pair.

        Args:
            idx (int): Sample index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Image tensor and mask tensor.
        """
        img_name = self.image_names[idx]
        image_path = self.image_dir / img_name
        mask_path = self.mask_dir / img_name

        img = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img = img.resize(self.image_size, Image.Resampling.BILINEAR)
        mask = mask.resize(self.image_size, Image.Resampling.NEAREST)

        img = np.asarray(img, dtype=np.float32) / 255.0
        mask = np.asarray(mask, dtype=np.int64)

        if self.augment:
            img, mask = self._apply_geometric_augment(img, mask)
            img = self._apply_intensity_augment(img)

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))

        img_tensor = torch.from_numpy(img).float()
        mask_tensor = torch.from_numpy(mask).long()

        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        return img_tensor, mask_tensor