from pathlib import Path
from typing import Optional, Callable

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class PuzzleEdgesDatasetGray(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        image_folder_name: str = "images_bw",
    ):
        """
        root_dir /
            train /
                images_bw /
                masks /
            val /
                images_bw /
                masks /
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        self.img_dir = self.root_dir / split / image_folder_name
        self.mask_dir = self.root_dir / split / "masks"

        self.image_paths = sorted(self.img_dir.glob("*.png")) + \
                           sorted(self.img_dir.glob("*.jpg"))
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {self.img_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        stem = img_path.stem
        mask_path = self.mask_dir / f"{stem}.png"

        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for {img_path.name}: {mask_path}")

        # grayscale image & mask
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.transform is not None:
            image, mask = self.transform(image, mask)
        else:
            # default basic transform: resize + ToTensor
            resize = T.Resize((512, 512))
            image = resize(image)
            mask = resize(mask)

            to_tensor = T.ToTensor()
            image = to_tensor(image)   # [1, H, W]
            mask = to_tensor(mask)     # [1, H, W]

            mask = (mask > 0.5).float()  # binary 0 / 1

        return image, mask


class BasicGrayTransform:
    def __init__(self, size=(512, 512)):
        self.resize = T.Resize(size)
        self.to_tensor = T.ToTensor()

    def __call__(self, image, mask):
        image = self.resize(image)
        mask = self.resize(mask)

        image = self.to_tensor(image)
        mask = self.to_tensor(mask)
        mask = (mask > 0.5).float()
        return image, mask
