import os
import shutil
from pathlib import Path
import random

RAW_FOLDERS = [
    Path("../data_raw/photos_puzzle1"),
    Path("../data_raw/photos_puzzle2"),
    Path("../data_raw/photos_puzzle3"),
]

OUT_DIR = Path("data")
TRAIN_RATIO = 0.8  # 80% train, 20% val
RANDOM_SEED = 42   # for reproducible splits


def main():
    random.seed(RANDOM_SEED)

    # Collect all images
    img_paths = []
    for folder in RAW_FOLDERS:
        print(f"Scanning {folder} ...")
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            img_paths.extend(sorted(folder.glob(ext)))

    if not img_paths:
        print("No images found in raw folders!")
        return

    print(f"Found {len(img_paths)} total images.")

    # Shuffle before splitting
    random.shuffle(img_paths)

    split_idx = int(len(img_paths) * TRAIN_RATIO)
    train_imgs = img_paths[:split_idx]
    val_imgs = img_paths[split_idx:]

    # Optional: clear old data (comment out if you want to keep)
    for sub in ["train/images", "train/masks", "val/images", "val/masks"]:
        out_sub = OUT_DIR / sub
        if out_sub.exists():
            shutil.rmtree(out_sub)

    # Create directories
    for sub in ["train/images", "train/masks", "val/images", "val/masks"]:
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)

    # Copy train data
    for img in train_imgs:
        shutil.copy(img, OUT_DIR / "train" / "images" / img.name)

    # Copy val data
    for img in val_imgs:
        shutil.copy(img, OUT_DIR / "val" / "images" / img.name)

    print("DONE!")
    print(f"Train images: {len(train_imgs)}")
    print(f"Val images:   {len(val_imgs)}")
    print("Masks will be generated later into data/*/masks by auto_generate_masks.py")


if __name__ == "__main__":
    main()
