import shutil
from pathlib import Path
import random

SRC_DIR = Path("multi_piece_dataset")  # single folder
OUT_DIR = Path("cnn_train_dataset")

TRAIN_RATIO = 0.8
RANDOM_SEED = 42


def main():
    random.seed(RANDOM_SEED)

    if not SRC_DIR.exists():
        print(f"Source folder not found: {SRC_DIR.resolve()}")
        return

    # Collect all .jpg images that do NOT contain "debug" in the filename
    img_paths = sorted(
        p for p in SRC_DIR.glob("*.jpg")
        if "debug" not in p.name.lower()
    )

    if not img_paths:
        print("No matching .jpg images found!")
        return

    print(f"Found {len(img_paths)} images in {SRC_DIR} (excluding *debug*).")

    random.shuffle(img_paths)
    split_idx = int(len(img_paths) * TRAIN_RATIO)
    train_imgs = img_paths[:split_idx]
    val_imgs = img_paths[split_idx:]

    # Clear old output folders
    for sub in ["train/images", "train/masks", "val/images", "val/masks"]:
        out_sub = OUT_DIR / sub
        if out_sub.exists():
            shutil.rmtree(out_sub)

    # Create directories
    for sub in ["train/images", "train/masks", "val/images", "val/masks"]:
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)

    # Copy images
    for img in train_imgs:
        shutil.copy2(img, OUT_DIR / "train" / "images" / img.name)

    for img in val_imgs:
        shutil.copy2(img, OUT_DIR / "val" / "images" / img.name)

    print("DONE!")
    print(f"Train images: {len(train_imgs)}")
    print(f"Val images:   {len(val_imgs)}")
    print("Masks will be generated later into data/*/masks by auto_generate_masks.py")


if __name__ == "__main__":
    main()
