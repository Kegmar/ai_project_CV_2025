from pathlib import Path
from PIL import Image

DATA_ROOT = Path("data")
SPLITS = ["train", "val"]

def main():
    for split in SPLITS:
        in_dir = DATA_ROOT / split / "images"
        out_dir = DATA_ROOT / split / "images_bw"

        if not in_dir.exists():
            print(f"Skip missing folder: {in_dir}")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(in_dir.glob("*.jpg")):
            print(f"Converting {img_path}")
            img = Image.open(img_path).convert("L")   # "L" = grayscale
            out_path = out_dir / (img_path.stem + ".png")
            img.save(out_path)

    print("Done converting to black & white.")

if __name__ == "__main__":
    main()
