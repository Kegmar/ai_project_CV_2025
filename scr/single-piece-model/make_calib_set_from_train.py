from pathlib import Path
import numpy as np
from PIL import Image
import random
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True, help="path to single_piece_dataset/train")
    ap.add_argument("--out", required=True, help="output .npy file")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--count", type=int, default=512)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    random.seed(args.seed)
    root = Path(args.train_dir)
    imgs = [p for p in root.rglob("*") if p.suffix.lower() in (".jpg",".jpeg")]
    if not imgs:
        raise SystemExit(f"No images found under: {root}")

    random.shuffle(imgs)
    imgs = imgs[:min(args.count, len(imgs))]

    arr = np.empty((len(imgs), args.size, args.size, 3), dtype=np.uint8)
    for i, p in enumerate(imgs):
        im = Image.open(p).convert("RGB").resize((args.size, args.size), Image.BILINEAR)
        arr[i] = np.asarray(im, dtype=np.uint8)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.save(outp, arr)
    print("Saved", outp.resolve(), arr.shape, arr.dtype, "range", int(arr.min()), int(arr.max()))

if __name__ == "__main__":
    main()
