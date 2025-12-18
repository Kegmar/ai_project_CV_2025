from pathlib import Path
import numpy as np
from PIL import Image
import random

IMG_DIR = Path("calib/images") # ignored
OUT = Path("calib/calib_set.npy")  # ignored
N = 400         
SIZE = (640, 640)

paths = []
for ext in ("*.jpg", "*.jpeg"):
    paths += list(IMG_DIR.glob(ext))
if not paths:
    raise SystemExit(f"No images found in {IMG_DIR.resolve()}")

random.shuffle(paths)
paths = paths[:min(N, len(paths))]

arr = np.zeros((len(paths), SIZE[1], SIZE[0], 3), dtype=np.uint8)
for i, p in enumerate(paths):
    im = Image.open(p).convert("RGB").resize(SIZE)
    arr[i] = np.asarray(im, dtype=np.uint8)

OUT.parent.mkdir(parents=True, exist_ok=True)
np.save(OUT, arr)
print("Saved:", OUT.resolve(), arr.shape, arr.dtype)
