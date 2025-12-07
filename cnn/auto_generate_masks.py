"""
V6 Mask Generator - keeps ONLY puzzle-piece edges using:
1) Canny on board-only region
2) Frame removal
3) Skeletonization
4) Length-based filtering of skeleton components

Run:
    python auto_generate_masks_v6.py --overwrite
"""

import argparse
from pathlib import Path
import numpy as np
import cv2
from skimage.morphology import skeletonize


# ---------------- Core mask generation ---------------- #

def generate_mask_v6(img_path: Path) -> np.ndarray | None:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] Cannot read {img_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # --- remove noise while keeping edges ---
    smooth = cv2.bilateralFilter(gray, 9, 70, 70)

    # --- board segmentation (Otsu) ---
    _, thr = cv2.threshold(smooth, 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr_inv = cv2.bitwise_not(thr)

    contours, _ = cv2.findContours(thr_inv, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"[WARN] No board contour found in {img_path.name}")
        return None

    board = max(contours, key=cv2.contourArea)

    board_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(board_mask, [board], -1, 255, cv2.FILLED)

    # keep only board interior
    inside = cv2.bitwise_and(smooth, smooth, mask=board_mask)

    # --- edge detection ---
    edges = cv2.Canny(inside, 40, 120)

    # --- remove frame edges ---
    frame = np.zeros_like(edges)
    cv2.drawContours(frame, [board], -1, 255, thickness=12)
    edges = cv2.bitwise_and(edges, cv2.bitwise_not(frame))

    # --- skeletonization ---
    edges_bin = (edges > 0).astype(np.uint8)
    skel = skeletonize(edges_bin).astype(np.uint8) * 255

    # --- component filtering by skeleton length ---
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        skel, connectivity=8)

    clean = np.zeros_like(skel)

    MIN_SKELETON_LEN = 30   # <-- tune here: longer = cleaner

    for lbl in range(1, num_labels):
        comp = (labels == lbl).astype(np.uint8)
        length = comp.sum()

        if length >= MIN_SKELETON_LEN:
            clean[labels == lbl] = 255

    # --- dilate back to thicker edge for CNN training ---
    clean = cv2.dilate(clean, np.ones((3, 3), np.uint8), iterations=2)

    return clean


# ---------------- Batch processing ---------------- #

def process_split(root, split, image_folder, overwrite):
    img_dir = root / split / image_folder
    mask_dir = root / split / "masks"
    mask_dir.mkdir(exist_ok=True, parents=True)

    imgs = sorted([p for p in img_dir.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])

    for img_path in imgs:
        out_path = mask_dir / f"{img_path.stem}.png"

        if out_path.exists() and not overwrite:
            print(f"[SKIP] {out_path.name}")
            continue

        print(f"[PROC] {img_path.name} -> {out_path.name}")
        mask = generate_mask_v6(img_path)
        if mask is None:
            continue

        cv2.imwrite(str(out_path), mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--image-folder", default="images")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(args.data_dir)

    for split in args.splits:
        process_split(root, split, args.image_folder, args.overwrite)


if __name__ == "__main__":
    main()
