import cv2
import os
import random
import shutil

# Root folders
RAW_ROOT = "data_raw"
OUTPUT_DATASET = "dataset"

IMG_TRAIN = "dataset/images/train"
IMG_VAL = "dataset/images/val"
LBL_TRAIN = "dataset/labels/train"
LBL_VAL = "dataset/labels/val"

VAL_RATIO = 0.2

# Create dataset folders
for p in [IMG_TRAIN, IMG_VAL, LBL_TRAIN, LBL_VAL]:
    os.makedirs(p, exist_ok=True)

# Collect all images recursively
all_images = []
for root, dirs, files in os.walk(RAW_ROOT):
    for f in files:
        if f.lower().endswith((".jpg", ".png", ".jpeg")):
            all_images.append(os.path.join(root, f))

random.shuffle(all_images)
val_count = int(len(all_images) * VAL_RATIO)

val_set = set(all_images[:val_count])
train_set = set(all_images[val_count:])

def process_image(img_path, img_out_path, lbl_out_path):
    # Read image
    img = cv2.imread(img_path)
    h, w, _ = img.shape

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # red ranges
    lower1 = (0, 80, 80)
    upper1 = (10, 255, 255)

    lower2 = (170, 80, 80)
    upper2 = (180, 255, 255)

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    labels = []

    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)

        if bw * bh < 500:  # skip noise
            continue

        xc = (x + bw/2) / w
        yc = (y + bh/2) / h
        ww = bw / w
        hh = bh / h

        labels.append(f"0 {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")

    # Save label
    out_label_path = os.path.join(lbl_out_path, os.path.basename(img_path).replace(".jpg", ".txt"))
    with open(out_label_path, "w") as f:
        f.write("\n".join(labels))

    # Copy image
    shutil.copy(img_path, os.path.join(img_out_path, os.path.basename(img_path)))


for img in all_images:
    if img in val_set:
        process_image(img, IMG_VAL, LBL_VAL)
    else:
        process_image(img, IMG_TRAIN, LBL_TRAIN)

print("Auto-labeling complete!")
