import os, glob
import cv2
import numpy as np

INPUT_SIZE = 640
CROP_LEFT  = 370
CROP_RIGHT = 1550

def letterbox(im, new_shape=(640,640), color=(114,114,114)):
    h,w = im.shape[:2]
    nh,nw = new_shape
    r = min(nh/h, nw/w)
    rw,rh = int(round(w*r)), int(round(h*r))
    dw,dh = (nw-rw)/2, (nh-rh)/2
    im = cv2.resize(im,(rw,rh),interpolation=cv2.INTER_LINEAR)
    top,bottom = int(round(dh-0.1)), int(round(dh+0.1))
    left,right = int(round(dw-0.1)), int(round(dw+0.1))
    return cv2.copyMakeBorder(im, top,bottom,left,right, cv2.BORDER_CONSTANT, value=color)

img_dir = "calib/images"
paths = sorted(glob.glob(os.path.join(img_dir, "*.*")))
paths = [p for p in paths if p.lower().endswith((".jpg",".jpeg",".png"))]

if not paths:
    raise SystemExit("No images found in calib/images (jpg/png)")

N = len(paths)
arr = np.zeros((N, INPUT_SIZE, INPUT_SIZE, 3), np.float32)

bad = 0
for i,p in enumerate(paths):
    bgr = cv2.imread(p)
    if bgr is None:
        bad += 1
        continue

    bgr = bgr[:, CROP_LEFT:CROP_RIGHT]

    # letterbox to 640x640
    bgr = letterbox(bgr, (INPUT_SIZE, INPUT_SIZE))

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    arr[i] = rgb.astype(np.float32) / 255.0

np.save("calib/calib_set.npy", arr)
print("saved:", arr.shape, arr.dtype, "min/max:", float(arr.min()), float(arr.max()))
if bad:
    print("warning: unreadable images:", bad)
