# predict_unet_gray.py

import argparse
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2

from unet_gray import UNetGray


def load_model(model_path, device):
    model = UNetGray(in_channels=1, out_channels=1)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = load_model(args.model, device)
    print(f"Loaded model from {args.model}")

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # SAME preprocessing as BasicGrayTransform
    resize = T.Resize((512, 512))
    to_tensor = T.ToTensor()

    img_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        img_paths.extend(sorted(input_dir.glob(ext)))

    if not img_paths:
        print(f"No images found in {input_dir}")
        return

    for img_path in img_paths:
        print("Predicting for:", img_path.name)

        # load grayscale image (BW)
        img = Image.open(img_path).convert("L")
        orig_w, orig_h = img.size

        x = resize(img)
        x = to_tensor(x)         # [1, H, W], values in [0,1]
        x = x.unsqueeze(0).to(device)  # [1,1,H,W]

        with torch.no_grad():
            logits = model(x)          # [1,1,H,W]
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()  # [H,W]

        # debug: print min/max so we know it's not all zeros
        print(f"  prob min={probs.min():.3f}, max={probs.max():.3f}")

        # threshold at 0.3
        mask_small = (probs > 0.3).astype(np.uint8) * 255

        # resize back to original size
        mask = cv2.resize(
            mask_small,
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST,
        )

        # save binary mask
        mask_path = out_dir / f"{img_path.stem}_edges.png"
        cv2.imwrite(str(mask_path), mask)

        # optional: save overlay on top of original COLOR image (if exists)
        if args.overlay:
            # try to load matching color image from same folder
            color_img = cv2.imread(str(img_path))
            if color_img is not None:
                overlay = color_img.copy()
                overlay[mask > 0] = (0, 255, 0)  # green edges
                vis = cv2.addWeighted(color_img, 0.7, overlay, 0.3, 0)
                overlay_path = out_dir / f"{img_path.stem}_overlay.png"
                cv2.imwrite(str(overlay_path), vis)

    print("Done, predictions saved to", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="models_cnn/unet_puzzle_gray.pth",
        help="Path to trained model weights",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="cnn_train_dataset/val/images_bw",
        help="Folder with BW images to segment",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="predictions",
        help="Where to save prediction masks",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Also save overlays of edges on top of original images",
    )
    args = parser.parse_args()
    main(args)
