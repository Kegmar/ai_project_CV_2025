# ai_project_CV_2025 - CNN MODEL

## 0. Prerequisites

- Python 3.9+ installed
- Git repo already cloned
- You are inside the `cnn/` folder

### Install Python packages (once)

From `cnn/`:

```bash
# (optional) create and activate virtual env
python -m venv .venv
# Windows:
# .venv\Scripts\activate
# macOS / Linux:
# source .venv/bin/activate

# install dependencies
pip install torch torchvision tqdm pillow opencv-python scikit-image
```

## 1. Prepare Data

Your raw images must be in:

../multi_piece_dataset

### 1.1 Split images into train / val

```bash

cd cnn
python split_images.py

```
### 1.2 Generate edge masks

```bash

python auto_generate_masks.py --overwrite

```

### 1.3 Convert images to grayscale (BW)
```bash
python preprocess_to_bw.py
```

## 2. Train the Model (if you want to use committed trained model, skip this step)
```bash
python train_unet_gray.py --epochs 5 --batch-size 2
```

## 3. Use the Trained Model (Prediction)
```bash
python predict_unet_gray.py \
  --model models/unet_puzzle_gray.pth \
  --input-dir cnn_train_dataset/val/images_bw \
  --out-dir predictions \
  --overlay
```
