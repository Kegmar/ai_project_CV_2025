This project performs object detection on custom puzzle-piece images using YOLOv8.  
The pipeline includes:

- Automatic dataset labeling (color-based segmentation)
- YOLO training (single class: piece)
- Batch inference on photos
- Dataset structure aligned with CV best practices

## Project Structure
AI_PROJECT_CV_2025/
│
├── data_raw/                 # Raw photos (input images)
│   ├── photos_puzzle1/
│   ├── photos_puzzle2/
│   └── photos_puzzle3/
│
├── dataset/                  # Auto-generated YOLO dataset
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
│
├── models/                   # Trained YOLO weights (best.pt)
│
├── results/                  # Detection outputs (auto-generated)
│
├── scripts/
│   ├── auto_label.py         # Automatic dataset labeling
│   ├── train.py              # YOLO training script
│   └── detect_on_folder.py   # Batch inference
│
├── dataset.yaml              # YOLO dataset config
└── README.md

## How to Run

### 1. Put your raw photos here:
data_raw/

### 2. Generate YOLO labels:
python scripts/auto_label.py

### 3. Train the model:
python scripts/train.py

### 4. Run inference on your images:
python scripts/detect_on_folder.py