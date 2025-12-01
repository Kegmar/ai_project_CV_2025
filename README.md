This project performs object detection on custom puzzle-piece images using YOLOv8.  
The pipeline includes:

- Automatic dataset labeling (color-based segmentation)
- YOLO training (single class: piece)
- Batch inference on photos
- Dataset structure aligned with CV best practices

## How to Run

### 1. Put your raw photos here:
data_raw/

### 2. Generate YOLO labels:
python scripts/auto_label.py

### 3. Train the model:
python scripts/train.py

### 4. Run inference on your images:
python scripts/detect_on_folder.py
