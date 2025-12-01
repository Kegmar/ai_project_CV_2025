from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="dataset.yaml",
    epochs=40,
    imgsz=640,
    batch=16,
    workers=8
)
