from ultralytics import YOLO
import glob
import os

model = YOLO("models/best.pt")  # move trained model here manually

INPUT = "data_raw"
OUTPUT = "results"
os.makedirs(OUTPUT, exist_ok=True)

# Detect on all images
for img in glob.glob(f"{INPUT}/**/*.jpg", recursive=True):
    model.predict(
        img,
        save=True,
        project=OUTPUT,
        name="detected",
        exist_ok=True
    )

print("Detection complete!")
