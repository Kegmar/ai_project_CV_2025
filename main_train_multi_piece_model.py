from pathlib import Path
import subprocess
import sys

from scr.multi_piece_model.add_debug_images import XmlDebugDrawConfig, XmlDebugDrawer
from scr.multi_piece_model.add_yolo_txt_labels import YoloLabelGenConfig, YoloLabelGenerator
from scr.multi_piece_model.crop_to_global_xminmax import MultiPieceCropperConfig, MultiPieceXmlCropper
from scr.multi_piece_model.split_dataset_for_yolo import YoloDatasetBuilder

def run_yolo_train(data_yaml: Path):
    data_yaml = Path(data_yaml).resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    # This is the python running the script (should be venv python)
    py = Path(sys.executable).resolve()

    # YOLO CLI installed by ultralytics should be right next to python.exe in the venv
    yolo_exe = py.parent / ("yolo.exe" if py.suffix.lower() == ".exe" else "yolo")

    if not yolo_exe.exists():
        raise RuntimeError(
            f"Cannot find YOLO CLI at: {yolo_exe}\n"
            f"Your python is: {py}\n"
            "Try: python -m pip install -U ultralytics"
        )

    cmd = [
        str(yolo_exe),
        "detect", "train",
        "model=yolo11s.pt",
        f"data={str(data_yaml)}",
        "imgsz=640",
        "epochs=200",
        "batch=16",
        "patience=30",
        "close_mosaic=20",
    ]

    print("[train] Using python:", py)
    print("[train] Using yolo:", yolo_exe)
    print("[train] Command:", " ".join(cmd))
    subprocess.run(cmd, check=True)


# 1) Debug images (optional)
add_debug_images = XmlDebugDrawConfig(
    src_dir=Path("multi_piece_dataset"),
    out_suffix="_debug",
    out_ext=".jpg",
    jpeg_quality=92,
    line_width=3,
)
XmlDebugDrawer(add_debug_images).run()

# 2) Generate YOLO .txt labels + yolo_classes.json
add_txt_lables = YoloLabelGenConfig(
    src_dir=Path("multi_piece_dataset"),
    info_json_path=Path("yolo_classes.json"),
)
YoloLabelGenerator(add_txt_lables).run()

# 3) Crop images + update XML
crop_images = MultiPieceCropperConfig(
    src_dir=Path("multi_piece_dataset"),
)
MultiPieceXmlCropper(crop_images).run()

# 4) Build YOLO dataset folder structure + data.yaml
builder = YoloDatasetBuilder(
    src_dir=Path("multi_piece_dataset"),
    out_dir=Path("multi_piece_train"),
    train_fraction=0.90,
    seed=42,
    image_exts=(".jpg", ".jpeg"),
)
builder.build()

# 5) Train
run_yolo_train(Path("multi_piece_train/data.yaml"))
