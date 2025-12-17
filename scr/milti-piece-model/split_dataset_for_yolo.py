import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Tuple


@dataclass
class YoloDatasetBuilder:
    """
    Build a YOLO-style dataset structure from a flat folder containing:
      - images: <stem>.jpg/.jpeg/.png
      - labels: <stem>.txt  (YOLO format)
    Uses .txt files as the source of truth, and copies matching images + labels
    into:
      OUT_DIR/images/{train,val}/
      OUT_DIR/labels/{train,val}/
    Also writes a minimal Ultralytics data.yaml.
    """
    src_dir: Path = Path("multi-piece-train")
    out_dir: Path = Path("dataset_segdet")
    train_fraction: float = 0.90
    seed: int = 42
    image_exts: Sequence[str] = (".jpg", ".jpeg")
    class_names: Sequence[str] = ("p_1", "p_2", "p_3", "p_4", "p_m")

    debug_suffix: str = "_debug"

    def is_debug_image(self, p: Path) -> bool:
        return p.stem.endswith(self.debug_suffix)

    def find_image(self, stem: str) -> Optional[Path]:
        for ext in self.image_exts:
            p = self.src_dir / f"{stem}{ext}"
            if p.exists():
                return p
        return None

    def collect_stems(self) -> list[str]:
        if not self.src_dir.exists():
            raise FileNotFoundError(f"Source folder not found: {self.src_dir}")

        stems: list[str] = []
        for txt_path in sorted(self.src_dir.glob("*.txt")):
            stem = txt_path.stem

            # skip debug label artifacts
            if stem.endswith(self.debug_suffix):
                continue

            img_path = self.find_image(stem)
            if img_path is None or self.is_debug_image(img_path):
                print(f"WARNING: missing/invalid image for {txt_path.name} -> skipping")
                continue

            stems.append(stem)

        if not stems:
            raise RuntimeError("No valid (image, label) pairs found.")

        return stems

    def split_stems(self, stems: Sequence[str]) -> Tuple[list[str], list[str]]:
        if not (0.0 < self.train_fraction < 1.0):
            raise ValueError("train_fraction must be between 0 and 1 (exclusive).")

        stems = list(stems)
        rng = random.Random(self.seed)
        rng.shuffle(stems)

        n_train = int(len(stems) * self.train_fraction)
        train_stems = stems[:n_train]
        val_stems = stems[n_train:]
        return train_stems, val_stems

    def make_dirs(self) -> None:
        (self.out_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

    def copy_pair(self, stem: str, split: str) -> bool:
        img_src = self.find_image(stem)
        lbl_src = self.src_dir / f"{stem}.txt"
        if img_src is None or not lbl_src.exists():
            return False

        shutil.copy2(img_src, self.out_dir / "images" / split / img_src.name)
        shutil.copy2(lbl_src, self.out_dir / "labels" / split / f"{stem}.txt")
        return True

    def write_data_yaml(self) -> Path:
        data_yaml = self.out_dir / "data.yaml"

        lines = [
            f"path: {self.out_dir.as_posix()}",
            "train: images/train",
            "val: images/val",
            "names:",
        ]
        for i, n in enumerate(self.class_names):
            lines.append(f"  {i}: {n}")
        lines.append("")

        data_yaml.write_text("\n".join(lines), encoding="utf-8")
        return data_yaml

    def build(self) -> Tuple[int, int, int]:
        stems = self.collect_stems()
        train_stems, val_stems = self.split_stems(stems)

        self.make_dirs()

        copied_train = sum(self.copy_pair(s, "train") for s in train_stems)
        copied_val = sum(self.copy_pair(s, "val") for s in val_stems)
        total = copied_train + copied_val

        print(f"Done. Output dataset: {self.out_dir.resolve()}")
        print(f"Train: {copied_train}  | Val: {copied_val}  | Total: {total}")

        yaml_path = self.write_data_yaml()
        print(f"Wrote {yaml_path.resolve()}")

        return copied_train, copied_val, total