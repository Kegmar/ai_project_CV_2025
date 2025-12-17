import json
from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple, Iterator

from PIL import Image

class YoloLabelGenConfig:
    def __init__(
        self,
        src_dir: Path = Path("multi_piece_dataset"),          
        info_json_path: Path = Path("yolo_classes.json"),   
        image_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
        class_names: List[str] = None,
        skip_debug_suffix: str = "_debug",
    ):
        if class_names is None:
            class_names = ["p_1", "p_2", "p_3", "p_4", "p_m"]  # ordered
        self.src_dir = src_dir
        self.info_json_path = info_json_path
        self.image_exts = image_exts
        self.class_names = class_names
        self.skip_debug_suffix = skip_debug_suffix

        # Derived
        self.class_to_id: Dict[str, int] = {name: i for i, name in enumerate(self.class_names)}


class YoloLabelGenerator:
    def __init__(self, cfg: YoloLabelGenConfig):
        self.cfg = cfg

    def find_image_for_xml(self, xml_path: Path) -> Optional[Path]:
        stem = xml_path.stem
        for ext in self.cfg.image_exts:
            p = xml_path.with_suffix(ext)
            if p.exists():
                return p
            p2 = xml_path.parent / f"{stem}{ext}"
            if p2.exists():
                return p2
        return None

    @staticmethod
    def clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    def voc_to_yolo(
        self, xmin: float, ymin: float, xmax: float, ymax: float, w: int, h: int
    ) -> Optional[Tuple[float, float, float, float]]:
        xmin = self.clamp(xmin, 0, w - 1)
        xmax = self.clamp(xmax, 0, w - 1)
        ymin = self.clamp(ymin, 0, h - 1)
        ymax = self.clamp(ymax, 0, h - 1)

        bw = xmax - xmin
        bh = ymax - ymin
        if bw <= 1 or bh <= 1:
            return None

        xc = xmin + bw / 2.0
        yc = ymin + bh / 2.0
        return (xc / w, yc / h, bw / w, bh / h)

    @staticmethod
    def iter_objects(xml_path: Path) -> Iterator[Tuple[str, float, float, float, float]]:
        root = ET.parse(xml_path).getroot()
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip()
            bb = obj.find("bndbox")
            if bb is None:
                continue
            try:
                xmin = float(bb.findtext("xmin", "0"))
                ymin = float(bb.findtext("ymin", "0"))
                xmax = float(bb.findtext("xmax", "0"))
                ymax = float(bb.findtext("ymax", "0"))
            except ValueError:
                continue
            yield name, xmin, ymin, xmax, ymax

    def run(self) -> None:
        if not self.cfg.src_dir.exists():
            raise SystemExit(f"Folder not found: {self.cfg.src_dir}")

        xml_files = sorted(self.cfg.src_dir.glob("*.xml"))
        if not xml_files:
            raise SystemExit(f"No XML files found in: {self.cfg.src_dir}")

        counts = Counter()

        for xml_path in xml_files:
            if xml_path.stem.endswith(self.cfg.skip_debug_suffix):
                continue

            img_path = self.find_image_for_xml(xml_path)
            if img_path is None:
                print(f"WARNING: Missing image for {xml_path.name} (skipping)")
                continue

            with Image.open(img_path) as im:
                w, h = im.size

            yolo_lines: List[str] = []
            for name, xmin, ymin, xmax, ymax in self.iter_objects(xml_path):
                if name not in self.cfg.class_to_id:
                    raise ValueError(
                        f"Unexpected class '{name}' in {xml_path.name}. "
                        f"Expected only: {list(self.cfg.class_to_id.keys())}"
                    )

                class_id = self.cfg.class_to_id[name]
                box = self.voc_to_yolo(xmin, ymin, xmax, ymax, w, h)
                if box is None:
                    continue

                xc, yc, bw, bh = box
                yolo_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                counts[name] += 1

            (self.cfg.src_dir / f"{xml_path.stem}.txt").write_text(
                "\n".join(yolo_lines), encoding="utf-8"
            )

        info = {
            "nc": len(self.cfg.class_names),
            "names": {str(i): self.cfg.class_names[i] for i in range(len(self.cfg.class_names))},
            "counts_per_class": dict(counts),
        }
        self.cfg.info_json_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

        print(f"Done. Wrote YOLO .txt labels into: {self.cfg.src_dir.resolve()}")
        print(f"Wrote class info JSON to: {self.cfg.info_json_path.resolve()}")