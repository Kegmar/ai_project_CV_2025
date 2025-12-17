from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Iterator, Tuple, Optional

from PIL import Image, ImageDraw, ImageFont


class XmlDebugDrawConfig:
    def __init__(
        self,
        src_dir: Path = Path("multi-piece-train"),
        image_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
        line_width: int = 3,
        text_pad: int = 2,
        out_suffix: str = "_debug",
        out_ext: str = ".jpg",
        jpeg_quality: int = 92,
    ):
        self.src_dir = src_dir
        self.image_exts = image_exts
        self.line_width = line_width
        self.text_pad = text_pad
        self.out_suffix = out_suffix
        self.out_ext = out_ext
        self.jpeg_quality = jpeg_quality


class XmlDebugDrawer:
    def __init__(self, cfg: XmlDebugDrawConfig):
        self.cfg = cfg
        self.font = ImageFont.load_default()

    def find_image_for_xml(self, xml_path: Path) -> Optional[Path]:
        base = xml_path.stem
        for ext in self.cfg.image_exts:
            p = xml_path.with_suffix(ext)
            if p.exists():
                return p
            p2 = xml_path.parent / f"{base}{ext}"
            if p2.exists():
                return p2
        return None

    @staticmethod
    def color_for_class(name: str) -> Tuple[int, int, int]:
        """
        Fixed colors per class (RGB). Deterministic and consistent across runs.
        """
        palette = {
            "p_1": (255, 0, 0),     # red
            "p_2": (0, 255, 0),     # green
            "p_3": (0, 0, 255),     # blue
            "p_4": (255, 255, 0),   # yellow
            "p_m": (255, 0, 255),   # magenta
        }
        name = (name or "").strip()
        return palette.get(name, (255, 255, 255))  # fallback: white

    @staticmethod
    def iter_objects(xml_path: Path) -> Iterator[Tuple[str, int, int, int, int]]:
        root = ET.parse(xml_path).getroot()
        for obj in root.findall("object"):
            name = obj.findtext("name", default="unknown")
            bb = obj.find("bndbox")
            if bb is None:
                continue
            try:
                xmin = int(float(bb.findtext("xmin", "0")))
                ymin = int(float(bb.findtext("ymin", "0")))
                xmax = int(float(bb.findtext("xmax", "0")))
                ymax = int(float(bb.findtext("ymax", "0")))
            except ValueError:
                continue
            yield name, xmin, ymin, xmax, ymax

    @staticmethod
    def clamp(v: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, v))

    def run(self) -> None:
        if not self.cfg.src_dir.exists():
            raise SystemExit(f"Source folder not found: {self.cfg.src_dir}")

        xml_files = sorted(self.cfg.src_dir.glob("*.xml"))
        if not xml_files:
            raise SystemExit(f"No XML files found in {self.cfg.src_dir}")

        wrote = 0
        skipped = 0

        for xml_path in xml_files:
            img_path = self.find_image_for_xml(xml_path)
            if img_path is None:
                print(f"WARNING: No matching image for {xml_path.name}")
                skipped += 1
                continue

            # Skip if this is already a debug xml (just in case)
            if xml_path.stem.endswith(self.cfg.out_suffix):
                continue

            out_path = self.cfg.src_dir / f"{xml_path.stem}{self.cfg.out_suffix}{self.cfg.out_ext}"

            with Image.open(img_path) as im:
                im = im.convert("RGB")
                draw = ImageDraw.Draw(im)
                W, H = im.size

                for name, xmin, ymin, xmax, ymax in self.iter_objects(xml_path):
                    xmin = self.clamp(xmin, 0, W - 1)
                    ymin = self.clamp(ymin, 0, H - 1)
                    xmax = self.clamp(xmax, 0, W - 1)
                    ymax = self.clamp(ymax, 0, H - 1)
                    if xmax <= xmin or ymax <= ymin:
                        continue

                    color = self.color_for_class(name)

                    # bbox
                    draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=self.cfg.line_width)

                    # label bg + text
                    label = name
                    lbox = draw.textbbox((0, 0), label, font=self.font)
                    tw, th = lbox[2] - lbox[0], lbox[3] - lbox[1]

                    tx0 = xmin
                    ty0 = max(0, ymin - th - 2 * self.cfg.text_pad)
                    tx1 = xmin + tw + 2 * self.cfg.text_pad
                    ty1 = ty0 + th + 2 * self.cfg.text_pad

                    draw.rectangle([tx0, ty0, tx1, ty1], fill=color)
                    draw.text((tx0 + self.cfg.text_pad, ty0 + self.cfg.text_pad), label, fill=(0, 0, 0), font=self.font)

                im.save(out_path, quality=self.cfg.jpeg_quality)

            wrote += 1

        print(f"Done. Wrote {wrote} debug images into: {self.cfg.src_dir.resolve()}")
        if skipped:
            print(f"Skipped {skipped} (missing images).")