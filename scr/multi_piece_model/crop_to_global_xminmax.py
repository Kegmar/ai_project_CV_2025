from pathlib import Path
import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple

from PIL import Image


class MultiPieceCropperConfig:
    def __init__(
        self,
        src_dir: Path = Path("multi_piece_dataset"),
        image_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
        margin_percent: float = 0.05,  # +5%
        set_folder_tag: str = "multi_piece_dataset",
        set_path_tag: str = "Unspecified",
        depth_value: str = "3",
        skip_debug_suffix: str = "_debug", 
        jpeg_quality: int = 92,
    ):
        self.src_dir = src_dir
        self.image_exts = image_exts
        self.margin_percent = margin_percent
        self.set_folder_tag = set_folder_tag
        self.set_path_tag = set_path_tag
        self.depth_value = depth_value
        self.skip_debug_suffix = skip_debug_suffix
        self.jpeg_quality = jpeg_quality


class MultiPieceXmlCropper:
    def __init__(self, cfg: MultiPieceCropperConfig):
        self.cfg = cfg

    # ---------------- helpers ----------------
    def is_debug_name(self, p: Path) -> bool:
        return self.cfg.skip_debug_suffix in p.stem

    def find_image_for_xml(self, xml_path: Path) -> Optional[Path]:
        # skip debug xmls
        if self.is_debug_name(xml_path):
            return None

        base = xml_path.stem
        for ext in self.cfg.image_exts:
            p = xml_path.with_suffix(ext)
            if p.exists() and not self.is_debug_name(p):
                return p

            p2 = xml_path.parent / f"{base}{ext}"
            if p2.exists() and not self.is_debug_name(p2):
                return p2

        return None

    @staticmethod
    def ensure_child(root: ET.Element, tag: str) -> ET.Element:
        el = root.find(tag)
        if el is None:
            el = ET.SubElement(root, tag)
        return el

    @staticmethod
    def parse_boxes(xml_path: Path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes = []
        for obj in root.findall("object"):
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
            boxes.append((obj, bb, xmin, ymin, xmax, ymax))
        return tree, root, boxes

    def update_size_block(self, root: ET.Element, new_w: int, new_h: int) -> None:
        """
        Ensure:
          <size>
            <width>new_w</width>
            <height>new_h</height>
            <depth>3</depth>
          </size>
        """
        size_el = self.ensure_child(root, "size")
        self.ensure_child(size_el, "width").text = str(new_w)
        self.ensure_child(size_el, "height").text = str(new_h)
        depth_el = self.ensure_child(size_el, "depth")
        depth_el.text = (
            depth_el.text.strip()
            if (depth_el.text and depth_el.text.strip())
            else self.cfg.depth_value
        )

    def crop_and_update_one_in_place(
        self,
        xml_path: Path,
        img_path: Path,
        crop_left: int,
        crop_right: int,
    ) -> None:
        with Image.open(img_path) as im:
            w, h = im.size
            cl = max(0, min(crop_left, w - 1))
            cr = max(cl + 1, min(crop_right, w))
            new_w = cr - cl
            new_h = h

            cropped = im.crop((cl, 0, cr, h)).copy()

        save_kwargs = {}
        if img_path.suffix.lower() in (".jpg", ".jpeg"):
            save_kwargs["quality"] = self.cfg.jpeg_quality

        cropped.save(img_path, **save_kwargs)

        tree, root, boxes = self.parse_boxes(xml_path)

        self.ensure_child(root, "folder").text = self.cfg.set_folder_tag
        self.ensure_child(root, "filename").text = img_path.name
        self.ensure_child(root, "path").text = self.cfg.set_path_tag

        self.update_size_block(root, new_w, new_h)

        to_remove = []
        for obj, bb, xmin, ymin, xmax, ymax in boxes:
            xmin2 = xmin - cl
            xmax2 = xmax - cl

            xmin2 = max(0, min(xmin2, new_w - 1))
            xmax2 = max(0, min(xmax2, new_w))
            ymin2 = max(0, min(ymin, new_h - 1))
            ymax2 = max(0, min(ymax, new_h))

            if xmax2 <= xmin2 or ymax2 <= ymin2:
                to_remove.append(obj)
                continue

            bb.find("xmin").text = str(xmin2)
            bb.find("xmax").text = str(xmax2)
            bb.find("ymin").text = str(ymin2)
            bb.find("ymax").text = str(ymax2)

        for obj in to_remove:
            root.remove(obj)

        try:
            ET.indent(tree, space="\t", level=0)
        except Exception:
            pass

        tree.write(xml_path, encoding="utf-8", xml_declaration=False)

    def run(self) -> None:
        if not self.cfg.src_dir.exists():
            raise SystemExit(f"Source folder not found: {self.cfg.src_dir}")

        pairs: List[Tuple[Path, Path]] = []
        global_min_xmin: Optional[int] = None
        global_max_xmax: Optional[int] = None

        # Pass 1: compute global min/max from all boxes
        for xml_path in sorted(self.cfg.src_dir.glob("*.xml")):
            if self.is_debug_name(xml_path):
                continue

            img_path = self.find_image_for_xml(xml_path)
            if img_path is None:
                continue

            _, _, boxes = self.parse_boxes(xml_path)
            if not boxes:
                continue

            for _, _, xmin, _, xmax, _ in boxes:
                global_min_xmin = xmin if global_min_xmin is None else min(global_min_xmin, xmin)
                global_max_xmax = xmax if global_max_xmax is None else max(global_max_xmax, xmax)

            pairs.append((xml_path, img_path))

        if not pairs or global_min_xmin is None or global_max_xmax is None:
            raise SystemExit("No valid labeled (xml,image) pairs with boxes found.")

        with Image.open(pairs[0][1]) as im0:
            W0, _ = im0.size

        span = max(1, global_max_xmax - global_min_xmin)
        margin = int(round(span * self.cfg.margin_percent))

        crop_left = max(0, global_min_xmin - margin)
        crop_right = min(W0, global_max_xmax + margin)

        print(f"min_xmin={global_min_xmin}, max_xmax={global_max_xmax}, span={span}")
        print(f"margin={margin}px ({self.cfg.margin_percent*100:.0f}%)")
        print(f"crop_left={crop_left}, crop_right={crop_right}, new_width~={crop_right - crop_left}")

        # Pass 2: crop + update all pairs in-place
        converted = 0
        for xml_path, img_path in pairs:
            self.crop_and_update_one_in_place(
                xml_path=xml_path,
                img_path=img_path,
                crop_left=crop_left,
                crop_right=crop_right,
            )
            converted += 1

        print(f"Done. Cropped {converted} pairs in-place inside: {self.cfg.src_dir.resolve()}")