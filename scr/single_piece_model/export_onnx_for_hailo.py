import argparse
from pathlib import Path
import torch
import timm

class OnnxExporterConfig:
    def __init__(
        self,
        ckpt: str = "runs/piece_cls/best.pth",
        model: str = "mobilenetv3_small_100",
        imgsz: int = 256,
        out: str = "runs/piece_cls/piece_cls_op13.onnx",
        opset_version: int = 13,
        input_name: str = "images",
        output_name: str = "logits",
        do_constant_folding: bool = True,
        dynamo: bool = False,  # legacy exporter for Hailo compatibility
        map_location: str = "cpu",
        strict: bool = True,
    ):
        self.ckpt = ckpt
        self.model = model
        self.imgsz = imgsz
        self.out = out
        self.opset_version = opset_version
        self.input_name = input_name
        self.output_name = output_name
        self.do_constant_folding = do_constant_folding
        self.dynamo = dynamo
        self.map_location = map_location
        self.strict = strict


class OnnxExporter:
    def __init__(self, cfg: OnnxExporterConfig):
        self.cfg = cfg

    @staticmethod
    def build_argparser() -> argparse.ArgumentParser:
        ap = argparse.ArgumentParser()
        ap.add_argument("--ckpt", default="runs/piece_cls/best.pth")
        ap.add_argument("--model", default="mobilenetv3_small_100")
        ap.add_argument("--imgsz", type=int, default=256)
        ap.add_argument("--out", default="runs/piece_cls/piece_cls_op13.onnx")
        return ap

    @classmethod
    def from_args(cls) -> "OnnxExporter":
        ap = cls.build_argparser()
        args = ap.parse_args()
        cfg = OnnxExporterConfig(
            ckpt=args.ckpt,
            model=args.model,
            imgsz=args.imgsz,
            out=args.out,
        )
        return cls(cfg)

    def run(self) -> Path:
        ckpt = torch.load(self.cfg.ckpt, map_location=self.cfg.map_location)
        classes = ckpt.get("classes", None)
        if classes is None:
            raise SystemExit("Checkpoint missing 'classes'. Re-save ckpt or pass num_classes manually.")
        num_classes = len(classes)

        model = timm.create_model(self.cfg.model, pretrained=False, num_classes=num_classes)
        model.load_state_dict(ckpt["model"], strict=self.cfg.strict)
        model.eval()

        dummy = torch.zeros(1, 3, self.cfg.imgsz, self.cfg.imgsz, dtype=torch.float32)

        out_path = Path(self.cfg.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # IMPORTANT: export with legacy exporter for Hailo compatibility
        torch.onnx.export(
            model,
            dummy,
            str(out_path),
            input_names=[self.cfg.input_name],
            output_names=[self.cfg.output_name],
            opset_version=self.cfg.opset_version,
            do_constant_folding=self.cfg.do_constant_folding,
            dynamo=self.cfg.dynamo,  # <-- key
        )

        print("[OK] Exported:", out_path.resolve())
        return out_path
