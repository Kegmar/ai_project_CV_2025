import os
import importlib
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import cv2
import numpy as np

class HailoYoloV8TestConfig:
    def __init__(
        self,
        hef_path: str = "pieces_det.hef",
        img_path: str = "61/30.jpg",
        crop_left: int = 370,
        crop_right: int = 1550,
        input_size: int = 640,
        class_names: Optional[List[str]] = None,  # 5 classes
        conf_thres: float = 0.50,
        iou_thres: float = 0.60,
        max_dets: int = 200,
        draw_topk: int = 60,
        color_mode: str = "RGB",  # "RGB" or "BGR"
        out_dir: Path = Path("."),  # where to write debug images
    ):
        self.hef_path = hef_path
        self.img_path = img_path
        self.crop_left = crop_left
        self.crop_right = crop_right
        self.input_size = input_size
        self.class_names = class_names or ["p_1", "p_2", "p_3", "p_4", "p_m"]
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_dets = max_dets
        self.draw_topk = draw_topk
        self.color_mode = color_mode
        self.out_dir = out_dir


class HailoYoloV8SingleImageTester:
    """
    Tests ONE image with a Hailo HEF that is expected to output YOLOv8-style
    (DFL reg head + cls head) with 5 classes.
    """

    def __init__(self, cfg: HailoYoloV8TestConfig):
        self.cfg = cfg
        self.cls_ch = len(cfg.class_names)  # expected 5

    @staticmethod
    def _hailo_imports():
        def pick(mod, *names):
            for n in names:
                if hasattr(mod, n):
                    return getattr(mod, n)
            return None

        last_err = None
        for modname in ("hailo_platform", "hailo_platform.pyhailort"):
            try:
                mod = importlib.import_module(modname)
            except Exception as e:
                last_err = e
                continue

            Hef = pick(mod, "Hef", "HEF")
            VDevice = pick(mod, "VDevice")
            ConfigureParams = pick(mod, "ConfigureParams")
            InferVStreams = pick(mod, "InferVStreams")
            InputVStreamsParams = pick(mod, "InputVStreamsParams", "InputVStreamParams")
            OutputVStreamsParams = pick(mod, "OutputVStreamsParams", "OutputVStreamParams")
            FormatType = pick(mod, "FormatType")
            HailoStreamInterface = pick(mod, "HailoStreamInterface")

            if all([Hef, VDevice, ConfigureParams, InferVStreams,
                    InputVStreamsParams, OutputVStreamsParams, FormatType]):
                return (Hef, VDevice, ConfigureParams, InferVStreams,
                        InputVStreamsParams, OutputVStreamsParams, FormatType, HailoStreamInterface)

        raise RuntimeError(f"Could not import Hailo bindings (last error: {last_err})")

    @staticmethod
    def _pick_format(FormatType, *cands):
        for c in cands:
            if hasattr(FormatType, c):
                return getattr(FormatType, c)
        raise RuntimeError(f"FormatType missing: {cands}")

    @staticmethod
    def _save_img(path: Path, bgr: np.ndarray):
        path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(path), bgr)
        if not ok:
            raise RuntimeError(f"cv2.imwrite FAILED: {path}")
        print("[SAVE]", path.resolve())

    @staticmethod
    def _letterbox(im: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)):
        h, w = im.shape[:2]
        new_h, new_w = new_shape
        r = min(new_h / h, new_w / w)
        rw, rh = int(round(w * r)), int(round(h * r))
        dw = (new_w - rw) / 2
        dh = (new_h - rh) / 2
        im_resized = cv2.resize(im, (rw, rh), interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        out = cv2.copyMakeBorder(im_resized, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=color)
        return out, r, (left, top)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        out = np.empty_like(x, dtype=np.float32)
        pos = x >= 0
        neg = ~pos
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        ex = np.exp(x[neg])
        out[neg] = ex / (1.0 + ex)
        return out

    @staticmethod
    def _softmax(x: np.ndarray, axis=-1) -> np.ndarray:
        x = x.astype(np.float32)
        x = x - np.max(x, axis=axis, keepdims=True)
        ex = np.exp(x)
        return ex / np.sum(ex, axis=axis, keepdims=True)

    @staticmethod
    def _iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (box[2] - box[0]) * (box[3] - box[1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return inter / (area1 + area2 - inter + 1e-9)

    def _nms_xyxy(self, boxes: np.ndarray, scores: np.ndarray, iou_th: float) -> List[int]:
        if len(boxes) == 0:
            return []
        idxs = np.argsort(scores)[::-1]
        keep: List[int] = []
        while idxs.size > 0:
            i = int(idxs[0])
            keep.append(i)
            if idxs.size == 1:
                break
            ious = self._iou_one_to_many(boxes[i], boxes[idxs[1:]])
            idxs = idxs[1:][ious < iou_th]
        return keep

    @staticmethod
    def _to_hwc(a: np.ndarray, c_hint: int) -> Optional[np.ndarray]:
        a = np.array(a)
        if a.ndim == 4 and a.shape[0] == 1:
            a = a[0]
        if a.ndim != 3:
            return None
        # (H,W,C)
        if a.shape[-1] in (64, c_hint):
            return a
        # (C,H,W) -> (H,W,C)
        if a.shape[0] in (64, c_hint):
            return np.transpose(a, (1, 2, 0))
        return None

    @staticmethod
    def _draw_topk(
        img_bgr: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        cls_ids: np.ndarray,
        names: List[str],
        color=(0, 255, 0),
        thick=3,
        topk=60,
    ) -> np.ndarray:
        out = img_bgr.copy()
        if len(scores) == 0:
            return out
        order = np.argsort(scores)[::-1][:topk]
        for i in order:
            x1, y1, x2, y2 = boxes[i]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cid = int(cls_ids[i])
            name = names[cid] if 0 <= cid < len(names) else str(cid)
            label = f"{name} {float(scores[i]):.2f}"
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
            cv2.putText(out, label, (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return out

    # ---------------- Decode (YOLOv8-style only) ----------------
    def decode_dfl_yolov8(
        self,
        outputs: Dict[str, np.ndarray],
        input_size: int,
        conf_th: float,
        iou_th: float,
        max_det: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        regs, clss = [], []
        for name, t in outputs.items():
            a = self._to_hwc(t, self.cls_ch)
            if a is None:
                continue
            if a.shape[-1] == 64:
                regs.append((name, a))
            elif a.shape[-1] == self.cls_ch:
                clss.append((name, a))

        if len(regs) != 3 or len(clss) != 3:
            keys = [(k, np.array(v).shape) for k, v in outputs.items()]
            raise RuntimeError(f"Bad heads: reg={len(regs)} cls={len(clss)} all={keys}")

        reg_map = {(r[1].shape[0], r[1].shape[1]): r[1] for r in regs}
        cls_map = {(c[1].shape[0], c[1].shape[1]): c[1] for c in clss}

        proj = np.arange(16, dtype=np.float32)

        all_boxes, all_scores, all_cls = [], [], []
        for (h, w), reg in reg_map.items():
            if (h, w) not in cls_map:
                continue
            cls = cls_map[(h, w)]
            stride = input_size / float(h)

            # YOLOv8-style class prob: sigmoid(cls_logits)
            prob = self._sigmoid(cls.astype(np.float32))  # (h,w,cls_ch)

            # DFL to l,t,r,b distances
            reg_f = reg.astype(np.float32).reshape(h, w, 4, 16)
            reg_p = self._softmax(reg_f, axis=-1)
            dist = (reg_p * proj).sum(axis=-1)

            yy, xx = np.meshgrid(np.arange(h, dtype=np.float32),
                                 np.arange(w, dtype=np.float32), indexing="ij")
            cx = xx + 0.5
            cy = yy + 0.5

            l, t, r, b = dist[..., 0], dist[..., 1], dist[..., 2], dist[..., 3]
            x1 = (cx - l) * stride
            y1 = (cy - t) * stride
            x2 = (cx + r) * stride
            y2 = (cy + b) * stride

            boxes = np.stack([x1, y1, x2, y2], axis=-1).reshape(-1, 4)
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, input_size - 1)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, input_size - 1)

            p = prob.reshape(-1, prob.shape[-1])
            cls_ids = np.argmax(p, axis=1).astype(np.int32)
            scores = p[np.arange(p.shape[0]), cls_ids]

            m = scores >= conf_th
            all_boxes.append(boxes[m])
            all_scores.append(scores[m])
            all_cls.append(cls_ids[m])

        if not all_boxes:
            return (np.zeros((0, 4), np.float32),
                    np.zeros((0,), np.float32),
                    np.zeros((0,), np.int32))

        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        cls_ids = np.concatenate(all_cls, axis=0)

        # class-wise NMS
        final: List[int] = []
        for c in range(int(cls_ids.max()) + 1 if cls_ids.size else 0):
            idx = np.where(cls_ids == c)[0]
            if idx.size == 0:
                continue
            keep = self._nms_xyxy(boxes[idx], scores[idx], iou_th)
            final.extend(idx[keep].tolist())

        if not final:
            return (np.zeros((0, 4), np.float32),
                    np.zeros((0,), np.float32),
                    np.zeros((0,), np.int32))

        final = np.array(final, dtype=np.int32)
        final = final[np.argsort(scores[final])[::-1]][:max_det]
        return boxes[final], scores[final], cls_ids[final]

    # ---------------- Quant/info printing ----------------
    @staticmethod
    def _extract_scale_zp(qi):
        def get(o, *names):
            for n in names:
                if hasattr(o, n):
                    return getattr(o, n)
            return None

        scale = get(qi, "scale", "qp_scale", "quantization_scale", "q_scale")
        zp = get(qi, "zero_point", "qp_zp", "zero", "zp", "q_zp")

        if isinstance(scale, (list, tuple, np.ndarray)):
            scale = float(np.array(scale).reshape(-1)[0])
        if isinstance(zp, (list, tuple, np.ndarray)):
            zp = float(np.array(zp).reshape(-1)[0])

        if scale is None or zp is None:
            return None, None
        return float(scale), float(zp)

    def print_hef_io_info(self, hef) -> None:
        in_infos = hef.get_input_vstream_infos()
        out_infos = hef.get_output_vstream_infos()

        print("\n[HEF] Inputs:")
        for info in in_infos:
            shp = getattr(info, "shape", None)
            print("  name:", info.name, "shape:", shp)
            qi = (getattr(info, "quant_info", None)
                  or getattr(info, "quantization_info", None)
                  or getattr(info, "quantization_params", None))
            if qi is not None:
                scale, zp = self._extract_scale_zp(qi)
                if scale is not None:
                    real0 = (0 - zp) * scale
                    real255 = (255 - zp) * scale
                    print(f"    quant: scale={scale} zp={zp}  => real_range~[{real0:.3f},{real255:.3f}]")
                else:
                    print("    quant: (present but unknown fields)")

        print("\n[HEF] Outputs:")
        for info in out_infos:
            print("  name:", info.name, "shape:", getattr(info, "shape", None))

    def print_cls_channel_stats(self, outputs: Dict[str, np.ndarray]) -> None:
        print("\n[DBG] Per-channel cls stats:")
        for k, v in outputs.items():
            a = self._to_hwc(v, self.cls_ch)
            if a is None or a.shape[-1] != self.cls_ch:
                continue
            mx = [float(a[..., i].max()) for i in range(self.cls_ch)]
            me = [float(a[..., i].mean()) for i in range(self.cls_ch)]
            mn = [float(a[..., i].min()) for i in range(self.cls_ch)]
            print(" ", k, "max:", ["%.2f" % x for x in mx])
            print("    mean:", ["%.2f" % x for x in me])
            print("    min :", ["%.2f" % x for x in mn])

    def run_hef_once_u8(self, letterbox_hwc_u8: np.ndarray):
        Hef, VDevice, ConfigureParams, InferVStreams, InputVStreamsParams, OutputVStreamsParams, FormatType, HailoStreamInterface = self._hailo_imports()
        hef = Hef(self.cfg.hef_path)

        UINT8 = self._pick_format(FormatType, "UINT8", "U8")
        FLOAT32 = self._pick_format(FormatType, "FLOAT32", "F32")

        with VDevice() as vdevice:
            try:
                if HailoStreamInterface is not None:
                    cfg = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
                else:
                    cfg = ConfigureParams.create_from_hef(hef)
            except Exception:
                cfg = ConfigureParams.create_from_hef(hef)

            network_group = vdevice.configure(hef, cfg)[0]

            in_infos = hef.get_input_vstream_infos()
            in_name = in_infos[0].name

            in_params = InputVStreamsParams.make_from_network_group(
                network_group, quantized=True, format_type=UINT8
            )
            out_params = OutputVStreamsParams.make_from_network_group(
                network_group, quantized=False, format_type=FLOAT32
            )

            with InferVStreams(network_group, in_params, out_params) as infer:
                with network_group.activate():
                    inp = np.expand_dims(letterbox_hwc_u8, axis=0)  # (1,H,W,C)
                    outputs = infer.infer({in_name: inp})

        return hef, outputs

    def run(self) -> None:
        if not os.path.exists(self.cfg.hef_path):
            raise SystemExit("Missing HEF: " + self.cfg.hef_path)

        img = cv2.imread(self.cfg.img_path)
        if img is None:
            raise SystemExit("Could not read image: " + self.cfg.img_path)

        # Crop ROI
        roi = img[:, self.cfg.crop_left:self.cfg.crop_right].copy()

        # Letterbox to model input size
        lb_bgr, _, _ = self._letterbox(roi, (self.cfg.input_size, self.cfg.input_size))
        self._save_img(self.cfg.out_dir / "input_letterbox_bgr.jpg", lb_bgr)

        # Convert color for inference
        if self.cfg.color_mode.upper() == "RGB":
            lb = cv2.cvtColor(lb_bgr, cv2.COLOR_BGR2RGB)
            self._save_img(self.cfg.out_dir / "input_letterbox_rgb.jpg", cv2.cvtColor(lb, cv2.COLOR_RGB2BGR))
        else:
            lb = lb_bgr

        # Run HEF once
        hef, outputs = self.run_hef_once_u8(lb.astype(np.uint8))
        self.print_hef_io_info(hef)
        self.print_cls_channel_stats(outputs)

        boxes, scores, cls_ids = self.decode_dfl_yolov8(
            outputs=outputs,
            input_size=self.cfg.input_size,
            conf_th=self.cfg.conf_thres,
            iou_th=self.cfg.iou_thres,
            max_det=self.cfg.max_dets,
        )

        vis_base = cv2.cvtColor(lb, cv2.COLOR_RGB2BGR) if self.cfg.color_mode.upper() == "RGB" else lb_bgr
        out = self._draw_topk(
            vis_base, boxes, scores, cls_ids,
            names=self.cfg.class_names,
            color=(0, 255, 0),
            thick=3,
            topk=self.cfg.draw_topk,
        )
        self._save_img(self.cfg.out_dir / "pred_yolov8_5cls.jpg", out)

        print(f"[YOLOV8 5cls] dets={len(scores)}  top_score={(float(scores.max()) if len(scores) else 0):.3f}")
        print("Open:", (self.cfg.out_dir / "pred_yolov8_5cls.jpg").resolve())


def main() -> None:
    cfg = HailoYoloV8TestConfig(
        hef_path="pieces_det.hef",
        img_path="61/30.jpg",
        crop_left=370,
        crop_right=1550,
        input_size=640,
        class_names=["p_1", "p_2", "p_3", "p_4", "p_m"],  # 5 classes
        conf_thres=0.50,
        iou_thres=0.60,
        max_dets=200,
        draw_topk=60,
        color_mode="RGB",
        out_dir=Path("."),
    )
    HailoYoloV8SingleImageTester(cfg).run()