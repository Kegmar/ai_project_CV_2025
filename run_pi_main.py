#!/usr/bin/env python3
from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import cv2
from picamera2 import Picamera2
from libcamera import controls

from hailo_platform import (
    HEF, VDevice, ConfigureParams, HailoStreamInterface,
    InputVStreamParams, OutputVStreamParams, InferVStreams, FormatType
)

# ---------------- CONFIG ----------------
DET_HEF_PATH = "pieces_det.hef"
CLS_HEF_PATH = "piece_cls.hef"
CLS_CLASSES_TXT = "classes.txt"

ROI_X_LEFT  = 370
ROI_X_RIGHT = 1550  # exclusive
CAM_RES = (1920, 1080)

DEFAULT_IMGSZ = 640
CLASS_NAMES = ["p_1", "p_2", "p_3", "p_4", "p_m"]
NUM_CLASSES = len(CLASS_NAMES)

CONF_THRES = 0.30
IOU_THRES  = 0.50
MAX_DETS   = 200

REG_MAX = 16

# Detector input mode (your detector proved OK)
DET_INPUT_MODE = "uint8"  # "uint8" or "float01"

AE_CONSTRAINT_MODE = "highlight"
DEBUG_DIR = Path.home() / "Desktop" / "hailo_debug"

# You said capture_array() is RGB (OpenCV display needs RGB->BGR swap)
CAPTURE_IS_RGB = True

# ---- Classifier specifics ----
CLS_SIZE = 256
CLS_PAD_COLOR = (114, 114, 114)  # RGB
MAX_CLASSIFY = 3

# Only classify detections of YOLO class "p_1"
CLS_ON_DET_CLASS = "p_1"
CLS_ON_CID = CLASS_NAMES.index(CLS_ON_DET_CLASS)

# IMPORTANT toggles to fix "always same class" issues:
# 1) If crops look correct but predictions are wrong -> try swapping RB.
CLS_SWAP_RB = False  # <-- if you only get piece_81/piece_88, try True

# 2) If quantization expectation is different than assumed -> try float255.
# "uint8_quantized": send uint8 and quantized=True (your stated expectation)
# "float255": send float32 0..255 and quantized=False (HailoRT quantizes)
CLS_INPUT_MODE = "uint8_quantized"  # <-- if stuck, try "float255"
# ----------------------------------------


def letterbox(im: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = im.shape[:2]
    nh, nw = new_shape
    r = min(nw / w, nh / h)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_x = (nw - new_w) // 2
    pad_y = (nh - new_h) // 2

    out = np.full((nh, nw, 3), color, dtype=resized.dtype)
    out[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return out, r, (pad_x, pad_y)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray, axis=-1) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-9)


def maybe_probs_from_output(vec: np.ndarray) -> np.ndarray:
    """
    If output already looks like probabilities (0..1, sums ~1), return it.
    Else treat as logits and softmax it.
    """
    v = vec.astype(np.float32).reshape(-1)
    s = float(np.sum(v))
    if np.all(v >= -1e-4) and np.all(v <= 1.0 + 1e-4) and 0.90 <= s <= 1.10:
        return v
    return softmax(v, axis=0)


def iou_one_to_many(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    a1 = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    a2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return inter / (a1 + a2 - inter + 1e-9)


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thres: float):
    if boxes.size == 0:
        return np.array([], dtype=np.int32)

    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = iou_one_to_many(boxes[i], boxes[order[1:]])
        order = order[1:][ious <= iou_thres]
    return np.array(keep, dtype=np.int32)


def decode_yolov8_dfl(outputs: dict[str, np.ndarray], imgsz: int):
    cls_heads = []
    reg_heads = []

    for _, t in outputs.items():
        a = np.asarray(t)
        if a.ndim == 4 and a.shape[0] == 1:
            a = a[0]
        if a.ndim != 3:
            continue

        # CHW -> HWC if needed
        if a.shape[0] in (NUM_CLASSES, 4 * REG_MAX) and a.shape[-1] not in (NUM_CLASSES, 4 * REG_MAX):
            a = np.transpose(a, (1, 2, 0))

        c = a.shape[-1]
        if c == NUM_CLASSES:
            cls_heads.append(a.astype(np.float32))
        elif c == 4 * REG_MAX:
            reg_heads.append(a.astype(np.float32))

    if len(cls_heads) != 3 or len(reg_heads) != 3:
        raise RuntimeError(f"Bad heads. cls={len(cls_heads)} reg={len(reg_heads)}")

    cls_heads.sort(key=lambda x: x.shape[0] * x.shape[1], reverse=True)
    reg_heads.sort(key=lambda x: x.shape[0] * x.shape[1], reverse=True)

    proj = np.arange(REG_MAX, dtype=np.float32)

    all_boxes, all_scores, all_cls = [], [], []
    for cls, reg in zip(cls_heads, reg_heads):
        h, w, _ = cls.shape
        stride = imgsz / float(h)

        prob = sigmoid(cls).reshape(-1, NUM_CLASSES)

        reg = reg.reshape(-1, 4, REG_MAX)
        reg = softmax(reg, axis=2)
        dist = (reg * proj).sum(axis=2) * stride

        yy, xx = np.meshgrid(np.arange(h, dtype=np.float32),
                             np.arange(w, dtype=np.float32), indexing="ij")
        cx = (xx.reshape(-1) + 0.5) * stride
        cy = (yy.reshape(-1) + 0.5) * stride

        l, t, r, b = dist[:, 0], dist[:, 1], dist[:, 2], dist[:, 3]
        x1 = cx - l
        y1 = cy - t
        x2 = cx + r
        y2 = cy + b
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, imgsz - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, imgsz - 1)

        cls_ids = np.argmax(prob, axis=1).astype(np.int32)
        scores = prob[np.arange(prob.shape[0]), cls_ids]

        m = scores >= CONF_THRES
        if np.any(m):
            all_boxes.append(boxes[m])
            all_scores.append(scores[m])
            all_cls.append(cls_ids[m])

    if not all_boxes:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int32)

    boxes = np.concatenate(all_boxes, axis=0).astype(np.float32)
    scores = np.concatenate(all_scores, axis=0).astype(np.float32)
    cls_ids = np.concatenate(all_cls, axis=0).astype(np.int32)

    final = []
    for c in range(NUM_CLASSES):
        idx = np.where(cls_ids == c)[0]
        if idx.size == 0:
            continue
        keep = nms_xyxy(boxes[idx], scores[idx], IOU_THRES)
        final.extend(idx[keep].tolist())

    if not final:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int32)

    final = np.array(final, dtype=np.int32)
    final = final[np.argsort(scores[final])[::-1]][:MAX_DETS]
    return boxes[final], scores[final], cls_ids[final]


def configure_picamera(picam2: Picamera2):
    camera_config = picam2.create_still_configuration(
        main={"size": CAM_RES, "format": "BGR888"}
    )
    picam2.configure(camera_config)
    picam2.start()

    try:
        picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
    except Exception as e:
        print("Autofocus controls not available or failed:", e)

    mode_map = {
        "normal": controls.AeConstraintModeEnum.Normal,
        "highlight": controls.AeConstraintModeEnum.Highlight,
        "shadows": controls.AeConstraintModeEnum.Shadows,
    }
    chosen_mode = mode_map.get(str(AE_CONSTRAINT_MODE).lower(),
                              controls.AeConstraintModeEnum.Normal)
    try:
        picam2.set_controls({"AeConstraintMode": chosen_mode})
        print(f"AeConstraintMode set to: {AE_CONSTRAINT_MODE}")
    except Exception as e:
        print("Failed to set AeConstraintMode:", e)


def load_class_list(path: Path) -> list[str]:
    out: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(s)
    return out


def draw_label(vis_bgr: np.ndarray, x: int, y: int, text: str, color=(0, 255, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thick = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
    x2 = x + tw + 6
    y2 = y + th + baseline + 6
    x = max(0, x)
    y = max(0, y)
    cv2.rectangle(vis_bgr, (x, y), (x2, y2), (0, 0, 0), -1)
    cv2.putText(vis_bgr, text, (x + 3, y + th + 3), font, scale, color, thick, cv2.LINE_AA)


def extract_logits(outputs: dict[str, np.ndarray]) -> np.ndarray:
    if not outputs:
        raise RuntimeError("Classifier returned no outputs")
    arr = np.asarray(next(iter(outputs.values())))
    arr = np.squeeze(arr).astype(np.float32)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def prep_classifier_input(full_rgb: np.ndarray, box_xyxy: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box_xyxy
    crop = full_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((CLS_SIZE, CLS_SIZE, 3), dtype=np.uint8)

    # Optional RB swap if your crop is actually BGR
    if CLS_SWAP_RB:
        crop = crop[..., ::-1]

    # letterbox to 256x256 (keeps aspect)
    inp, _, _ = letterbox(crop, (CLS_SIZE, CLS_SIZE), color=CLS_PAD_COLOR)
    return np.ascontiguousarray(inp, dtype=np.uint8)


def main():
    det_hef_path = Path(DET_HEF_PATH)
    cls_hef_path = Path(CLS_HEF_PATH)
    classes_path = Path(CLS_CLASSES_TXT)

    if not det_hef_path.exists():
        raise SystemExit(f"Detector HEF not found: {det_hef_path.resolve()}")
    if not cls_hef_path.exists():
        raise SystemExit(f"Classifier HEF not found: {cls_hef_path.resolve()}")
    if not classes_path.exists():
        raise SystemExit(f"classes.txt not found: {classes_path.resolve()}")

    cls_names = load_class_list(classes_path)
    if len(cls_names) == 0:
        raise SystemExit("classes.txt is empty")

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    picam2 = Picamera2()
    configure_picamera(picam2)

    hef_det = HEF(str(det_hef_path))
    hef_cls = HEF(str(cls_hef_path))

    with VDevice(VDevice.create_params()) as vdevice:
        # --- detector NG ---
        cfg_det = ConfigureParams.create_from_hef(hef_det, interface=HailoStreamInterface.PCIe)
        ng_det = vdevice.configure(hef_det, cfg_det)[0]

        det_in_info = hef_det.get_input_vstream_infos()[0]
        det_in_name = det_in_info.name
        det_in_shape = det_in_info.shape
        imgsz = int(det_in_shape[0]) if (len(det_in_shape) == 3 and det_in_shape[0] == det_in_shape[1]) else DEFAULT_IMGSZ
        print("DET input:", det_in_name, det_in_shape, "imgsz:", imgsz, "DET_INPUT_MODE:", DET_INPUT_MODE)

        if DET_INPUT_MODE == "float01":
            det_in_params = InputVStreamParams.make(ng_det, quantized=False, format_type=FormatType.FLOAT32)
        else:
            det_in_params = InputVStreamParams.make(ng_det, quantized=True, format_type=FormatType.UINT8)

        det_out_params = OutputVStreamParams.make(ng_det, quantized=False, format_type=FormatType.FLOAT32)

        # --- classifier NG ---
        cfg_cls = ConfigureParams.create_from_hef(hef_cls, interface=HailoStreamInterface.PCIe)
        ng_cls = vdevice.configure(hef_cls, cfg_cls)[0]

        cls_in_info = hef_cls.get_input_vstream_infos()[0]
        cls_in_name = cls_in_info.name
        cls_in_shape = cls_in_info.shape
        print("CLS input:", cls_in_name, cls_in_shape, "CLS_INPUT_MODE:", CLS_INPUT_MODE, "CLS_SWAP_RB:", CLS_SWAP_RB)
        print("Classify only YOLO class:", CLS_ON_DET_CLASS, "(cid:", CLS_ON_CID, ")")

        if CLS_INPUT_MODE == "float255":
            cls_in_params = InputVStreamParams.make(ng_cls, quantized=False, format_type=FormatType.FLOAT32)
        else:
            cls_in_params = InputVStreamParams.make(ng_cls, quantized=True, format_type=FormatType.UINT8)

        cls_out_params = OutputVStreamParams.make(ng_cls, quantized=False, format_type=FormatType.FLOAT32)

        # Create pipelines once; activate ONE NG at a time during infer
        with InferVStreams(ng_det, det_in_params, det_out_params) as infer_det, \
             InferVStreams(ng_cls, cls_in_params, cls_out_params) as infer_cls:

            last = time.time()
            fps = 0.0
            save_idx = 1

            while True:
                frame = picam2.capture_array("main")
                full_rgb = frame if CAPTURE_IS_RGB else frame[..., ::-1]

                H, W = full_rgb.shape[:2]
                roi_rgb = full_rgb[:, ROI_X_LEFT:ROI_X_RIGHT]
                roi_h, roi_w = roi_rgb.shape[:2]

                # -------- DETECT --------
                det_inp_rgb, scale, (padx, pady) = letterbox(roi_rgb, (imgsz, imgsz))

                if DET_INPUT_MODE == "float01":
                    det_inp = np.ascontiguousarray(det_inp_rgb.astype(np.float32) / 255.0)
                else:
                    det_inp = np.ascontiguousarray(det_inp_rgb, dtype=np.uint8)

                with ng_det.activate():
                    det_outputs = infer_det.infer({det_in_name: det_inp[np.newaxis, ...]})

                boxes, scores, cls_ids = decode_yolov8_dfl(det_outputs, imgsz)

                det_list = []
                for (x1, y1, x2, y2), conf, cid in zip(boxes, scores, cls_ids):
                    # letterbox -> ROI coords
                    x1 = (x1 - padx) / scale
                    x2 = (x2 - padx) / scale
                    y1 = (y1 - pady) / scale
                    y2 = (y2 - pady) / scale

                    x1 = max(0.0, min(roi_w - 1.0, x1))
                    x2 = max(0.0, min(roi_w - 1.0, x2))
                    y1 = max(0.0, min(roi_h - 1.0, y1))
                    y2 = max(0.0, min(roi_h - 1.0, y2))

                    # ROI -> full coords
                    x1f = int(x1 + ROI_X_LEFT)
                    x2f = int(x2 + ROI_X_LEFT)
                    y1f = int(y1)
                    y2f = int(y2)

                    x1f = max(0, min(W - 1, x1f))
                    x2f = max(0, min(W - 1, x2f))
                    y1f = max(0, min(H - 1, y1f))
                    y2f = max(0, min(H - 1, y2f))
                    if x2f <= x1f + 1 or y2f <= y1f + 1:
                        continue

                    det_name = CLASS_NAMES[int(cid)] if 0 <= int(cid) < NUM_CLASSES else str(int(cid))
                    det_list.append({
                        "box": (x1f, y1f, x2f, y2f),
                        "conf": float(conf),
                        "det_name": det_name,
                        "cid": int(cid),
                        "cls_name": None,
                        "cls_conf": None,
                    })

                det_list.sort(key=lambda d: d["conf"], reverse=True)

                # -------- CLASSIFY ONLY p_1 --------
                p1_indices = [i for i, d in enumerate(det_list) if d["cid"] == CLS_ON_CID][:MAX_CLASSIFY]

                for i in p1_indices:
                    inp_u8 = prep_classifier_input(full_rgb, det_list[i]["box"])  # UINT8 RGB

                    if CLS_INPUT_MODE == "float255":
                        cls_inp = inp_u8.astype(np.float32)  # 0..255
                    else:
                        cls_inp = inp_u8  # uint8

                    with ng_cls.activate():
                        cls_outputs = infer_cls.infer({cls_in_name: cls_inp[np.newaxis, ...]})

                    vec = extract_logits(cls_outputs)
                    probs = maybe_probs_from_output(vec)

                    pred_idx = int(np.argmax(probs))
                    pred_conf = float(probs[pred_idx])
                    pred_name = cls_names[pred_idx] if 0 <= pred_idx < len(cls_names) else str(pred_idx)

                    det_list[i]["cls_name"] = pred_name
                    det_list[i]["cls_conf"] = pred_conf

                # -------- VIS --------
                vis = full_rgb[..., ::-1].copy()
                cv2.rectangle(vis, (ROI_X_LEFT, 0), (ROI_X_RIGHT, H - 1), (255, 0, 0), 2)

                for d in det_list:
                    x1f, y1f, x2f, y2f = d["box"]
                    cv2.rectangle(vis, (x1f, y1f), (x2f, y2f), (0, 255, 0), 2)

                    if d["cls_name"] is not None:
                        txt = f'{d["det_name"]} {d["conf"]:.2f} | {d["cls_name"]} {d["cls_conf"]:.2f}'
                    else:
                        txt = f'{d["det_name"]} {d["conf"]:.2f}'
                    draw_label(vis, x1f, max(0, y1f - 22), txt, color=(0, 255, 0))

                now = time.time()
                dt = now - last
                last = now
                fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6))

                cv2.putText(
                    vis,
                    f"FPS {fps:.1f}  dets:{len(det_list)}  p_1:{len([d for d in det_list if d['cid']==CLS_ON_CID])}  cls:{len(p1_indices)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2
                )

                cv2.imshow("pieces_det+cls", vis)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('s'):
                    base = DEBUG_DIR / f"dbg_{save_idx:03d}"
                    cv2.imwrite(str(base) + "_vis_bgr.jpg", vis)
                    cv2.imwrite(str(base) + "_full_bgr.jpg", full_rgb[..., ::-1])

                    # save the exact classifier inputs used (top p_1)
                    for j, det_i in enumerate(p1_indices):
                        inp_u8 = prep_classifier_input(full_rgb, det_list[det_i]["box"])
                        cv2.imwrite(str(base) + f"_cls_in{j}_bgr.jpg", inp_u8[..., ::-1])  # saved as BGR for viewing

                    print("Saved:", base)
                    save_idx += 1

                if key in (27, ord('q')):
                    break

    cv2.destroyAllWindows()
    try:
        picam2.stop()
    except Exception:
        pass


if __name__ == "__main__":
    main()
