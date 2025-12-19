#!/usr/bin/env python3
import os
import cv2
import numpy as np

# -------- SETTINGS --------
HEF_PATH   = "pieces_det.hef"
IMG_PATH   = "61/30.jpg"

CROP_LEFT  = 370
CROP_RIGHT = 1550           # exclusive

INPUT_SIZE = 640
NAMES = ["p_1","p_2","p_3","p_4","p_m"]
NUM_CLASSES = len(NAMES)

CONF_THRES = 0.50           # keep high while validating
IOU_THRES  = 0.60
MAX_DETS   = 200
COLOR_MODE = "RGB"          # "RGB" or "BGR"
# --------------------------


# ---------- Hailo imports (robust) ----------
def hailo_imports():
    import importlib

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
        HailoStreamInterface = pick(mod, "HailoStreamInterface")  # optional

        if all([Hef, VDevice, ConfigureParams, InferVStreams,
                InputVStreamsParams, OutputVStreamsParams, FormatType]):
            return (Hef, VDevice, ConfigureParams, InferVStreams,
                    InputVStreamsParams, OutputVStreamsParams, FormatType, HailoStreamInterface)

    raise RuntimeError(f"Could not import required Hailo bindings (last error: {last_err})")


def pick_format(FormatType, *cands):
    for c in cands:
        if hasattr(FormatType, c):
            return getattr(FormatType, c)
    raise RuntimeError(f"FormatType missing: {cands}")


# ---------- Letterbox ----------
def letterbox(im, new_shape=(640, 640), color=(114,114,114)):
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


# ---------- Math ----------
def sigmoid(x):
    x = x.astype(np.float32)
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out

def softmax(x, axis=-1):
    x = x.astype(np.float32)
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)

def iou_one_to_many(box, boxes):
    x1 = np.maximum(box[0], boxes[:,0])
    y1 = np.maximum(box[1], boxes[:,1])
    x2 = np.minimum(box[2], boxes[:,2])
    y2 = np.minimum(box[3], boxes[:,3])
    inter = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
    area1 = (box[2]-box[0]) * (box[3]-box[1])
    area2 = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
    return inter / (area1 + area2 - inter + 1e-9)

def nms_xyxy(boxes, scores, iou_th):
    if len(boxes) == 0:
        return []
    idxs = np.argsort(scores)[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        ious = iou_one_to_many(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_th]
    return keep


def to_hwc(a, c_hint):
    a = np.array(a)
    if a.ndim == 4 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 3:
        return None
    if a.shape[-1] in (64, c_hint):
        return a
    if a.shape[0] in (64, c_hint):
        return np.transpose(a, (1,2,0))
    return None


# ---------- YOLOv8 DFL decode ----------
def decode_yolov8_dfl(outputs, num_classes, input_size, conf_th, iou_th, max_det):
    regs, clss = [], []
    for name, t in outputs.items():
        a = to_hwc(t, num_classes)
        if a is None:
            continue
        if a.shape[-1] == 64:
            regs.append(a)
        elif a.shape[-1] == num_classes:
            clss.append(a)

    if len(regs) != 3 or len(clss) != 3:
        keys = [(k, np.array(v).shape) for k, v in outputs.items()]
        raise RuntimeError(f"Bad heads: reg={len(regs)} cls={len(clss)} outputs={keys}")

    reg_map = {(r.shape[0], r.shape[1]): r for r in regs}
    cls_map = {(c.shape[0], c.shape[1]): c for c in clss}

    proj = np.arange(16, dtype=np.float32)
    all_boxes, all_scores, all_cls = [], [], []

    for (h, w), reg in reg_map.items():
        if (h, w) not in cls_map:
            continue
        cls = cls_map[(h, w)]
        stride = input_size / float(h)

        prob = sigmoid(cls.astype(np.float32))  # always sigmoid

        reg_f = reg.astype(np.float32).reshape(h, w, 4, 16)
        reg_p = softmax(reg_f, axis=-1)
        dist = (reg_p * proj).sum(axis=-1)  # l,t,r,b

        yy, xx = np.meshgrid(np.arange(h, dtype=np.float32),
                             np.arange(w, dtype=np.float32), indexing="ij")
        cx = xx + 0.5
        cy = yy + 0.5

        l, t, r, b = dist[...,0], dist[...,1], dist[...,2], dist[...,3]
        x1 = (cx - l) * stride
        y1 = (cy - t) * stride
        x2 = (cx + r) * stride
        y2 = (cy + b) * stride

        boxes = np.stack([x1,y1,x2,y2], axis=-1).reshape(-1,4)
        boxes[:, [0,2]] = np.clip(boxes[:, [0,2]], 0, input_size - 1)
        boxes[:, [1,3]] = np.clip(boxes[:, [1,3]], 0, input_size - 1)

        p = prob.reshape(-1, num_classes)
        cls_ids = np.argmax(p, axis=1).astype(np.int32)
        scores = p[np.arange(p.shape[0]), cls_ids]

        m = scores >= conf_th
        all_boxes.append(boxes[m])
        all_scores.append(scores[m])
        all_cls.append(cls_ids[m])

    if not all_boxes:
        return np.zeros((0,4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int32)

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    cls_ids = np.concatenate(all_cls, axis=0)

    final = []
    for c in range(num_classes):
        idx = np.where(cls_ids == c)[0]
        if idx.size == 0:
            continue
        keep = nms_xyxy(boxes[idx], scores[idx], iou_th)
        final.extend(idx[keep].tolist())

    if not final:
        return np.zeros((0,4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int32)

    final = np.array(final, dtype=np.int32)
    final = final[np.argsort(scores[final])[::-1]][:max_det]
    return boxes[final], scores[final], cls_ids[final]


# ---------- Hailo infer (uint8 in, float32 out) ----------
def infer_u8(letterbox_hwc_u8):
    HEF, VDevice, ConfigureParams, InferVStreams, InputVStreamsParams, OutputVStreamsParams, FormatType, HailoStreamInterface = hailo_imports()
    hef = HEF(HEF_PATH)

    UINT8   = pick_format(FormatType, "UINT8", "U8")
    FLOAT32 = pick_format(FormatType, "FLOAT32", "F32")

    with VDevice() as vdevice:
        try:
            if HailoStreamInterface is not None:
                cfg = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
            else:
                cfg = ConfigureParams.create_from_hef(hef)
        except Exception:
            cfg = ConfigureParams.create_from_hef(hef)

        network_group = vdevice.configure(hef, cfg)[0]
        in_name = hef.get_input_vstream_infos()[0].name

        in_params = InputVStreamsParams.make_from_network_group(network_group, quantized=True,  format_type=UINT8)
        out_params = OutputVStreamsParams.make_from_network_group(network_group, quantized=False, format_type=FLOAT32)

        with InferVStreams(network_group, in_params, out_params) as infer:
            with network_group.activate():
                inp = np.expand_dims(letterbox_hwc_u8, axis=0)  # (1,H,W,C)
                outputs = infer.infer({in_name: inp})
    return outputs


def draw(img_bgr, boxes, scores, cls_ids):
    out = img_bgr.copy()
    for (x1,y1,x2,y2), sc, cid in zip(boxes, scores, cls_ids):
        x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
        label = f"{NAMES[int(cid)]} {float(sc):.2f}"
        cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 3)
        cv2.putText(out, label, (x1, max(20, y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return out


def main():
    if not os.path.exists(HEF_PATH):
        raise SystemExit(f"Missing HEF: {HEF_PATH}")
    img = cv2.imread(IMG_PATH)
    if img is None:
        raise SystemExit(f"Could not read image: {IMG_PATH}")

    roi = img[:, CROP_LEFT:CROP_RIGHT].copy()
    rh, rw = roi.shape[:2]

    lb_bgr, r, (padx, pady) = letterbox(roi, (INPUT_SIZE, INPUT_SIZE))

    if COLOR_MODE.upper() == "RGB":
        lb = cv2.cvtColor(lb_bgr, cv2.COLOR_BGR2RGB)   # feed RGB to network
        vis_base = lb_bgr                               # for saving
    else:
        lb = lb_bgr
        vis_base = lb_bgr

    cv2.imwrite("input_letterbox.jpg", vis_base)

    outputs = infer_u8(lb.astype(np.uint8))

    boxes, scores, cls_ids = decode_yolov8_dfl(outputs, NUM_CLASSES, INPUT_SIZE, CONF_THRES, IOU_THRES, MAX_DETS)
    print(f"[INFO] dets={len(scores)} top={(float(scores.max()) if len(scores) else 0):.3f}")

    # draw on letterbox
    pred_lb = draw(vis_base, boxes, scores, cls_ids)
    cv2.imwrite("pred_letterbox.jpg", pred_lb)

    # map boxes back to ROI and full image
    boxes_roi = boxes.astype(np.float32).copy()
    boxes_roi[:, [0,2]] = (boxes_roi[:, [0,2]] - padx) / r
    boxes_roi[:, [1,3]] = (boxes_roi[:, [1,3]] - pady) / r
    boxes_roi[:, [0,2]] = np.clip(boxes_roi[:, [0,2]], 0, rw - 1)
    boxes_roi[:, [1,3]] = np.clip(boxes_roi[:, [1,3]], 0, rh - 1)

    pred_roi = draw(roi, boxes_roi, scores, cls_ids)
    cv2.imwrite("pred_roi.jpg", pred_roi)

    full = img.copy()
    for (x1,y1,x2,y2), sc, cid in zip(boxes_roi, scores, cls_ids):
        x1f = int(x1 + CROP_LEFT); x2f = int(x2 + CROP_LEFT)
        y1f = int(y1);            y2f = int(y2)
        cv2.rectangle(full, (x1f,y1f), (x2f,y2f), (0,255,255), 3)
        cv2.putText(full, f"{NAMES[int(cid)]} {float(sc):.2f}", (x1f, max(20, y1f-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.imwrite("pred_full.jpg", full)

    print("Saved: input_letterbox.jpg, pred_letterbox.jpg, pred_roi.jpg, pred_full.jpg")


if __name__ == "__main__":
    main()
