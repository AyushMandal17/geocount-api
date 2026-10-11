"""Counting engine: YOLO26-OBB with tiled inference and rotated NMS.

Why tiled: on dense scenes (100+ objects) tiling cut mean abs. error roughly in half
vs. a single full-image pass (see docs/benchmarks.md).
"""
from __future__ import annotations

import base64
import math
from collections import Counter, defaultdict
from dataclasses import dataclass

import cv2
import numpy as np

DEFAULT_WEIGHTS = "yolo26m-obb.pt"

# Per-class confidence thresholds tuned on a 30-window dev split with YOLO26-x.
# They are weight-specific: re-tune if you change MODEL_WEIGHTS.
TUNED_CLASS_CONF = {
    "ship": 0.6, "harbor": 0.4, "storage tank": 0.1, "bridge": 0.6,
    "swimming pool": 0.6, "plane": 0.6, "small vehicle": 0.6, "large vehicle": 0.3,
}
TUNED_DEFAULT_CONF = 0.6
LENGTH_CLASSES = {"ship", "plane", "large vehicle", "bridge", "harbor"}


@dataclass
class Detection:
    cls_id: int
    cls_name: str
    conf: float
    xywhr: tuple  # (cx, cy, w, h, angle_rad) in full-image pixels

    @property
    def polygon(self) -> np.ndarray:
        cx, cy, w, h, a = self.xywhr
        return cv2.boxPoints(((cx, cy), (w, h), math.degrees(a)))

    @property
    def long_side(self) -> float:
        return max(self.xywhr[2], self.xywhr[3])


def load_model(weights: str = DEFAULT_WEIGHTS):
    from ultralytics import YOLO  # lazy: keeps helpers importable without torch

    return YOLO(weights)


def tile_origins(length: int, size: int, overlap: float) -> list[int]:
    if length <= size:
        return [0]
    step = max(1, int(size * (1 - overlap)))
    starts = list(range(0, length - size + 1, step))
    if starts[-1] + size < length:
        starts.append(length - size)
    return starts


def _intersection(a: tuple, b: tuple) -> float:
    ra = ((a[0], a[1]), (a[2], a[3]), math.degrees(a[4]))
    rb = ((b[0], b[1]), (b[2], b[3]), math.degrees(b[4]))
    ret, region = cv2.rotatedRectangleIntersection(ra, rb)
    if ret == cv2.INTERSECT_NONE or region is None or len(region) < 3:
        return 0.0
    return float(cv2.contourArea(region))


def rotated_iou(a: tuple, b: tuple) -> float:
    inter = _intersection(a, b)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def rotated_nms(dets: list[Detection], iou_thr: float = 0.3, ios_thr: float = 0.7) -> list[Detection]:
    """Class-wise NMS. Drops a box if IoU > iou_thr, or if most of the smaller box lies
    inside a kept box (IoS > ios_thr), which removes fragments cut off by tile edges."""
    by_cls = defaultdict(list)
    for d in dets:
        by_cls[d.cls_id].append(d)

    keep: list[Detection] = []
    for ds in by_cls.values():
        ds.sort(key=lambda d: -d.conf)
        kept: list[Detection] = []
        for d in ds:
            dr = math.hypot(d.xywhr[2], d.xywhr[3]) / 2
            dup = False
            for k in kept:
                kr = math.hypot(k.xywhr[2], k.xywhr[3]) / 2
                if math.hypot(d.xywhr[0] - k.xywhr[0], d.xywhr[1] - k.xywhr[1]) > dr + kr:
                    continue  # bounding circles don't touch
                inter = _intersection(d.xywhr, k.xywhr)
                if inter <= 0:
                    continue
                ad, ak = d.xywhr[2] * d.xywhr[3], k.xywhr[2] * k.xywhr[3]
                iou = inter / (ad + ak - inter) if (ad + ak - inter) > 0 else 0.0
                ios = inter / min(ad, ak) if min(ad, ak) > 0 else 0.0
                if iou > iou_thr or ios > ios_thr:
                    dup = True
                    break
            if not dup:
                kept.append(d)
        keep.extend(kept)
    return keep


def _to_detections(res, model, ox=0, oy=0) -> list[Detection]:
    if res.obb is None or len(res.obb) == 0:
        return []
    xywhr = res.obb.xywhr.cpu().numpy()
    confs = res.obb.conf.cpu().numpy()
    clss = res.obb.cls.cpu().numpy().astype(int)
    return [
        Detection(int(k), model.names[int(k)], float(c),
                  (float(cx) + ox, float(cy) + oy, float(w), float(h), float(a)))
        for (cx, cy, w, h, a), c, k in zip(xywhr, confs, clss)
    ]


def run_inference(
    model,
    image_bgr: np.ndarray,
    class_conf: dict | None = None,
    default_conf: float = 0.25,
    use_slicing: bool = True,
    tile: int = 1024,
    overlap: float = 0.2,
    imgsz: int = 1024,
) -> list[Detection]:
    """Detect objects. With `class_conf`, each class uses its own threshold; the model is
    run at the lowest threshold and results are filtered per class afterwards."""
    base_conf = min([*class_conf.values(), default_conf]) if class_conf else default_conf
    h, w = image_bgr.shape[:2]

    if use_slicing and max(h, w) > tile:
        tiles, offs = [], []
        for y in tile_origins(h, tile, overlap):
            for x in tile_origins(w, tile, overlap):
                tiles.append(image_bgr[y:y + tile, x:x + tile])
                offs.append((x, y))
        results = model.predict(tiles, imgsz=imgsz, conf=base_conf, verbose=False)
        dets = []
        for res, (ox, oy) in zip(results, offs):
            dets += _to_detections(res, model, ox, oy)
        dets = rotated_nms(dets)
    else:
        dets = _to_detections(model.predict(image_bgr, imgsz=imgsz, conf=base_conf, verbose=False)[0], model)

    thr = class_conf or {}
    return [d for d in dets if d.conf >= thr.get(d.cls_name, default_conf)]


def count_by_class(dets, classes=None) -> dict:
    c = Counter(d.cls_name for d in dets)
    if classes:
        return {k: c.get(k, 0) for k in classes}
    return dict(sorted(c.items(), key=lambda kv: -kv[1]))


def _palette(n: int = 16):
    cols = []
    for i in range(n):
        hsv = np.uint8([[[int(i * 180 / n), 220, 255]]])
        cols.append(tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]))
    return cols


_PALETTE = _palette()


def annotate(image_bgr: np.ndarray, dets, classes=None, max_side: int = 1600) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    s = min(1.0, max_side / max(h, w))
    out = cv2.resize(image_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else image_bgr.copy()
    th = max(1, round(max(out.shape[:2]) / 700))
    for d in dets:
        if classes and d.cls_name not in classes:
            continue
        pts = (d.polygon * s).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, _PALETTE[d.cls_id % len(_PALETTE)], th)
    return out


def to_data_uri(image_bgr: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
