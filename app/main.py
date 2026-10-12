"""GeoCount API: upload an aerial image (+ optional question) -> counts + annotated image."""
import asyncio
import os
import statistics
import time
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from . import engine, query

MAX_UPLOAD_MB = 15
MAX_SIDE = 4096  # larger images are downscaled to keep CPU latency sane

@asynccontextmanager
async def lifespan(app):
    if os.getenv("PRELOAD_MODEL", "1") == "1":
        await run_in_threadpool(get_model)  # load weights once at startup
    yield


app = FastAPI(title="GeoCount API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_lock = asyncio.Lock()  # one inference at a time: protects RAM on small hosts


def get_model():
    global _model
    if _model is None:
        _model = engine.load_model(os.getenv("MODEL_WEIGHTS", engine.DEFAULT_WEIGHTS))
    return _model


@app.get("/api/health")
def health():
    return {"status": "ok", "weights": os.getenv("MODEL_WEIGHTS", engine.DEFAULT_WEIGHTS), "model_loaded": _model is not None}


@app.get("/api/classes")
def classes():
    return {"classes": query.CLASSES}


@app.post("/api/count")
async def count(
    image: UploadFile = File(...),
    question: str = Form(""),
    use_slicing: bool = Form(True),
    conf: Optional[float] = Form(None),   # omit to use the tuned per-class thresholds
    gsd: Optional[float] = Form(None),    # metres per pixel; enables length estimates
    imgsz: Optional[int] = Form(None),    # model input size; smaller helps close-up images
):
    if conf is not None and not 0.05 <= conf <= 0.95:
        raise HTTPException(422, "conf must be between 0.05 and 0.95")
    if gsd is not None and gsd <= 0:
        raise HTTPException(422, "gsd must be positive")
    if imgsz is not None and not 256 <= imgsz <= 1280:
        raise HTTPException(422, "imgsz must be between 256 and 1280")

    data = await image.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Image is larger than {MAX_UPLOAD_MB} MB.")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not read that file as an image (try JPG, PNG or TIFF).")

    h0, w0 = img.shape[:2]
    downscaled = max(h0, w0) > MAX_SIDE
    if downscaled:
        s = MAX_SIDE / max(h0, w0)
        img = cv2.resize(img, (int(w0 * s), int(h0 * s)), interpolation=cv2.INTER_AREA)
        if gsd:
            gsd = gsd / s  # each pixel now covers more ground

    class_conf = None if conf is not None else engine.TUNED_CLASS_CONF
    default_conf = conf if conf is not None else engine.TUNED_DEFAULT_CONF

    model = await run_in_threadpool(get_model)  # no-op once loaded
    t0 = time.perf_counter()
    async with _lock:
        dets = await run_in_threadpool(engine.run_inference, model, img, class_conf, default_conf, use_slicing, imgsz=imgsz or 1024)
    elapsed = time.perf_counter() - t0

    asked = query.parse_query(question)
    counts = engine.count_by_class(dets, asked)

    measurements = []
    if gsd:
        for name in counts:
            if name in engine.LENGTH_CLASSES:
                lengths = [d.long_side * gsd for d in dets if d.cls_name == name]
                if lengths:
                    measurements.append({
                        "class": name, "n": len(lengths),
                        "mean_length_m": round(statistics.mean(lengths), 1),
                        "median_length_m": round(statistics.median(lengths), 1),
                    })

    annotated = engine.annotate(img, dets, asked)
    return {
        "answer": query.format_answer(counts),
        "counts": [{"class": k, "count": v} for k, v in counts.items()],
        "total": int(sum(counts.values())),
        "matched_classes": asked,
        "measurements": measurements,
        "image": {"width": img.shape[1], "height": img.shape[0], "downscaled": downscaled},
        "settings": {"use_slicing": use_slicing, "conf": conf, "tuned_thresholds": conf is None, "imgsz": imgsz or 1024},
        "elapsed_sec": round(elapsed, 2),
        "annotated_image": engine.to_data_uri(annotated),
    }
