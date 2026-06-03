import base64
import io
import time
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

MODEL_PATH = Path("outputs/best_model.keras")
IMG_SIZE = (224, 224)
CONF_THRESHOLD = 0.60

# class_indices.json kullanmıyoruz
# Sıralama modelin eğitim sırasıyla aynı olmalı.
# flow_from_dataframe alfabetik sıra kullandıysa bu liste doğru olmalı.
CLASS_NAMES = [
    "alaxan",
    "bactidol",
    "bioflu",
    "biogesic",
    "dayzinc",
    "decolgen",
    "fish oil",
    "kemil s",
    "medicol",
    "neozep",
]

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

app = FastAPI(title="Drug Vision")

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

templates = Jinja2Templates(directory="templates")


def pil_to_base64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def preprocess_image(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_resized = img.resize(IMG_SIZE)

    arr = np.array(img_resized).astype(np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    return img_resized, arr


def read_text_file(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return "File not found."


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": None,
        },
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        processed_pil, x = preprocess_image(image_bytes)

        t0 = time.perf_counter()
        probs = model.predict(x, verbose=0)[0]
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        pred_label = CLASS_NAMES[pred_idx]

        if confidence < CONF_THRESHOLD:
            pred_label = "Emin değilim (Low confidence)"

        top3_idx = np.argsort(probs)[-3:][::-1]
        top3_preds = [
            {
                "label": CLASS_NAMES[i],
                "confidence": round(float(probs[i]) * 100, 2),
            }
            for i in top3_idx
        ]

        result = {
            "image_b64": pil_to_base64_png(processed_pil),
            "pred_label": pred_label,
            "pred_idx": pred_idx,
            "confidence_pct": round(confidence * 100, 2),
            "bar_width": max(0.0, min(100.0, confidence * 100)),
            "latency_ms": round(latency_ms, 1),
            "top3_preds": top3_preds,
        }

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": result,
            },
        )

    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": None,
                "error": str(e),
            },
        )


@app.get("/metrics", response_class=HTMLResponse)
async def metrics_page(request: Request):
    metrics = read_text_file(Path("outputs/test_metrics.txt"))
    topk = read_text_file(Path("outputs/topk_metrics.txt"))
    report = read_text_file(Path("outputs/classification_report.txt"))
    hard_cases = read_text_file(Path("outputs/hard_cases.txt"))

    figures = [
        {"title": "Class Distribution", "path": "/outputs/figures/class_distribution.png"},
        {"title": "Sample Grid", "path": "/outputs/figures/sample_grid_16.png"},
        {"title": "Training Curves", "path": "/outputs/figures/training_curves.png"},
        {"title": "Confusion Matrix", "path": "/outputs/figures/confusion_matrix.png"},
    ]

    return templates.TemplateResponse(
        "metrics.html",
        {
            "request": request,
            "metrics": metrics,
            "topk": topk,
            "report": report,
            "hard_cases": hard_cases,
            "figures": figures,
        },
    )