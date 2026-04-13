import os
import io
import base64
import logging
import tempfile
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from utils.satellite import preprocess_image, predict_satellite, get_vegetation_message, get_estimated_coverage
from utils.tree_detection import detect_trees, draw_boxes, classify_vegetation_level, compute_coverage

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Vegetation Analysis System",
    description="AI-powered satellite image classification and tree detection",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Serve frontend ───────────────────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")

# ─── Allowed extensions ───────────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def validate_image(file: UploadFile) -> None:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )


# ─── Satellite endpoint ───────────────────────────────────────────────────────
@app.post("/predict-satellite")
async def predict_satellite_endpoint(file: UploadFile = File(...)):
    """
    Accepts an image upload, runs the Keras EuroSAT classifier,
    and returns the predicted class, confidence and a descriptive message.
    """
    validate_image(file)

    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Write to a temp file so Keras can load it
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        try:
            img_tensor = preprocess_image(tmp_path)
            predicted_class, confidence = predict_satellite(img_tensor)
        finally:
            os.unlink(tmp_path)

        vegetation_msg = get_vegetation_message(predicted_class)
        estimated_coverage = get_estimated_coverage(predicted_class)

        logger.info(f"[Satellite] class={predicted_class}  conf={confidence:.4f}  est_coverage={estimated_coverage['value']}%")
        return JSONResponse({
            "status": "success",
            "mode": "satellite",
            "predicted_class": predicted_class,
            "confidence": round(confidence * 100, 2),
            "vegetation_message": vegetation_msg,
            "estimated_coverage": estimated_coverage["value"],
            "estimated_coverage_range": estimated_coverage["range"],
        })

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"[Satellite] Error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ─── Tree detection endpoint ──────────────────────────────────────────────────
@app.post("/predict-trees")
async def predict_trees_endpoint(file: UploadFile = File(...)):
    """
    Accepts an image upload, runs DeepForest tree detection,
    draws bounding boxes and returns the annotated image + statistics.
    """
    validate_image(file)

    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Write to a temp file (DeepForest needs a path)
        suffix = Path(file.filename or "img.jpg").suffix or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        try:
            predictions = detect_trees(tmp_path)
            annotated_bgr = draw_boxes(tmp_path, predictions)
        finally:
            os.unlink(tmp_path)

        tree_count = len(predictions)
        vegetation_level = classify_vegetation_level(tree_count)
        coverage_percentage = compute_coverage(predictions, annotated_bgr)

        # Encode annotated image as base64 PNG
        _, buffer = cv2.imencode(".png", annotated_bgr)
        img_b64 = base64.b64encode(buffer).decode("utf-8")

        logger.info(f"[Trees] count={tree_count}  level={vegetation_level}  coverage={coverage_percentage}%")
        return JSONResponse({
            "status": "success",
            "mode": "tree_detection",
            "tree_count": tree_count,
            "vegetation_level": vegetation_level,
            "coverage_percentage": coverage_percentage,
            "annotated_image": img_b64,
        })

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"[Trees] Error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ─── Health check ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


# ─── Auto-detect endpoint (bonus) ─────────────────────────────────────────────
@app.post("/predict-auto")
async def predict_auto_endpoint(file: UploadFile = File(...)):
    """
    Heuristic: if the image looks small / coarse → satellite; else → tree detection.
    Uses object-density metric on the image to decide.
    """
    validate_image(file)

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Cannot decode image.")

        # Heuristic: high-frequency content → likely an aerial/ground-level photo of trees
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        mode = "tree_detection" if lap_var > 400 else "satellite"

        return JSONResponse({"status": "success", "detected_mode": mode})

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"[Auto] Error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
