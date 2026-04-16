import os
import sys
import base64
import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure project root is on sys.path so we can import predict.py / model.py
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from utils.satellite import preprocess_image, predict_satellite, get_vegetation_message, get_estimated_coverage
from utils.tree_detection import detect_trees, draw_boxes
from utils.predict import load_predictor, predict as leaf_predict

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ─── Leaf predictor — loaded once at startup, shared via app.state ────────────
_LEAF_MODEL_PATH  = str(Path(__file__).resolve().parent.parent / "models" / "leaf_model.keras")
_LEAF_LABELS_DIR  = str(Path(__file__).resolve().parent.parent / "training_archive" / "PlantVillage" / "train")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the leaf predictor once on startup; release on shutdown."""
    logger.info("[Startup] Loading leaf predictor from '%s'...", _LEAF_MODEL_PATH)
    app.state.leaf_predictor = load_predictor(
        model_path=_LEAF_MODEL_PATH,
        labels_dir=_LEAF_LABELS_DIR,
    )
    logger.info("[Startup] Leaf predictor ready (%d classes).",
                app.state.leaf_predictor.num_classes)
    yield
    # Nothing to clean up for TF models
    logger.info("[Shutdown] Leaf predictor released.")


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Vegetation Analysis System",
    description="AI-powered satellite image classification and tree detection",
    version="1.0.0",
    lifespan=lifespan,
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
            annotated_bgr, tree_count, vegetation_level, coverage_percentage = draw_boxes(tmp_path, predictions)
        finally:
            os.unlink(tmp_path)

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


# ─── Leaf classification endpoint ────────────────────────────────────────────
@app.post("/predict-leaf")
async def predict_leaf_endpoint(file: UploadFile = File(...)):
    """
    POST /predict-leaf
    ------------------
    Accept a leaf image upload, run the MobileNetV2 PlantVillage classifier,
    and return the predicted species/disease label with confidence.

    Request : multipart/form-data  field name = "file"
    Response:
        {
          "species":    "Apple___Apple_scab",
          "confidence": 0.9646
        }
    """
    validate_image(file)

    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Save to a temp file — predict.preprocess() needs a file path
        suffix = Path(file.filename or "leaf.jpg").suffix.lower() or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        try:
            result = leaf_predict(
                image_path=tmp_path,
                predictor=app.state.leaf_predictor,
                top_n=5,
            )
        finally:
            os.unlink(tmp_path)   # always clean up, even on error

        logger.info(
            "[Leaf] file=%s  species=%s  confidence=%.4f",
            file.filename, result["label"], result["confidence"],
        )
        return JSONResponse({
            "species":    result["label"],
            "confidence": round(result["confidence"], 4),
        })

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[Leaf] Prediction error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))



