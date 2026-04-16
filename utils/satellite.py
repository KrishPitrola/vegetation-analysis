"""
Satellite image classification utilities.
Loads the trained Keras MobileNetV2 model (model.h5) and provides
helper functions for preprocessing and inference.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent          # project root
MODEL_PATH = _ROOT / "models" / "model.h5"
DATASET_DIR = _ROOT / "training_archive" / "eurosat"

# ─── EuroSAT class names (alphabetical, matching Keras loader order) ──────────
EUROSAT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

# ─── Vegetation type messages ─────────────────────────────────────────────────
_VEGETATION_MESSAGES: dict[str, str] = {
    "AnnualCrop": (
        "🌾 Annual Crop — Cultivated fields with seasonal crops such as wheat, corn or rice. "
        "Vegetation is dense during growing season."
    ),
    "Forest": (
        "🌲 Forest — Dense, multi-layered tree canopy. High biodiversity and carbon storage. "
        "Strong vegetation presence."
    ),
    "HerbaceousVegetation": (
        "🌿 Herbaceous Vegetation — Low-growing plants, grasses and shrubs dominate. "
        "Moderate vegetation coverage."
    ),
    "Highway": (
        "🛣️ Highway — Major road infrastructure detected. Minimal vegetation in the immediate area."
    ),
    "Industrial": (
        "🏭 Industrial Zone — Built-up industrial land use. Very limited vegetation; "
        "high impervious surface coverage."
    ),
    "Pasture": (
        "🐄 Pasture — Open grassland used for grazing. Moderate to high vegetation "
        "coverage, dominated by grasses."
    ),
    "PermanentCrop": (
        "🍇 Permanent Crop — Vineyards, orchards or other long-term crops. "
        "Structured vegetation pattern visible."
    ),
    "Residential": (
        "🏘️ Residential Area — Mixed urban land with gardens and street trees. "
        "Low to moderate scattered vegetation."
    ),
    "River": (
        "🌊 River — Water body detected. Riparian vegetation may be present along banks."
    ),
    "SeaLake": (
        "🌊 Sea / Lake — Open water surface detected. No significant vegetation present."
    ),
}

# ─── Estimated vegetation coverage per EuroSAT class ─────────────────────────
# Each entry: (display_range: str, numeric_midpoint: float)
_COVERAGE_MAP: dict[str, tuple[str, float]] = {
    "AnnualCrop":            ("40 – 70%",  55.0),
    "Forest":                ("70 – 100%", 85.0),
    "HerbaceousVegetation": ("50 – 80%",  65.0),
    "Highway":              ("0 – 10%",    5.0),
    "Industrial":           ("0 – 10%",    5.0),
    "Pasture":              ("60 – 90%",  75.0),
    "PermanentCrop":        ("40 – 70%",  55.0),
    "Residential":          ("10 – 40%",  25.0),
    "River":                ("5 – 20%",   12.5),
    "SeaLake":              ("0 – 5%",     2.5),
}


def get_estimated_coverage(predicted_class: str) -> dict:
    """
    Return an estimated vegetation coverage range and numeric midpoint
    for the given EuroSAT class label.

    Returns:
        dict with keys 'range' (str) and 'value' (float).
    """
    display_range, value = _COVERAGE_MAP.get(predicted_class, ("Unknown", 0.0))
    return {"range": display_range, "value": value}


@lru_cache(maxsize=1)
def _load_model() -> tf.keras.Model:
    """Load and cache the Keras model (called once)."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{MODEL_PATH}'. "
            "Please train the model first using train.py."
        )
    logger.info(f"Loading satellite model from {MODEL_PATH} …")
    model = tf.keras.models.load_model(str(MODEL_PATH))
    logger.info("Satellite model loaded successfully.")
    return model


def _infer_class_names() -> list[str]:
    """Dynamically read class names from the dataset folder (fallback: hardcoded list)."""
    if DATASET_DIR.exists():
        names = sorted([
            d.name for d in DATASET_DIR.iterdir()
            if d.is_dir()
        ])
        if names:
            return names
    return EUROSAT_CLASSES


def preprocess_image(img_path: str, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Load an image from disk, resize to target_size, normalise to [0, 1],
    and expand dims to (1, H, W, 3) ready for model inference.
    """
    img = tf.keras.utils.load_img(img_path, color_mode="rgb", target_size=target_size)
    arr = tf.keras.utils.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def predict_satellite(img_tensor: np.ndarray) -> tuple[str, float]:
    """
    Run inference with the trained classifier.

    Returns:
        (predicted_class_name, confidence_score_0_to_1)
    """
    model = _load_model()
    class_names = _infer_class_names()

    probs = model.predict(img_tensor, verbose=0)[0]
    idx = int(np.argmax(probs))
    confidence = float(probs[idx])
    label = class_names[idx] if idx < len(class_names) else f"Class_{idx}"
    return label, confidence


def get_vegetation_message(predicted_class: str) -> str:
    """Return a human-readable description for the predicted EuroSAT class."""
    return _VEGETATION_MESSAGES.get(
        predicted_class,
        f"Land-cover type: {predicted_class}. No detailed description available.",
    )
