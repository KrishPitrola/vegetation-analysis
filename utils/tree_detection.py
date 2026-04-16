"""
Tree detection utilities using the DeepForest pretrained model.
Provides confidence thresholding, small-box filtering,
OpenCV bounding-box drawing, vegetation coverage computation
and vegetation level classification.
"""
from __future__ import annotations

import logging
from functools import lru_cache

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── Thresholds ───────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.25
MIN_BOX_DIM = 10  # pixels — boxes smaller than this are discarded

# ─── Vegetation level thresholds ─────────────────────────────────────────────
LEVEL_LOW_MAX = 10
LEVEL_MOD_MAX = 30


@lru_cache(maxsize=1)
def _load_deepforest_model():
    """Load and cache the DeepForest pretrained model (called once)."""
    try:
        from deepforest import main as df_main
        logger.info("Loading DeepForest pretrained model …")
        model = df_main.deepforest()
        model.load_model()
        logger.info("DeepForest model loaded successfully.")
        return model
    except ImportError as exc:
        raise ImportError(
            "DeepForest is not installed. Run: pip install deepforest"
        ) from exc


def detect_trees(image_path: str) -> pd.DataFrame:
    """
    Run DeepForest prediction on *image_path*, apply confidence threshold
    and minimum bounding-box size filter.

    Returns:
        Filtered DataFrame with columns [xmin, ymin, xmax, ymax, score, label].
    """
    model = _load_deepforest_model()

    logger.info(f"[DeepForest] Running prediction on {image_path}")
    predictions: pd.DataFrame = model.predict_image(path=image_path)

    if predictions is None or predictions.empty:
        logger.warning("[DeepForest] No predictions returned.")
        return pd.DataFrame(columns=["xmin", "ymin", "xmax", "ymax", "score", "label"])

    # 1. Confidence filter
    predictions = predictions[predictions["score"] > CONFIDENCE_THRESHOLD]

    # 2. Minimum box size filter
    predictions = predictions[
        (predictions["xmax"] - predictions["xmin"] > MIN_BOX_DIM)
        & (predictions["ymax"] - predictions["ymin"] > MIN_BOX_DIM)
    ]

    predictions = predictions.reset_index(drop=True)
    logger.info(f"[DeepForest] {len(predictions)} trees detected after filtering.")
    return predictions


def compute_coverage(predictions: pd.DataFrame, image: np.ndarray) -> float:
    """
    Estimate vegetation coverage as a percentage of total image area covered
    by bounding boxes (approximation — does NOT use segmentation).

    Args:
        predictions: Filtered DataFrame with columns [xmin, ymin, xmax, ymax].
        image:       Source image as a NumPy array (H x W x C).

    Returns:
        Coverage percentage rounded to 2 decimal places (0.00 – 100.00).
    """
    if predictions is None or predictions.empty:
        return 0.0

    h, w = image.shape[:2]
    image_area = float(h * w)
    if image_area == 0:
        return 0.0

    tree_area = 0.0
    for _, row in predictions.iterrows():
        box_w = float(row["xmax"]) - float(row["xmin"])
        box_h = float(row["ymax"]) - float(row["ymin"])
        tree_area += max(0.0, box_w) * max(0.0, box_h)

    coverage = (tree_area / image_area) * 100.0
    return round(min(coverage, 100.0), 2)


def classify_vegetation_level(tree_count: int) -> str:
    """
    Classify vegetation density based on detected tree count.

    < 10  → Low
    10–30 → Moderate
    > 30  → High
    """
    if tree_count < LEVEL_LOW_MAX:
        return "Low"
    elif tree_count <= LEVEL_MOD_MAX:
        return "Moderate"
    else:
        return "High"


def draw_boxes(image_path: str, predictions: pd.DataFrame) -> tuple[np.ndarray, int, str, float]:
    """
    Load the image with OpenCV and draw bright-green bounding boxes
    with confidence text overlays for each detected tree.
    Also overlays a summary banner showing tree count and coverage %.

    Returns:
        tuple: (annotated_bgr_array, tree_count, vegetation_level, coverage_percentage)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"OpenCV could not read image at: {image_path}")

    h, w = image.shape[:2]

    for _, row in predictions.iterrows():
        xmin = max(0, int(row["xmin"]))
        ymin = max(0, int(row["ymin"]))
        xmax = min(w, int(row["xmax"]))
        ymax = min(h, int(row["ymax"]))
        score = float(row.get("score", 0.0))

        # Draw box
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 220, 80), 2)

        # Draw score label
        label_text = f"{score:.2f}"
        font_scale = 0.4
        label_y = max(ymin - 5, 12)
        cv2.putText(
            image,
            label_text,
            (xmin, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 220, 80),
            1,
            cv2.LINE_AA,
        )

    tree_count = len(predictions)
    coverage = compute_coverage(predictions, image)
    level = classify_vegetation_level(tree_count)

    # Overlay summary banner (dark semi-transparent strip at top)
    banner_h = 42
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, image, 0.35, 0, image)

    summary = f"Trees: {tree_count}   |   Coverage: {coverage:.2f}%   |   Level: {level}"
    cv2.putText(
        image,
        summary,
        (10, 28),
        cv2.FONT_HERSHEY_DUPLEX,
        0.58,
        (0, 220, 80),
        1,
        cv2.LINE_AA,
    )

    return image, tree_count, level, coverage
