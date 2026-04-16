"""
predict.py
----------
Single-image inference for the PlantVillage MobileNetV2 classifier.

Preprocessing mirrors data_generator.py exactly:
  - Resize  : 224 x 224 (bilinear)
  - Rescale : pixel / 255.0  ->  [0.0, 1.0]
  - No augmentation (inference-time)

Usage — CLI:
    python predict.py path/to/leaf.jpg
    python predict.py path/to/leaf.jpg --model leaf_model.keras --top 5

Usage — module (e.g. from backend/app.py):
    from predict import load_predictor, predict

    predictor = load_predictor()          # loads model + labels once
    result    = predict("leaf.jpg", predictor)
    print(result["label"], result["confidence"])
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# Defaults — keep in sync with data_generator.py GeneratorConfig
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH  = "./models/leaf_model.keras"
DEFAULT_LABELS_DIR  = "./training_archive/PlantVillage/train"   # class names read from sub-folders
IMG_SIZE            = (224, 224)               # must match training target_size
RESCALE             = 1.0 / 255.0             # must match data_generator rescale


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------
def load_class_names(labels_dir: str = DEFAULT_LABELS_DIR) -> list[str]:
    """
    Return a sorted list of class names inferred from the dataset folder.

    Matches the alphabetical ordering used by:
        ImageDataGenerator.flow_from_directory(...)
        tf.keras.utils.image_dataset_from_directory(...)

    Raises FileNotFoundError if the directory is missing.
    """
    p = Path(labels_dir)
    if not p.exists():
        raise FileNotFoundError(
            f"Labels directory not found: '{labels_dir}'\n"
            "Set --dataset to your PlantVillage/train path."
        )
    classes = sorted([d.name for d in p.iterdir() if d.is_dir()])
    if not classes:
        raise ValueError(f"No sub-folders found in '{labels_dir}'.")
    return classes


# ---------------------------------------------------------------------------
# Preprocessing — must be identical to validation generator (no augmentation)
# ---------------------------------------------------------------------------
def preprocess(image_path: str) -> np.ndarray:
    """
    Load and preprocess one image for inference.

    Steps
    -----
    1. Load as RGB (3 channels) — handles PNG, JPEG, BMP, TIFF
    2. Resize to 224×224 using bilinear interpolation
       (same interpolation= kwarg used in flow_from_directory)
    3. Cast to float32 and scale to [0, 1] by dividing by 255
    4. Add batch dimension → shape (1, 224, 224, 3)

    Returns
    -------
    np.ndarray of shape (1, 224, 224, 3), dtype float32
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: '{image_path}'")
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}:
        raise ValueError(f"Unsupported image format: '{path.suffix}'")

    img       = tf.keras.utils.load_img(str(path), color_mode="rgb", target_size=IMG_SIZE)
    arr       = tf.keras.utils.img_to_array(img)   # uint8 HxWx3 → float32 HxWx3
    arr       = arr * RESCALE                       # [0, 255] → [0.0, 1.0]
    arr       = np.expand_dims(arr, axis=0)         # → (1, 224, 224, 3)
    return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# Predictor bundle — load once, reuse across calls (important for the backend)
# ---------------------------------------------------------------------------
class Predictor:
    """Holds a loaded model + class list so both are initialised only once."""

    def __init__(self, model_path: str, labels_dir: str):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model not found: '{model_path}'\n"
                "Train first (train_model.py) or confirm the path."
            )
        # Strategy: rebuild architecture from model.py, then load weights only.
        # This bypasses Keras version mismatches where the saved .keras file
        # contains keys (e.g. quantization_config in Dense) that the currently
        # installed Keras version does not recognise during deserialization.
        from utils.model import ModelConfig, build_model
        import warnings
        cfg        = ModelConfig(num_classes=38)
        self.model = build_model(cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # suppress optimizer-skip warning (inference only)
            self.model.load_weights(model_path)
        self.class_names  = load_class_names(labels_dir)
        self.num_classes  = len(self.class_names)

        # Sanity-check output shape matches number of labels
        out_units = self.model.output_shape[-1]
        if out_units != self.num_classes:
            raise ValueError(
                f"Model output units ({out_units}) != "
                f"number of label folders ({self.num_classes}). "
                "Ensure --dataset points to the correct PlantVillage/train directory."
            )


def load_predictor(
    model_path: str = DEFAULT_MODEL_PATH,
    labels_dir: str = DEFAULT_LABELS_DIR,
) -> Predictor:
    """Convenience factory used by external modules (e.g. the FastAPI backend)."""
    return Predictor(model_path=model_path, labels_dir=labels_dir)


# ---------------------------------------------------------------------------
# Core prediction function
# ---------------------------------------------------------------------------
def predict(
    image_path: str,
    predictor: Predictor,
    top_n: int = 5,
) -> dict:
    """
    Run inference on a single image.

    Parameters
    ----------
    image_path : str
        Path to the input leaf image.
    predictor  : Predictor
        A loaded Predictor instance (model + class names).
    top_n      : int
        Number of top predictions to return (default 5).

    Returns
    -------
    dict with keys:
        label        (str)         - top-1 predicted class name
        confidence   (float)       - top-1 probability [0.0, 1.0]
        top_n        (list[dict])  - ranked list of {label, confidence}
        image_path   (str)         - echoed input path
    """
    tensor = preprocess(image_path)                          # (1, 224, 224, 3)
    probs  = predictor.model.predict(tensor, verbose=0)[0]  # (38,)

    # Top-N indices sorted by probability descending
    top_indices = np.argsort(probs)[::-1][:top_n]

    top_predictions = [
        {
            "label":      predictor.class_names[i],
            "confidence": float(probs[i]),
        }
        for i in top_indices
    ]

    return {
        "label":      top_predictions[0]["label"],
        "confidence": top_predictions[0]["confidence"],
        "top_n":      top_predictions,
        "image_path": str(image_path),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _print_result(result: dict) -> None:
    width = 48
    print("\n" + "=" * width)
    print("  PLANT DISEASE PREDICTION RESULT")
    print("=" * width)
    print(f"  Image      : {Path(result['image_path']).name}")
    print(f"  Prediction : {result['label']}")
    print(f"  Confidence : {result['confidence']:.4f}  ({result['confidence']*100:.2f}%)")
    print("-" * width)
    print("  Top predictions:")
    for rank, pred in enumerate(result["top_n"], start=1):
        bar_len = int(pred["confidence"] * 20)
        bar     = "#" * bar_len + "-" * (20 - bar_len)
        print(f"  {rank}. [{bar}] {pred['confidence']*100:5.2f}%  {pred['label']}")
    print("=" * width + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict plant disease from a leaf image using the trained MobileNetV2 model."
    )
    parser.add_argument(
        "image_path",
        help="Path to the input image (JPEG / PNG).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the saved Keras model (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_LABELS_DIR,
        help=f"Path to PlantVillage/train for class label inference (default: {DEFAULT_LABELS_DIR}).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of top predictions to display (default: 5).",
    )
    args = parser.parse_args()

    try:
        predictor = load_predictor(model_path=args.model, labels_dir=args.dataset)
        result    = predict(args.image_path, predictor, top_n=args.top)
        _print_result(result)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[ERROR] {exc}\n")
        raise SystemExit(1)
