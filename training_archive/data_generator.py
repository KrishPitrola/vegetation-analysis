"""
data_generator.py
-----------------
Preprocessing pipeline for the PlantVillage dataset using
Keras ImageDataGenerator.

Dataset layout expected:
    PlantVillage/
    ├── train/   (38 class sub-folders)
    └── val/     (38 class sub-folders)

Usage:
    from data_generator import create_generators, GeneratorConfig

    config = GeneratorConfig()
    train_gen, val_gen, class_info = create_generators(config)
    model.fit(train_gen, validation_data=val_gen, ...)
"""

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Dict

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass — change defaults here, never edit call-sites
# ---------------------------------------------------------------------------
@dataclass
class GeneratorConfig:
    """Central configuration for the data pipeline."""

    # Paths
    train_dir: str = "./PlantVillage/train"
    val_dir:   str = "./PlantVillage/val"

    # Image dimensions
    img_height: int = 224
    img_width:  int = 224

    # Training hyperparameters
    batch_size: int = 32
    seed:       int = 42

    # Augmentation settings (training only)
    rotation_range:     float = 20.0
    zoom_range:         float = 0.15
    width_shift_range:  float = 0.10
    height_shift_range: float = 0.10
    shear_range:        float = 0.10
    horizontal_flip:    bool  = True
    fill_mode:          str   = "nearest"   # how to fill newly created pixels

    # MobileNetV2 / other ImageNet-pretrained models expect [0, 1] or
    # [-1, 1] normalisation.  rescale=1/255 maps uint8 → [0, 1].
    rescale: float = 1.0 / 255.0


# ---------------------------------------------------------------------------
# Helper — validate directories before generator creation
# ---------------------------------------------------------------------------
def _validate_directory(path: str, role: str) -> None:
    """Raise FileNotFoundError if *path* does not exist or is empty."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{role} directory not found: '{path}'. "
            "Ensure PlantVillage/train and PlantVillage/val are present."
        )
    class_dirs = [d for d in p.iterdir() if d.is_dir()]
    if not class_dirs:
        raise ValueError(
            f"{role} directory '{path}' exists but contains no class sub-folders."
        )
    logger.info("%s directory: '%s'  |  class folders: %d", role, path, len(class_dirs))


# ---------------------------------------------------------------------------
# Core factory
# ---------------------------------------------------------------------------
def create_generators(
    config: GeneratorConfig,
) -> Tuple[object, object, Dict]:
    """
    Build training and validation generators from a GeneratorConfig.

    Returns
    -------
    train_generator : DirectoryIterator
        Augmented, shuffled batches from the train split.
    val_generator : DirectoryIterator
        Non-augmented, ordered batches from the val split.
    class_info : dict
        {
          'class_names':  list[str],     # alphabetically sorted class labels
          'num_classes':  int,            # total number of classes (38)
          'train_samples': int,           # total training images
          'val_samples':   int,           # total validation images
          'class_indices': dict[str,int], # label → integer index mapping
        }
    """
    _validate_directory(config.train_dir, "Train")
    _validate_directory(config.val_dir,   "Validation")

    img_size = (config.img_height, config.img_width)

    # ------------------------------------------------------------------
    # Training ImageDataGenerator — augmentation + normalisation
    # ------------------------------------------------------------------
    train_datagen = ImageDataGenerator(
        rescale=config.rescale,
        rotation_range=config.rotation_range,
        zoom_range=config.zoom_range,
        width_shift_range=config.width_shift_range,
        height_shift_range=config.height_shift_range,
        shear_range=config.shear_range,
        horizontal_flip=config.horizontal_flip,
        fill_mode=config.fill_mode,
    )

    # ------------------------------------------------------------------
    # Validation ImageDataGenerator — normalisation only, no augmentation
    # Augmenting validation data would distort evaluation metrics.
    # ------------------------------------------------------------------
    val_datagen = ImageDataGenerator(rescale=config.rescale)

    # ------------------------------------------------------------------
    # Generators — flow_from_directory infers class labels from folder names
    # ------------------------------------------------------------------
    train_generator = train_datagen.flow_from_directory(
        directory=config.train_dir,
        target_size=img_size,
        batch_size=config.batch_size,
        class_mode="categorical",   # one-hot labels for softmax output
        shuffle=True,
        seed=config.seed,
        interpolation="bilinear",
    )

    val_generator = val_datagen.flow_from_directory(
        directory=config.val_dir,
        target_size=img_size,
        batch_size=config.batch_size,
        class_mode="categorical",
        shuffle=False,              # keep deterministic order for evaluation
        interpolation="bilinear",
    )

    # ------------------------------------------------------------------
    # Collect metadata for downstream use (model head size, class names, etc.)
    # ------------------------------------------------------------------
    class_indices: Dict[str, int] = train_generator.class_indices
    class_names = [
        cls for cls, _ in sorted(class_indices.items(), key=lambda kv: kv[1])
    ]

    class_info = {
        "class_names":   class_names,
        "num_classes":   len(class_names),
        "train_samples": train_generator.samples,
        "val_samples":   val_generator.samples,
        "class_indices": class_indices,
    }

    logger.info(
        "Pipeline ready  |  classes: %d  |  train: %d imgs  |  val: %d imgs",
        class_info["num_classes"],
        class_info["train_samples"],
        class_info["val_samples"],
    )

    return train_generator, val_generator, class_info


# ---------------------------------------------------------------------------
# Quick diagnostics — run directly to sanity-check the pipeline
# ---------------------------------------------------------------------------
def _inspect_batch(generator, label: str) -> None:
    """Fetch one batch and log shape / pixel range."""
    images, labels = next(generator)
    logger.info(
        "%s batch  |  images: %s  |  labels: %s  |  px range: [%.3f, %.3f]",
        label, images.shape, labels.shape,
        images.min(), images.max(),
    )


if __name__ == "__main__":
    config = GeneratorConfig()

    logger.info("=" * 60)
    logger.info("PlantVillage — ImageDataGenerator Pipeline")
    logger.info("=" * 60)

    try:
        train_gen, val_gen, info = create_generators(config)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Setup failed: %s", exc)
        raise SystemExit(1)

    # Print class map
    logger.info("\nClass index mapping (first 5 shown):")
    for cls, idx in list(info["class_indices"].items())[:5]:
        logger.info("  [%02d]  %s", idx, cls)
    logger.info("  ... and %d more classes", info["num_classes"] - 5)

    # Validate one batch from each generator
    _inspect_batch(train_gen, "Train")
    _inspect_batch(val_gen,   "Val  ")

    # Compute steps per epoch (needed for model.fit with generators)
    steps_per_epoch  = info["train_samples"] // config.batch_size
    validation_steps = info["val_samples"]   // config.batch_size

    logger.info("\nFit parameters:")
    logger.info("  steps_per_epoch  = %d", steps_per_epoch)
    logger.info("  validation_steps = %d", validation_steps)
    logger.info("\nAll checks passed. Ready for model.fit().")
