"""
model.py
--------
MobileNetV2 transfer learning model for PlantVillage 38-class classification.

Two-phase training strategy
────────────────────────────
  Phase 1 — Feature extraction (this file)
    Base model frozen → only the custom head trains.
    Converges in ~10 epochs, stable gradients, no risk of destroying
    pretrained weights.

  Phase 2 — Fine-tuning (optional, via unfreeze_top_layers)
    Unfreeze the top N layers of the base → very low LR refines features
    already learned by the head.  Run AFTER Phase 1 completes.

Usage:
    from model import ModelConfig, build_model, unfreeze_top_layers, compile_model
    from data_generator import GeneratorConfig, create_generators

    gen_cfg   = GeneratorConfig()
    train_gen, val_gen, info = create_generators(gen_cfg)

    cfg   = ModelConfig(num_classes=info["num_classes"])
    model = build_model(cfg)
    model.summary()
    model.fit(train_gen, ...)
"""

import logging
from dataclasses import dataclass
from typing import Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2

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
# Configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """All model hyperparameters in one place."""

    # Architecture
    num_classes:  int   = 38
    input_shape:  tuple = (224, 224, 3)
    pooling:      str   = "avg"          # GlobalAveragePooling2D

    # Custom head
    dense_units:  int   = 128
    dropout_rate: float = 0.4            # 0.3–0.5 range per requirement

    # Phase-1 compilation (frozen base)
    learning_rate: float = 1e-3          # Adam default — safe for frozen base
    optimizer:     str   = "adam"

    # Phase-2 fine-tuning
    fine_tune_lr:  float = 1e-5          # 100× smaller — prevents catastrophic forgetting
    fine_tune_layers: int = 30           # unfreeze last 30 layers of base model


# ---------------------------------------------------------------------------
# Core builder — functional API for clarity and inspectability
# ---------------------------------------------------------------------------
def build_model(cfg: ModelConfig) -> Model:
    """
    Build a MobileNetV2 transfer learning model.

    Architecture
    ────────────
    Input (224×224×3)
      └─ MobileNetV2 backbone (frozen, ImageNet weights, no top)
           └─ GlobalAveragePooling2D        →  1280-d feature vector
                └─ BatchNormalization       →  stabilises head training
                     └─ Dense(128, ReLU)    →  task-specific features
                          └─ Dropout(0.4)   →  regularisation
                               └─ Dense(38, softmax)  →  class probabilities

    Why GlobalAveragePooling instead of Flatten?
      GAP reduces each 7×7×1280 feature map to a single 1280-d vector,
      giving spatial invariance and far fewer parameters than Flatten
      (7×7×1280 = 62,720 inputs vs 1,280), which reduces overfitting.

    Returns
    -------
    model : keras.Model  (compiled, ready for model.fit)
    """
    logger.info(
        "Building MobileNetV2 model  |  classes: %d  |  input: %s",
        cfg.num_classes, cfg.input_shape,
    )

    # ------------------------------------------------------------------
    # 1. Backbone — pretrained MobileNetV2, top layer excluded
    # ------------------------------------------------------------------
    base_model = MobileNetV2(
        input_shape=cfg.input_shape,
        include_top=False,          # remove ImageNet 1000-class head
        weights="imagenet",         # load pretrained feature extractor
        pooling=None,               # we add our own GAP below
    )

    # Freeze ALL base layers for Phase 1.
    # ┌─────────────────────────────────────────────────────────────────┐
    # │  WHY FREEZE?                                                    │
    # │                                                                 │
    # │  MobileNetV2 was trained on 1.28M ImageNet images.  Its early  │
    # │  layers detect universal features: edges, textures, colours.   │
    # │  These transfer perfectly to plant images.                      │
    # │                                                                 │
    # │  If we allow gradients to flow into the base immediately:       │
    # │   • The randomly-initialised head produces large, noisy         │
    # │     gradients that corrupt the pretrained weights.              │
    # │   • This is called "catastrophic forgetting".                   │
    # │                                                                 │
    # │  Freezing (trainable=False) means those weights are treated     │
    # │  as constants — only the head parameters are updated.           │
    # │  The head converges first, then we optionally unfreeze the top  │
    # │  N base layers (Phase 2) with a 100× smaller LR to refine      │
    # │  high-level features for plant-specific patterns.               │
    # └─────────────────────────────────────────────────────────────────┘
    base_model.trainable = False

    trainable_count   = sum(1 for l in base_model.layers if l.trainable)
    non_trainable     = len(base_model.layers) - trainable_count
    logger.info(
        "Base model: %d total layers | %d trainable | %d frozen",
        len(base_model.layers), trainable_count, non_trainable,
    )

    # ------------------------------------------------------------------
    # 2. Custom classification head — Functional API
    # ------------------------------------------------------------------
    inputs = keras.Input(shape=cfg.input_shape, name="input_image")

    # Pass through frozen base (training=False keeps BatchNorm in inference
    # mode even when model.fit runs — critical for frozen BN layers)
    x = base_model(inputs, training=False)

    # Spatial pooling — collapses (7, 7, 1280) → (1280,)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    # Stabilise activations before the dense block
    x = layers.BatchNormalization(name="head_bn")(x)

    # Task-specific feature transformation
    x = layers.Dense(cfg.dense_units, activation="relu", name="head_dense")(x)

    # Regularisation — randomly zero 40% of activations during training
    x = layers.Dropout(cfg.dropout_rate, name="head_dropout")(x)

    # Output — one probability per class; softmax sums to 1.0
    outputs = layers.Dense(
        cfg.num_classes,
        activation="softmax",
        name="predictions",
    )(x)

    model = keras.Model(inputs, outputs, name="PlantVillage_MobileNetV2")

    # ------------------------------------------------------------------
    # 3. Compile — Phase 1 settings
    # ------------------------------------------------------------------
    model = compile_model(model, learning_rate=cfg.learning_rate)

    return model


# ---------------------------------------------------------------------------
# Compile helper — reused for both phases
# ---------------------------------------------------------------------------
def compile_model(
    model: Model,
    learning_rate: float = 1e-3,
    label_smoothing: float = 0.1,
) -> Model:
    """
    Compile *model* with Adam + categorical cross-entropy + accuracy.

    Label smoothing (0.1) softens one-hot targets from [0,1] to [0.05, 0.95],
    which prevents the model from becoming overconfident and improves
    generalisation on the 38-class head.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_accuracy"),
        ],
    )
    logger.info("Model compiled  |  lr: %.0e  |  loss: CategoricalCE (smoothing=%.1f)",
                learning_rate, label_smoothing)
    return model


# ---------------------------------------------------------------------------
# Phase 2 — Fine-tuning helper
# ---------------------------------------------------------------------------
def unfreeze_top_layers(model: Model, cfg: ModelConfig) -> Model:
    """
    Unfreeze the last *cfg.fine_tune_layers* layers of the MobileNetV2 base
    and recompile with a much smaller learning rate for Phase 2 fine-tuning.

    Call this ONLY after Phase 1 training has converged.

    Example
    -------
        history_phase1 = model.fit(train_gen, epochs=10, ...)

        model = unfreeze_top_layers(model, cfg)
        history_phase2 = model.fit(train_gen, epochs=10, ...)
    """
    # Locate the base model sub-layer by name
    base_model = model.get_layer("mobilenetv2_1.00_224")

    base_model.trainable = True

    # Re-freeze everything EXCEPT the last N layers
    freeze_until = len(base_model.layers) - cfg.fine_tune_layers
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False

    newly_trainable = sum(1 for l in base_model.layers if l.trainable)
    logger.info(
        "Fine-tune: unfroze last %d base layers  |  trainable base layers: %d / %d",
        cfg.fine_tune_layers, newly_trainable, len(base_model.layers),
    )

    # Recompile at a much smaller LR — 100× less than Phase 1
    model = compile_model(model, learning_rate=cfg.fine_tune_lr)
    return model


# ---------------------------------------------------------------------------
# Trainable parameter counter utility
# ---------------------------------------------------------------------------
def print_param_summary(model: Model) -> None:
    """Log trainable vs non-trainable parameter counts."""
    trainable     = sum(tf.size(w).numpy() for w in model.trainable_weights)
    non_trainable = sum(tf.size(w).numpy() for w in model.non_trainable_weights)
    total         = trainable + non_trainable
    logger.info(
        "Parameters  |  total: %s  |  trainable: %s (%.1f%%)  |  frozen: %s",
        f"{total:,}", f"{trainable:,}", 100 * trainable / total, f"{non_trainable:,}",
    )


# ---------------------------------------------------------------------------
# Quick smoke-test — run directly to validate architecture
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("PlantVillage — MobileNetV2 Model Builder")
    logger.info("=" * 60)

    cfg = ModelConfig(num_classes=38)

    # ── Build ────────────────────────────────────────────────────────────
    model = build_model(cfg)

    # ── Summary ──────────────────────────────────────────────────────────
    model.summary(line_length=90, show_trainable=True)
    print()
    print_param_summary(model)

    # ── Forward pass sanity-check ────────────────────────────────────────
    import numpy as np
    dummy_batch = np.random.rand(4, 224, 224, 3).astype("float32")
    preds = model.predict(dummy_batch, verbose=0)

    logger.info("\nForward pass check:")
    logger.info("  Input  shape : %s", dummy_batch.shape)
    logger.info("  Output shape : %s", preds.shape)         # (4, 38)
    logger.info("  Prob sum (batch[0]): %.6f", preds[0].sum())  # must be ≈ 1.0
    logger.info("  Top-1 class (batch[0]): %d", preds[0].argmax())

    assert preds.shape == (4, 38), "Unexpected output shape!"
    assert abs(preds[0].sum() - 1.0) < 1e-5, "Softmax probabilities do not sum to 1!"

    logger.info("\nAll assertions passed. Model is ready for training.")

    # ── Show fine-tune config (no actual unfreeze in smoke test) ─────────
    logger.info("\nFine-tuning config (Phase 2):")
    logger.info("  Layers to unfreeze : last %d of base", cfg.fine_tune_layers)
    logger.info("  Fine-tune LR       : %.0e", cfg.fine_tune_lr)
    logger.info("  Call unfreeze_top_layers(model, cfg) after Phase 1 converges.")
