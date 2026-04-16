"""
train_model.py
--------------
End-to-end training script for PlantVillage 38-class classification.
Integrates data_generator.py and model.py into a single executable pipeline.

Two-phase strategy
──────────────────
  Phase 1 (here)  — Frozen base, train head only (~15 epochs)
  Phase 2 (bonus) — Unfreeze top base layers, fine-tune at 1e-5 LR

Run:
    python train_model.py

Outputs:
    checkpoints/best_model.keras   ← best val_accuracy checkpoint
    logs/training_curves.png       ← loss & accuracy curves (overfitting guide)
    logs/training_history.csv      ← epoch-by-epoch metrics
"""

import csv
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for all environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import tensorflow as tf
from tensorflow import keras

from data_generator import GeneratorConfig, create_generators
from model import ModelConfig, build_model, compile_model, unfreeze_top_layers

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
# Training configuration — single source of truth
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """All training hyperparameters."""

    # Epochs
    phase1_epochs: int = 15      # frozen base; 15–20 as requested
    phase2_epochs: int = 10      # optional fine-tuning after phase 1
    run_phase2:    bool = False  # set True to run fine-tuning automatically

    # Regularisation
    early_stop_patience: int = 3           # stop after 3 epochs of no val improvement
    early_stop_monitor:  str = "val_loss"  # watch val_loss (more stable than val_accuracy)
    reduce_lr_patience:  int = 2           # halve LR if val_loss stalls for 2 epochs
    reduce_lr_factor:    float = 0.5
    reduce_lr_min:       float = 1e-7

    # Paths
    checkpoint_dir: str = "./checkpoints"
    log_dir:        str = "./logs"
    best_model_name: str = "best_model.keras"  # SavedModel format (recommended over .h5)


# ---------------------------------------------------------------------------
# Callback factory
# ---------------------------------------------------------------------------
def build_callbacks(cfg: TrainConfig, phase: int = 1) -> List[keras.callbacks.Callback]:
    """
    Construct the callback stack:

    EarlyStopping
      Stops training once val_loss stops improving for `patience` epochs,
      then restores the best weights seen — so you always get the best model
      regardless of when training stops.  patience=3 is aggressive (fast
      feedback) but fine for transfer learning where convergence is fast.

    ModelCheckpoint
      Saves the full model (weights + architecture + optimizer state) only
      when val_accuracy improves.  Saving only the best avoids bloated disk
      usage over many epochs.

    ReduceLROnPlateau
      Halves the learning rate if val_loss doesn't improve for 2 epochs.
      Works synergistically with EarlyStopping — the LR reduction often
      rescues a plateau before early stopping triggers.

    TensorBoard (optional)
      Enable by passing --tensorboard flag or by keeping the callback.
    """
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    checkpoint_path = os.path.join(cfg.checkpoint_dir, cfg.best_model_name)

    callbacks = [
        # ── 1. Early Stopping ─────────────────────────────────────────────
        keras.callbacks.EarlyStopping(
            monitor=cfg.early_stop_monitor,
            patience=cfg.early_stop_patience,
            restore_best_weights=True,   # ← crucial: reverts to best epoch weights
            verbose=1,
            mode="min",
        ),

        # ── 2. Model Checkpoint ───────────────────────────────────────────
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,         # only overwrite if val_accuracy improves
            save_weights_only=False,     # save full model (architecture + weights)
            verbose=1,
            mode="max",
        ),

        # ── 3. Learning Rate Reduction on Plateau ─────────────────────────
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=cfg.reduce_lr_factor,
            patience=cfg.reduce_lr_patience,
            min_lr=cfg.reduce_lr_min,
            verbose=1,
            mode="min",
        ),

        # ── 4. CSV Logger — full epoch-by-epoch history ───────────────────
        keras.callbacks.CSVLogger(
            filename=os.path.join(cfg.log_dir, f"phase{phase}_history.csv"),
            separator=",",
            append=False,
        ),
    ]

    logger.info(
        "Callbacks ready  |  checkpoint: %s  |  early stop patience: %d",
        checkpoint_path, cfg.early_stop_patience,
    )
    return callbacks


# ---------------------------------------------------------------------------
# Overfitting analyser — called after each phase
# ---------------------------------------------------------------------------
def analyse_overfitting(history: keras.callbacks.History, phase: int = 1) -> Dict:
    """
    Programmatically detect overfitting and print a diagnosis.

    Overfitting signatures
    ──────────────────────
    1. Gap widening  — val_accuracy plateaus/falls while train_accuracy rises
    2. Loss divergence — train_loss keeps falling but val_loss rises
    3. Gap threshold  — if (train_acc - val_acc) > 0.10 at the last epoch,
                        generalisation is likely suffering

    Returns a dict of summary statistics for logging/reporting.
    """
    h = history.history

    train_acc = h.get("accuracy", [])
    val_acc   = h.get("val_accuracy", [])
    train_loss = h.get("loss", [])
    val_loss   = h.get("val_loss", [])

    if not train_acc:
        return {}

    best_val_acc   = max(val_acc)
    best_val_epoch = val_acc.index(best_val_acc) + 1
    final_train    = train_acc[-1]
    final_val      = val_acc[-1]
    gap            = final_train - final_val

    # Trend: is val_accuracy decreasing in the last 3 epochs?
    declining_val = len(val_acc) >= 3 and (val_acc[-1] < val_acc[-3])

    logger.info("")
    logger.info("-" * 56)
    logger.info("  Overfitting Analysis - Phase %d", phase)
    logger.info("-" * 56)
    logger.info("  Best val_accuracy : %.4f  (epoch %d)", best_val_acc, best_val_epoch)
    logger.info("  Final train acc   : %.4f", final_train)
    logger.info("  Final val acc     : %.4f", final_val)
    logger.info("  Train-Val gap     : %.4f", gap)

    # Diagnosis
    if gap < 0.05:
        status = "[GOOD FIT] No significant overfitting detected."
    elif gap < 0.10:
        status = "[MILD OVERFIT] Small gap; dropout/augmentation may help."
    else:
        status = "[OVERFITTING] Large gap; increase dropout or add more augmentation."

    if declining_val:
        status += " Val accuracy declining in last 3 epochs."

    logger.info("  Diagnosis: %s", status)
    logger.info("-" * 56)

    return {
        "best_val_accuracy": best_val_acc,
        "best_epoch": best_val_epoch,
        "final_train_acc": final_train,
        "final_val_acc": final_val,
        "gap": gap,
        "status": status,
    }


# ---------------------------------------------------------------------------
# Curve plotter — loss + accuracy side by side
# ---------------------------------------------------------------------------
def plot_training_curves(
    history: keras.callbacks.History,
    save_path: str,
    phase: int = 1,
) -> None:
    """
    Save a two-panel figure:
      Left  — Training vs Validation Loss
      Right — Training vs Validation Accuracy

    HOW TO READ THESE CURVES FOR OVERFITTING:
    ─────────────────────────────────────────
    Good fit:     Both curves fall/rise together and converge closely.
    Overfitting:  Val loss starts rising while train loss keeps falling.
                  Val accuracy flatlines while train accuracy keeps rising.
                  A visible "gap" opens and widens between the two lines.
    Underfitting: Both loss curves are high and flat; accuracy is low.
                  The model hasn't learned — try more epochs or unfreeze base.
    """
    h = history.history
    epochs = range(1, len(h["loss"]) + 1)

    palette = {"train": "#4F8EF7", "val": "#F76B4F", "grid": "#2a2a3e"}
    bg      = "#1a1a2e"

    fig = plt.figure(figsize=(14, 5), facecolor=bg)
    fig.suptitle(
        f"Phase {phase} Training Curves — PlantVillage MobileNetV2",
        fontsize=14, fontweight="bold", color="white", y=1.01,
    )

    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    for ax_idx, (metric, title, best_fn) in enumerate([
        ("loss",     "Loss (lower = better)",     min),
        ("accuracy", "Accuracy (higher = better)", max),
    ]):
        ax = fig.add_subplot(gs[ax_idx])
        ax.set_facecolor(bg)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor(palette["grid"])

        train_vals = h[metric]
        val_vals   = h.get(f"val_{metric}", [])

        ax.plot(epochs, train_vals, color=palette["train"],
                linewidth=2, label="Train", marker="o", markersize=4)

        if val_vals:
            ax.plot(epochs, val_vals, color=palette["val"],
                    linewidth=2, label="Validation", marker="s", markersize=4,
                    linestyle="--")

            # Shade the gap between train and val
            ax.fill_between(
                epochs, train_vals, val_vals,
                alpha=0.12, color=palette["val"],
                label="Gap (overfit indicator)",
            )

            # Mark best validation point
            best_val = best_fn(val_vals)
            best_ep  = val_vals.index(best_val) + 1
            ax.axvline(best_ep, color="#a0ffa0", linewidth=1.2,
                       linestyle=":", alpha=0.7, label=f"Best val epoch ({best_ep})")
            ax.scatter([best_ep], [best_val], color="#a0ffa0", zorder=5, s=60)

        ax.set_title(title, color="white", fontsize=11, pad=8)
        ax.set_xlabel("Epoch", color="white", fontsize=9)
        ax.set_ylabel(metric.capitalize(), color="white", fontsize=9)
        ax.legend(fontsize=8, facecolor="#2a2a3e", labelcolor="white",
                  edgecolor="gray", loc="best")
        ax.grid(True, color=palette["grid"], linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=bg)
    plt.close()
    logger.info("Training curves saved → %s", save_path)


# ---------------------------------------------------------------------------
# Core training runner
# ---------------------------------------------------------------------------
def run_training(
    train_cfg: TrainConfig,
    gen_cfg:   GeneratorConfig,
    model_cfg: ModelConfig,
) -> Tuple[keras.Model, keras.callbacks.History]:
    """
    Full Phase-1 training loop.

    Returns the best model (weights restored by EarlyStopping) and history.
    """
    # ── Data ────────────────────────────────────────────────────────────────
    logger.info("Loading dataset...")
    train_gen, val_gen, info = create_generators(gen_cfg)

    steps_per_epoch  = info["train_samples"] // gen_cfg.batch_size
    validation_steps = info["val_samples"]   // gen_cfg.batch_size

    logger.info(
        "Dataset  |  train: %d  |  val: %d  |  steps/epoch: %d  |  val_steps: %d",
        info["train_samples"], info["val_samples"],
        steps_per_epoch, validation_steps,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model_cfg.num_classes = info["num_classes"]    # sync with discovered classes
    model = build_model(model_cfg)

    logger.info("\nModel built. Starting Phase 1 training...")
    logger.info("  Epochs   : %d (+ EarlyStopping patience %d)",
                train_cfg.phase1_epochs, train_cfg.early_stop_patience)
    logger.info("  Optimizer: Adam  lr=%.0e", model_cfg.learning_rate)
    logger.info("  Loss     : CategoricalCrossentropy (label_smoothing=0.1)")

    # ── Phase 1 ─────────────────────────────────────────────────────────────
    t0 = time.time()
    history = model.fit(
        train_gen,
        epochs=train_cfg.phase1_epochs,
        validation_data=val_gen,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=build_callbacks(train_cfg, phase=1),
        verbose=1,
    )
    elapsed = time.time() - t0
    logger.info("\nPhase 1 complete in %.1f min", elapsed / 60)

    # ── Analyse & plot ───────────────────────────────────────────────────────
    analyse_overfitting(history, phase=1)
    plot_training_curves(
        history,
        save_path=os.path.join(train_cfg.log_dir, "phase1_curves.png"),
        phase=1,
    )

    # ── Phase 2 (optional fine-tuning) ──────────────────────────────────────
    if train_cfg.run_phase2:
        logger.info("\nStarting Phase 2 — fine-tuning top base layers...")
        model = unfreeze_top_layers(model, model_cfg)

        t1 = time.time()
        history2 = model.fit(
            train_gen,
            epochs=train_cfg.phase2_epochs,
            validation_data=val_gen,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=build_callbacks(train_cfg, phase=2),
            verbose=1,
        )
        logger.info("Phase 2 complete in %.1f min", (time.time() - t1) / 60)

        analyse_overfitting(history2, phase=2)
        plot_training_curves(
            history2,
            save_path=os.path.join(train_cfg.log_dir, "phase2_curves.png"),
            phase=2,
        )

    return model, history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info(" PlantVillage — MobileNetV2 Training Pipeline")
    logger.info("=" * 60)

    # Reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    # GPU memory growth (prevents OOM on shared GPUs)
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
        logger.info("GPU detected: %s — memory growth enabled", gpu.name)

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        logger.info("No GPU detected — training on CPU (will be slow)")

    # Configs
    train_cfg = TrainConfig(
        phase1_epochs=15,
        early_stop_patience=3,
        run_phase2=False,           # flip to True for full two-phase training
    )
    gen_cfg   = GeneratorConfig()
    model_cfg = ModelConfig(
        num_classes=38,
        learning_rate=1e-3,         # Adam lr=0.001 as required
        dropout_rate=0.4,
    )

    # Train
    model, history = run_training(train_cfg, gen_cfg, model_cfg)

    # Final report
    h = history.history
    best_val = max(h.get("val_accuracy", [0]))
    logger.info("")
    logger.info("=" * 60)
    logger.info("  Training Complete")
    logger.info("  Best val_accuracy : %.4f (%.1f%%)", best_val, best_val * 100)
    logger.info("  Saved model       : %s/%s",
                train_cfg.checkpoint_dir, train_cfg.best_model_name)
    logger.info("  Curves plot       : %s/phase1_curves.png", train_cfg.log_dir)
    logger.info("%s", "=" * 60)
