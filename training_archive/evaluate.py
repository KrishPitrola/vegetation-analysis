"""
evaluate.py
-----------
Post-training evaluation for the PlantVillage MobileNetV2 classifier.

What this script does
─────────────────────
  1. Loads training history from CSV  → plots loss & accuracy curves
  2. Loads best_model.keras           → runs model.evaluate() on val set
  3. Runs full inference on val set   → builds confusion matrix & class report

Outputs saved to ./logs/
  eval_curves.png          - loss + accuracy training curves
  confusion_matrix.png     - 38×38 confusion matrix (normalised)
  classification_report.txt- per-class precision, recall, F1

Run:
    python evaluate.py
"""

import logging
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from data_generator import GeneratorConfig, create_generators

warnings.filterwarnings("ignore", category=UserWarning)

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
# Paths — change here if your layout differs
# ---------------------------------------------------------------------------
MODEL_PATH   = "./checkpoints/best_model.keras"
HISTORY_CSV  = "./logs/phase1_history.csv"
LOG_DIR      = "./logs"


# ===========================================================================
# 1.  TRAINING CURVE PLOTS
# ===========================================================================
def plot_history_from_csv(csv_path: str, save_dir: str) -> pd.DataFrame:
    """
    Load CSVLogger output and plot loss + accuracy curves.

    HOW TO INTERPRET
    ─────────────────
    Loss panel (left):
      • Both curves falling together   → healthy learning
      • val_loss flattening / rising   → model is overfitting; gap opening
      • Both curves stuck high         → underfitting; try more epochs or unfreeze base

    Accuracy panel (right):
      • val_accuracy tracks train_acc  → good generalisation
      • train_acc >> val_acc           → overfitting; increase Dropout or augmentation
      • Both stuck low                 → underfitting or wrong LR

    The shaded region between the two lines visually highlights the
    generalisation gap — keep it as thin as possible.
    """
    if not Path(csv_path).exists():
        logger.warning("History CSV not found at '%s'. Skipping curve plot.", csv_path)
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    # Epoch column is 0-indexed in CSVLogger; shift to 1-indexed for display
    df["epoch"] += 1

    logger.info("Loaded training history: %d epochs from '%s'", len(df), csv_path)

    bg      = "#1a1a2e"
    palette = {"train": "#4F8EF7", "val": "#F76B4F", "grid": "#2a2a3e"}

    fig = plt.figure(figsize=(14, 5), facecolor=bg)
    fig.suptitle(
        "Training History — PlantVillage MobileNetV2",
        fontsize=14, fontweight="bold", color="white",
    )
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    panels = [
        ("loss",     "val_loss",     "Loss (lower = better)",      min),
        ("accuracy", "val_accuracy", "Accuracy (higher = better)", max),
    ]

    for idx, (train_col, val_col, title, best_fn) in enumerate(panels):
        ax = fig.add_subplot(gs[idx])
        ax.set_facecolor(bg)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor(palette["grid"])

        train_vals = df[train_col].tolist()
        val_vals   = df[val_col].tolist() if val_col in df.columns else []
        epochs     = df["epoch"].tolist()

        ax.plot(epochs, train_vals, color=palette["train"], linewidth=2,
                marker="o", markersize=5, label="Train")

        if val_vals:
            ax.plot(epochs, val_vals, color=palette["val"], linewidth=2,
                    marker="s", markersize=5, linestyle="--", label="Validation")
            ax.fill_between(epochs, train_vals, val_vals,
                            alpha=0.12, color=palette["val"],
                            label="Generalisation gap")

            # Annotate the best validation point
            best_val   = best_fn(val_vals)
            best_epoch = epochs[val_vals.index(best_val)]
            ax.axvline(best_epoch, color="#a0ffa0", linewidth=1.2,
                       linestyle=":", alpha=0.8, label=f"Best val (ep {best_epoch})")
            ax.scatter([best_epoch], [best_val], color="#a0ffa0", zorder=5, s=70)
            ax.annotate(
                f"{best_val:.4f}",
                xy=(best_epoch, best_val),
                xytext=(10, 8), textcoords="offset points",
                color="#a0ffa0", fontsize=8,
            )

        ax.set_title(title, color="white", fontsize=11, pad=8)
        ax.set_xlabel("Epoch", color="white", fontsize=9)
        ax.set_ylabel(train_col.capitalize(), color="white", fontsize=9)
        ax.set_xticks(epochs)
        ax.legend(fontsize=8, facecolor="#2a2a3e", labelcolor="white",
                  edgecolor="gray", loc="best")
        ax.grid(True, color=palette["grid"], linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    out = os.path.join(save_dir, "eval_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=bg)
    plt.close()
    logger.info("Training curves saved -> %s", out)
    return df


# ===========================================================================
# 2.  MODEL.EVALUATE ON VALIDATION SET
# ===========================================================================
def evaluate_model(model: tf.keras.Model, val_gen, steps: int) -> dict:
    """
    Run model.evaluate() on the full validation set.

    Returns a dict of metric_name → value.

    HOW TO INTERPRET
    ─────────────────
    val_loss     : cross-entropy; lower is better.  Compare with train_loss;
                   a large gap (val >> train) signals overfitting.
    val_accuracy : fraction of images classified correctly.
                   For 38 classes, random chance = 1/38 ≈ 2.6%.
                   Anything above 85% is strong for a frozen MobileNetV2 head.
    top5_accuracy: fraction where the true class appears in the top-5
                   predictions. Should be >99% for a well-trained model.
    """
    logger.info("\nRunning model.evaluate() on validation set (%d steps)...", steps)
    results = model.evaluate(val_gen, steps=steps, verbose=1, return_dict=True)

    logger.info("")
    logger.info("=" * 50)
    logger.info("  Validation Evaluation Results")
    logger.info("=" * 50)
    for name, value in results.items():
        logger.info("  %-25s : %.4f", name, value)
    logger.info("=" * 50)
    return results


# ===========================================================================
# 3.  CONFUSION MATRIX + CLASSIFICATION REPORT
# ===========================================================================
def run_predictions(model: tf.keras.Model, val_gen, total_samples: int) -> tuple:
    """
    Run inference across the full validation set.

    Returns (y_true, y_pred) as integer label arrays.

    Note: val_gen must have shuffle=False so label order is deterministic.
    """
    logger.info("\nGenerating predictions over %d validation images...", total_samples)

    y_true, y_pred = [], []
    steps = int(np.ceil(total_samples / val_gen.batch_size))

    for step in range(steps):
        images, labels = next(val_gen)
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels, axis=1))
        y_pred.extend(np.argmax(preds,  axis=1))
        if (step + 1) % 20 == 0:
            logger.info("  Processed %d / %d batches", step + 1, steps)

    # Trim to exact sample count (last batch may be partial)
    y_true = np.array(y_true)[:total_samples]
    y_pred = np.array(y_pred)[:total_samples]

    logger.info("Predictions complete. y_true: %s, y_pred: %s",
                y_true.shape, y_pred.shape)
    return y_true, y_pred


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    save_dir: str,
) -> None:
    """
    Plot and save a normalised confusion matrix.

    HOW TO INTERPRET
    ─────────────────
    Rows = true class,  Columns = predicted class.

    Perfect model: bright diagonal, dark everywhere else.

    - Diagonal cell (i, i)  → fraction of class i correctly classified.
      e.g., 0.98 = 98% of Tomato_Bacterial_spot images labelled correctly.

    - Off-diagonal cell (i, j) → class i was predicted as class j that % of
      the time.  High off-diagonal = the model confuses two visually similar
      diseases (e.g. Tomato Early Blight vs Septoria Leaf Spot).

    Action:
      • If a column is bright for many rows → the model over-predicts that class
        (class imbalance issue or similar textures).
      • If a row is dark on the diagonal     → that class is hard to classify;
        add more training samples or targeted augmentation.
    """
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    n = len(class_names)
    # Scale figure with number of classes
    fig_size = max(20, n * 0.55)
    font_size = max(4, 8 - n // 15)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)

    # Colour bar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.ax.tick_params(colors="white", labelsize=8)
    cbar.set_label("Proportion", color="white", fontsize=9)

    # Tick labels — use short names (truncate at 20 chars)
    short_names = [n[:22] for n in class_names]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_names, rotation=90, fontsize=font_size, color="white")
    ax.set_yticklabels(short_names, fontsize=font_size, color="white")

    # Annotate cells (only if small enough matrix)
    if n <= 20:
        thresh = cm.max() / 2.0
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{cm[i, j]:.2f}",
                        ha="center", va="center", fontsize=5,
                        color="white" if cm[i, j] < thresh else "black")

    ax.set_title(
        "Normalised Confusion Matrix — PlantVillage (38 classes)",
        color="white", fontsize=13, fontweight="bold", pad=14,
    )
    ax.set_xlabel("Predicted Label", color="white", fontsize=10, labelpad=10)
    ax.set_ylabel("True Label",      color="white", fontsize=10, labelpad=10)
    ax.tick_params(axis="both", which="both", length=0)

    plt.tight_layout()
    out = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    logger.info("Confusion matrix saved -> %s", out)


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    save_dir: str,
) -> pd.DataFrame:
    """
    Print and save a per-class precision / recall / F1 report.

    HOW TO INTERPRET
    ─────────────────
    Precision  : Of all images predicted as class X, how many were actually X?
                 Low precision → model is generating false positives for that class.

    Recall     : Of all true class X images, how many did the model find?
                 Low recall → model is missing instances of that class (false negatives).

    F1-score   : Harmonic mean of precision and recall.
                 Use this as your single-number per-class performance metric.
                 F1=1.0 is perfect; F1<0.70 indicates a struggling class.

    Support    : Number of validation images for that class.
                 Low-support classes often have worse F1 — normal for imbalanced data.

    macro avg  : Mean across all classes (treats each class equally).
    weighted avg: Support-weighted mean (biased toward large classes).
    """
    report_str = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    report_path = os.path.join(save_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("PlantVillage MobileNetV2 — Per-Class Classification Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(report_str)

    logger.info("Classification report saved -> %s", report_path)

    # Also build a DataFrame for sorting & inspection
    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    df = pd.DataFrame(report_dict).T
    df = df.drop(["accuracy"], errors="ignore")

    # Bottom-5 worst classes by F1
    class_only = df[df.index.isin(class_names)].copy()
    class_only = class_only.astype(float)
    worst5 = class_only.nsmallest(5, "f1-score")

    logger.info("")
    logger.info("Top-5 HARDEST classes (lowest F1):")
    for cls, row in worst5.iterrows():
        logger.info(
            "  %-45s  F1=%.4f  prec=%.4f  rec=%.4f  n=%d",
            cls, row["f1-score"], row["precision"], row["recall"], int(row["support"]),
        )

    return df


# ===========================================================================
# 4.  MAIN
# ===========================================================================
def main() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)

    logger.info("=" * 60)
    logger.info(" PlantVillage — Post-Training Evaluation")
    logger.info("=" * 60)

    # ── Step 1: Plot training curves from CSV ───────────────────────────────
    logger.info("\n[1/4] Plotting training history from CSV...")
    df_history = plot_history_from_csv(HISTORY_CSV, LOG_DIR)

    if not df_history.empty:
        final = df_history.iloc[-1]
        logger.info(
            "Final epoch metrics — train_acc: %.4f | val_acc: %.4f | "
            "train_loss: %.4f | val_loss: %.4f",
            final["accuracy"], final["val_accuracy"],
            final["loss"],     final["val_loss"],
        )

    # ── Step 2: Load model ──────────────────────────────────────────────────
    logger.info("\n[2/4] Loading model from '%s'...", MODEL_PATH)
    if not Path(MODEL_PATH).exists():
        logger.error("Model not found at '%s'. Train first using train_model.py", MODEL_PATH)
        raise SystemExit(1)

    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("Model loaded successfully.")
    logger.info("Input:  %s", model.input_shape)
    logger.info("Output: %s", model.output_shape)

    # ── Step 3: Build validation generator ─────────────────────────────────
    logger.info("\n[3/4] Building validation generator...")
    gen_cfg = GeneratorConfig()
    _, val_gen, info = create_generators(gen_cfg)

    class_names  = info["class_names"]
    val_samples  = info["val_samples"]
    val_steps    = val_samples // gen_cfg.batch_size

    logger.info("Val samples: %d | Classes: %d | Steps: %d",
                val_samples, len(class_names), val_steps)

    # ── Step 4a: model.evaluate() ───────────────────────────────────────────
    logger.info("\n[4/4] Evaluation pipeline...")
    # Reset generator to ensure consistent ordering
    val_gen.reset()
    eval_results = evaluate_model(model, val_gen, steps=val_steps)

    # ── Step 4b: Predictions → confusion matrix ─────────────────────────────
    val_gen.reset()
    y_true, y_pred = run_predictions(model, val_gen, total_samples=val_samples)

    plot_confusion_matrix(y_true, y_pred, class_names, LOG_DIR)
    df_report = save_classification_report(y_true, y_pred, class_names, LOG_DIR)

    # ── Final summary ───────────────────────────────────────────────────────
    overall_acc = (y_true == y_pred).mean()
    logger.info("")
    logger.info("=" * 60)
    logger.info("  Evaluation Complete")
    logger.info("  model.evaluate val_accuracy  : %.4f (%.2f%%)",
                eval_results.get("accuracy", 0),
                eval_results.get("accuracy", 0) * 100)
    logger.info("  Prediction-based accuracy    : %.4f (%.2f%%)",
                overall_acc, overall_acc * 100)
    logger.info("")
    logger.info("  Saved outputs:")
    logger.info("    logs/eval_curves.png")
    logger.info("    logs/confusion_matrix.png")
    logger.info("    logs/classification_report.txt")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
