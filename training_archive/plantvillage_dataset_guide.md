# PlantVillage Dataset Guide for TensorFlow/Keras Classification

## 1. Dataset Structure

The PlantVillage dataset is an image classification dataset containing **leaf images** of healthy and diseased plants. Your local copy is already split and structured correctly:

```
PlantVillage/
├── train/
│   ├── Apple___Apple_scab/
│   ├── Apple___Black_rot/
│   ├── Apple___Cedar_apple_rust/
│   ├── Apple___healthy/
│   ├── Blueberry___healthy/
│   ├── Cherry_(including_sour)___Powdery_mildew/
│   ├── Cherry_(including_sour)___healthy/
│   ├── Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot/
│   ├── Corn_(maize)___Common_rust_/
│   ├── Corn_(maize)___Northern_Leaf_Blight/
│   ├── Corn_(maize)___healthy/
│   ├── Grape___Black_rot/
│   ├── Grape___Esca_(Black_Measles)/
│   ├── Grape___Leaf_blight_(Isariopsis_Leaf_Spot)/
│   ├── Grape___healthy/
│   ├── Orange___Haunglongbing_(Citrus_greening)/
│   ├── Peach___Bacterial_spot/
│   ├── Peach___healthy/
│   ├── Pepper,_bell___Bacterial_spot/
│   ├── Pepper,_bell___healthy/
│   ├── Potato___Early_blight/
│   ├── Potato___Late_blight/
│   ├── Potato___healthy/
│   ├── Raspberry___healthy/
│   ├── Soybean___healthy/
│   ├── Squash___Powdery_mildew/
│   ├── Strawberry___Leaf_scorch/
│   ├── Strawberry___healthy/
│   ├── Tomato___Bacterial_spot/
│   ├── Tomato___Early_blight/
│   ├── Tomato___Late_blight/
│   ├── Tomato___Leaf_Mold/
│   ├── Tomato___Septoria_leaf_spot/
│   ├── Tomato___Spider_mites Two-spotted_spider_mite/
│   ├── Tomato___Target_Spot/
│   ├── Tomato___Tomato_Yellow_Leaf_Curl_Virus/
│   ├── Tomato___Tomato_mosaic_virus/
│   └── Tomato___healthy/
└── val/
    └── (same 38 class folders mirrored here)
```

### Class Summary

| Plant Species | # of Classes | Notes |
|---|---|---|
| Tomato | 10 | Largest share — 10 disease/healthy classes |
| Corn (Maize) | 4 | Common cereal crop |
| Grape | 4 | |
| Apple | 4 | |
| Potato | 3 | |
| Cherry | 2 | |
| Pepper, bell | 2 | |
| Peach | 2 | |
| Strawberry | 2 | |
| Orange | 1 | Only Citrus Greening |
| Blueberry | 1 | Only healthy |
| Raspberry | 1 | Only healthy |
| Soybean | 1 | Only healthy |
| Squash | 1 | Only Powdery mildew |
| **TOTAL** | **38** | |

> [!IMPORTANT]
> **You have 38 classes** — this is the standard PlantVillage split used in most research papers.
> The naming convention is `{PlantName}___{Disease}` (three underscores), which TensorFlow auto-reads as the class label directly from the folder name.

---

## 2. How TensorFlow Reads This Structure

Your existing `train.py` loads data using `tf.keras.utils.image_dataset_from_directory`. Since your `PlantVillage/` already has a `train/` and `val/` split, you should point the loader **directly at each split folder**, not the root:

```python
# CORRECT — point at each split separately
train_ds = tf.keras.utils.image_dataset_from_directory(
    "PlantVillage/train",          # ← reads 38 sub-folders as 38 classes
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical',      # one-hot encoded labels
    shuffle=True,
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "PlantVillage/val",
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical',
    shuffle=False                  # no shuffle for validation
)

class_names = train_ds.class_names  # list of 38 class name strings
print(f"Loaded {len(class_names)} classes")
```

> [!WARNING]
> Your current `train.py` uses `validation_split=` on a **single folder** (originally pointed at `./eurosat`).
> Since your PlantVillage data is **already pre-split** into `train/` and `val/`, you should **not** use `validation_split`. Load each folder individually as shown above.

---

## 3. Train / Validation Split Best Practices

### A. Use the Pre-existing Split (Recommended for PlantVillage)

The PlantVillage dataset comes with a canonical 80/20 split used by most published papers. Using the pre-split `train/` and `val/` folders ensures:
- **Reproducibility**: Same splits as papers you compare against
- **No data leakage**: The same image can't appear in both sets
- **Consistent evaluation**: Metrics are comparable across experiments

### B. If Splitting Manually (e.g., from a single flat folder)

```python
import os, shutil, random
from pathlib import Path

def split_dataset(src_dir, dest_dir, val_ratio=0.2, seed=42):
    """
    Splits a flat class-folder dataset into train/ and val/ subfolders.
    src_dir:  PlantVillage/  (contains 38 class folders)
    dest_dir: PlantVillage_split/  (will create train/ and val/)
    """
    random.seed(seed)
    src = Path(src_dir)
    dest = Path(dest_dir)

    for class_dir in src.iterdir():
        if not class_dir.is_dir():
            continue
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.JPG"))
        random.shuffle(images)

        split_idx = int(len(images) * (1 - val_ratio))
        train_images = images[:split_idx]
        val_images   = images[split_idx:]

        for split, image_list in [("train", train_images), ("val", val_images)]:
            out_dir = dest / split / class_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for img in image_list:
                shutil.copy(img, out_dir / img.name)

    print("Dataset split complete.")
```

### Key Rules for Splitting

| Rule | Why |
|---|---|
| Split **per class** (stratified), not globally | Prevents class imbalance in val set |
| Use a fixed `seed` | Reproducibility |
| Split **files**, not augmented images | Otherwise augmented copies could leak into val |
| Never include val data in normalization stats | Compute mean/std on train only |
| Use at minimum **10% for validation** | Enough to measure generalization |

---

## 4. Handling Class Imbalance

### Step 1 — Detect Imbalance

```python
import os
from pathlib import Path
import matplotlib.pyplot as plt

train_dir = Path("PlantVillage/train")
class_counts = {
    cls.name: len(list(cls.glob("*.jpg")) + list(cls.glob("*.JPG")))
    for cls in sorted(train_dir.iterdir()) if cls.is_dir()
}

# Print sorted counts
for cls, count in sorted(class_counts.items(), key=lambda x: x[1]):
    print(f"  {count:>5}  {cls}")

# Plot
plt.figure(figsize=(14, 6))
plt.bar(range(len(class_counts)), list(class_counts.values()))
plt.xticks(range(len(class_counts)), list(class_counts.keys()), rotation=90, fontsize=7)
plt.title("Samples per Class — PlantVillage Train Set")
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150)
plt.show()
```

### Step 2 — Choose a Strategy

| Strategy | When to Use | Code Complexity |
|---|---|---|
| **Class Weights** | Moderate imbalance (2–5× ratio) | Low |
| **Oversampling (repeat minority)** | Moderate-high imbalance | Medium |
| **Data Augmentation on minority** | High imbalance, small minority class | Medium |
| **Undersampling majority** | Very large datasets | Low |
| **Focal Loss** | Severe imbalance, advanced training | High |

### Strategy A — Class Weights (easiest, works well for PlantVillage)

```python
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Collect all training labels
all_labels = []
train_dir = Path("PlantVillage/train")
class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])

for idx, cls in enumerate(class_names):
    count = len(list((train_dir / cls).glob("*.jpg")))
    all_labels.extend([idx] * count)

all_labels = np.array(all_labels)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(all_labels),
    y=all_labels
)
class_weight_dict = dict(enumerate(class_weights))

# Pass to model.fit()
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    class_weight=class_weight_dict   # ← key line
)
```

> [!NOTE]
> `class_weight` in `model.fit()` works with **integer labels** (`label_mode='int'`).
> If you use `label_mode='categorical'` (one-hot), class weights are **ignored by Keras**.
> Switch to `label_mode='int'` and `loss='sparse_categorical_crossentropy'` when using class weights.

### Strategy B — Data Augmentation (within the tf.data pipeline)

```python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomBrightness(0.1),
    tf.keras.layers.RandomContrast(0.1),
], name="augmentation")

def prepare_dataset(ds, augment=False, shuffle=False):
    AUTOTUNE = tf.data.AUTOTUNE
    normalization_layer = tf.keras.layers.Rescaling(1./255)

    ds = ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)

    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)

    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=50)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

train_ds = prepare_dataset(train_ds, augment=True, shuffle=True)
val_ds   = prepare_dataset(val_ds,   augment=False, shuffle=False)
```

> [!TIP]
> Apply augmentation **only to training data**, never to validation. The `training=True` flag ensures augmentation is active only during training forward passes.

---

## 5. Updated `train.py` Entry Point for PlantVillage

Replace the `DATA_DIR` block in your `train.py`:

```python
if __name__ == "__main__":

    TRAIN_DIR = "./PlantVillage/train"
    VAL_DIR   = "./PlantVillage/val"

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, image_size=(224, 224), batch_size=32,
        label_mode='int', shuffle=True, seed=42
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR, image_size=(224, 224), batch_size=32,
        label_mode='int', shuffle=False
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)  # 38

    print(f"Classes ({num_classes}): {class_names}")
    
    model = build_mobilenetv2_model(num_classes=num_classes)
    history = train_model(model, train_ds, val_ds, epochs=20)
```

---

## 6. Summary Checklist

- [x] **Dataset structure**: `PlantVillage/train/` and `PlantVillage/val/` with **38 class folders** each
- [x] **Folder naming**: `{Plant}___{Disease}` — TF auto-reads as class labels
- [x] **Loading**: Use separate `image_dataset_from_directory` calls per split — do NOT use `validation_split` on pre-split data
- [x] **Label mode**: Use `'int'` for class weights + sparse CE; `'categorical'` for one-hot + CE
- [x] **Augmentation**: Apply only to training set, inside `tf.data.map`
- [x] **Class imbalance**: Use `compute_class_weight('balanced', ...)` → pass to `model.fit(class_weight=...)`
- [x] **Shuffle**: Only training set; use fixed seed for reproducibility
- [x] **Normalization**: Scale to `[0, 1]` via `Rescaling(1./255)` — applied before caching
