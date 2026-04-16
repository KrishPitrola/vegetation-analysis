import os
import tensorflow as tf

def get_data_generators(data_dir: str, 
                        batch_size: int = 32, 
                        img_height: int = 224, 
                        img_width: int = 224, 
                        validation_split: float = 0.2, 
                        seed: int = 123):
    """
    Creates optimized tf.data.Dataset pipelines for training and validation.
    
    Args:
        data_dir: Path to the root directory containing class folders.
        batch_size: Number of images per batch.
        img_height: Target height to resize images to.
        img_width: Target width to resize images to.
        validation_split: Float between 0 and 1 indicating fraction of data for validation.
        seed: Random seed for reproducibility.
        
    Returns:
        train_ds: A tf.data.Dataset object for training.
        val_ds: A tf.data.Dataset object for validation.
        class_names: List of strings containing the inferred class names.
    """
    
    print(f"Loading training dataset from {data_dir}...")
    # Load training dataset, holding out a fraction for validation
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='categorical' # Use 'int' if using sparse_categorical_crossentropy
    )

    print(f"Loading validation dataset from {data_dir}...")
    # Load validation dataset using the same split and seed
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='categorical'
    )

    # Classes are automatically inferred from directory names
    class_names = train_ds.class_names
    
    # Configure the dataset for performance
    AUTOTUNE = tf.data.AUTOTUNE
    
    # Apply normalization (scaling pixels to [0, 1])
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    
    def prepare_dataset(ds, shuffle=False):
        # Normalize the pixel values mapping the transformation
        ds = ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)
        
        # Dataset caching for faster epoch loading (kept in memory or disk)
        ds = ds.cache()
        
        if shuffle:
            # Reduce shuffle buffer size. Since tf.keras.utils.image_dataset_from_directory 
            # returns a batched dataset, buffer_size=1000 would hold 1000 * batch_size images in RAM!
            # Reducing it to 50 provides a good mix of batch-level shuffling without OOM errors.
            ds = ds.shuffle(buffer_size=50)
            
        # Prefetching to overlap data loading & model execution using AUTOTUNE
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    # Apply optimizations
    train_ds = prepare_dataset(train_ds, shuffle=True)
    val_ds = prepare_dataset(val_ds)

    return train_ds, val_ds, class_names

def build_mobilenetv2_model(num_classes: int, input_shape: tuple = (224, 224, 3)):
    """
    Builds a transfer learning model using MobileNetV2 with a custom classification head.
    """
    # Load pretrained MobileNetV2 without the top layer
    base_model = tf.keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze the base layers
    base_model.trainable = False
    
    # Add custom head
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, train_ds, val_ds, epochs: int = 20):
    """
    Trains the built model with early stopping and model checkpointing.
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    print(f"\nStarting training for up to {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        steps_per_epoch=200,
        validation_steps=50
    )
    
    return history

if __name__ == "__main__":

    DATA_DIR = "./eurosat" # Change this to your actual dataset directory
    
    if os.path.exists(DATA_DIR):
        train_dataset, validation_dataset, classes = get_data_generators(DATA_DIR)
        
        print(f"\\nDiscovered {len(classes)} classes: {classes}")
        
        # Verify batch shape and normalization
        for image_batch, labels_batch in train_dataset.take(1):
            print(f"Image batch shape: {image_batch.shape}")
            print(f"Label batch shape: {labels_batch.shape}")
            print(f"Min pixel value: {tf.math.reduce_min(image_batch):.4f}")
            print(f"Max pixel value: {tf.math.reduce_max(image_batch):.4f}")
            
        # Build the transfer learning model
        print("\nBuilding MobileNetV2 transfer learning model...")
        model = build_mobilenetv2_model(num_classes=len(classes))
        model.summary()
        
        # Train the model
        history = train_model(model, train_dataset, validation_dataset, epochs=20)
        
        # Display final results
        final_train_acc = history.history.get('accuracy', [-1])[-1]
        final_val_acc = history.history.get('val_accuracy', [-1])[-1]
        print(f"\nTraining Complete!")
        print(f"Final Training Accuracy: {final_train_acc:.4f}")
        print(f"Final Validation Accuracy: {final_val_acc:.4f}")
        
    else:
        print(f"Dataset directory '{DATA_DIR}' not found. Please create it or update DATA_DIR.")
