import argparse
import os
import numpy as np
import tensorflow as tf

def preprocess_image_for_prediction(img_path: str, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Preprocesses an image for prediction using a trained Keras model.
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found at path: {img_path}")
        
    # Load image, convert to RGB, resize
    img = tf.keras.utils.load_img(
        img_path, 
        color_mode="rgb", 
        target_size=target_size
    )
    
    # Convert to NumPy array (float32)
    img_array = tf.keras.utils.img_to_array(img)
    
    # Normalize pixel values to [0, 1]
    img_array = img_array / 255.0
    
    # Expand dimensions to shape (1, 224, 224, 3)
    input_tensor = np.expand_dims(img_array, axis=0)
    
    return input_tensor

def get_class_names_from_dataset(dataset_dir: str) -> list:
    """
    Dynamically infers class labels from the dataset folder structure.
    Overrides the need for hardcoded arrays by mapping folders alphabetically,
    which ensures perfect consistency with tf.keras.utils.image_dataset_from_directory.
    """
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        
    classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    
    # Return sorted alphabetically just like the Keras loader does
    return sorted(classes)

def predict_single_image(img_path: str, model_path: str = 'model.h5', class_names: list = None) -> tuple:
    """
    Loads model, runs inference on the input image, and extracts the highest prediction.
    
    Returns:
        tuple: (predicted_class_label, confidence_score)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at path: {model_path}")
        
    print(f"Loading Keras model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    print(f"Pre-processing image: {img_path}")
    img_tensor = preprocess_image_for_prediction(img_path)
    
    print("Running prediction...")
    prediction_probs = model.predict(img_tensor, verbose=0)[0]
    
    # Extract predicted class index
    predicted_idx = int(np.argmax(prediction_probs))
    
    # Extract associated confidence score
    confidence_score = float(prediction_probs[predicted_idx])
    
    # Map index to class label
    if class_names and predicted_idx < len(class_names):
        predicted_class = class_names[predicted_idx]
    else:
        # Fallback to index if string labels are absent
        predicted_class = f"Class_{predicted_idx}"
        
    return predicted_class, confidence_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict an image's class using the trained model.")
    parser.add_argument("image_path", help="Path to the input image for prediction")
    parser.add_argument("--model", type=str, default="model.h5", help="Path to the saved Keras model file")
    parser.add_argument("--dataset", type=str, default="./eurosat", help="Path to dataset directory for dynamic class inference")
    
    args = parser.parse_args()
    
    # Avoid hardcoding by extracting the sorted folder names exactly as the Keras loader does
    try:
        dynamic_class_names = get_class_names_from_dataset(args.dataset)
    except FileNotFoundError:
        print(f"Warning: Dataset directory '{args.dataset}' not found. Falling back to standard string index class identifiers.")
        dynamic_class_names = None
    
    try:
        label, score = predict_single_image(args.image_path, args.model, class_names=dynamic_class_names)
        print("\n" + "==============================")
        print("      PREDICTION RESULT       ")
        print("==============================")
        print(f" Predicted Class : {label}")
        print(f" Confidence Score: {score:.5f} ({score*100:.2f}%)")
        print("==============================\n")
    except Exception as e:
        print(f"\nAn error occurred during prediction: {e}")
