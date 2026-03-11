import json
import os
import sys
from typing import Dict

import cv2
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.image_preprocess import preprocess_image_for_model

MODEL_PATH = os.path.join(BASE_DIR, "model", "lung_cancer_model.h5")
CLASS_MAP_PATH = os.path.join(BASE_DIR, "model", "class_names.json")
DEFAULT_CLASS_NAMES = [
    "Normal",
    "Adenocarcinoma",
    "Squamous Cell Carcinoma",
    "Large Cell Carcinoma",
]

_model = None


def load_class_names():
    if os.path.exists(CLASS_MAP_PATH):
        try:
            with open(CLASS_MAP_PATH, "r", encoding="utf-8") as fp:
                payload = json.load(fp)
            classes = payload.get("display_classes", [])
            if classes:
                return classes
        except Exception:
            pass
    return DEFAULT_CLASS_NAMES


CLASS_NAMES = load_class_names()


def load_trained_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "Model file not found at model/lung_cancer_model.h5. Run python model/train_model.py first."
            )
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def _find_last_conv_layer(model: tf.keras.Model) -> str:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found for Grad-CAM generation.")


def generate_gradcam(image_path: str, model: tf.keras.Model, output_path: str) -> str:
    input_tensor = preprocess_image_for_model(image_path)
    last_conv_name = _find_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_tensor)
        class_idx = tf.argmax(predictions[0])
        class_channel = predictions[:, class_idx]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    original = cv2.imread(image_path)
    if original is None:
        raise ValueError("Cannot read input image for Grad-CAM.")

    heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)

    cv2.imwrite(output_path, overlay)
    return output_path


def predict_scan(image_path: str, generate_cam: bool = True, cam_dir: str = None) -> Dict:
    model = load_trained_model()
    class_names = load_class_names()
    input_tensor = preprocess_image_for_model(image_path)

    probabilities = model.predict(input_tensor, verbose=0)[0]

    n = min(len(class_names), len(probabilities))
    if n == 0:
        raise ValueError("Prediction output is empty.")

    probabilities = probabilities[:n]
    class_names = class_names[:n]

    predicted_idx = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_idx]

    probability_map = {class_names[i]: float(probabilities[i]) for i in range(n)}
    confidence = float(probabilities[predicted_idx])

    gradcam_path = None
    if generate_cam and cam_dir:
        os.makedirs(cam_dir, exist_ok=True)
        filename = os.path.splitext(os.path.basename(image_path))[0]
        gradcam_path = os.path.join(cam_dir, f"{filename}_gradcam.jpg")
        try:
            gradcam_path = generate_gradcam(image_path, model, gradcam_path)
        except Exception:
            gradcam_path = None

    return {
        "predicted_class": predicted_class,
        "probabilities": probability_map,
        "confidence": confidence,
        "gradcam_path": gradcam_path,
    }
