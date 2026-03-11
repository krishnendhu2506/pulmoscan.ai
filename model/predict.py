from utils.image_preprocess import preprocess_image_for_model
import json
import os
import sys
from typing import Dict

import numpy as np
import tensorflow as tf

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


H5_MODEL_PATH = os.path.join(BASE_DIR, "model", "lung_cancer_model.h5")
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


def load_tf_model():
    global _model

    if _model is None:
        if not os.path.exists(H5_MODEL_PATH):
            raise FileNotFoundError(
                "Model not found at model/lung_cancer_model.h5. Train or place the model file first."
            )
        _model = tf.keras.models.load_model(H5_MODEL_PATH, compile=False)

    return _model


def predict_scan(image_path: str, generate_cam: bool = True, cam_dir: str = None) -> Dict:
    model = load_tf_model()
    class_names = load_class_names()

    input_tensor = preprocess_image_for_model(image_path).astype(np.float32)
    raw_outputs = model.predict(input_tensor, verbose=0)

    probabilities = np.array(raw_outputs).squeeze()
    if probabilities.ndim == 0:
        probabilities = np.array([float(probabilities)])

    n = min(len(class_names), len(probabilities))
    if n == 0:
        raise ValueError("Prediction output is empty.")

    probabilities = probabilities[:n]
    class_names = class_names[:n]

    predicted_idx = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_idx]

    probability_map = {class_names[i]: float(probabilities[i]) for i in range(n)}
    confidence = float(probabilities[predicted_idx])

    # Grad-CAM is optional; not generated in this path.
    gradcam_path = None

    return {
        "predicted_class": predicted_class,
        "probabilities": probability_map,
        "confidence": confidence,
        "gradcam_path": gradcam_path,
    }
