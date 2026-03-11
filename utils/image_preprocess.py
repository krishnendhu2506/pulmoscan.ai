import os
from typing import Dict, Tuple

import cv2
import numpy as np


IMAGE_SIZE = (128, 128)
CLASS_FOLDER_MAP = {
    "lung_n": "Normal",
    "lung_aca": "Adenocarcinoma",
    "lung_scc": "Squamous Cell Carcinoma",
}
CLASS_INDEX = {
    "Normal": 0,
    "Adenocarcinoma": 1,
    "Squamous Cell Carcinoma": 2,
}


def preprocess_image_for_model(image_path: str, target_size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image at: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image.astype("float32") / 255.0
    return np.expand_dims(image, axis=0)


def _iter_class_dirs(dataset_root: str):
    train_dir = os.path.join(dataset_root, "train")
    test_dir = os.path.join(dataset_root, "test")

    if os.path.isdir(train_dir) or os.path.isdir(test_dir):
        for split_dir in [train_dir, test_dir]:
            if not os.path.isdir(split_dir):
                continue
            for folder_name in CLASS_FOLDER_MAP:
                candidate = os.path.join(split_dir, folder_name)
                if os.path.isdir(candidate):
                    yield candidate, CLASS_FOLDER_MAP[folder_name]
    else:
        for folder_name in CLASS_FOLDER_MAP:
            candidate = os.path.join(dataset_root, folder_name)
            if os.path.isdir(candidate):
                yield candidate, CLASS_FOLDER_MAP[folder_name]


def load_dataset_arrays(dataset_root: str, target_size: Tuple[int, int] = IMAGE_SIZE):
    images = []
    labels = []

    for class_dir, class_name in _iter_class_dirs(dataset_root):
        class_idx = CLASS_INDEX[class_name]
        for filename in os.listdir(class_dir):
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            image_path = os.path.join(class_dir, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, target_size)
            image = image.astype("float32") / 255.0
            images.append(image)
            labels.append(class_idx)

    if not images:
        raise FileNotFoundError(
            "No valid images found. Ensure dataset contains lung_n, lung_aca, lung_scc folders."
        )

    x = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    return x, y


def allowed_image_extension(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"png", "jpg", "jpeg"}
