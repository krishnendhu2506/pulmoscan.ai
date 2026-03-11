import argparse
import json
import os
import random
import sys
from typing import List, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "model", "lung_cancer_model.h5")
CLASS_MAP_PATH = os.path.join(BASE_DIR, "model", "class_names.json")

SEED = 42
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 12
IMG_SIZE = (128, 128)
ALLOWED_EXTS = (".png", ".jpg", ".jpeg")

DISPLAY_LABELS = {
    "normal": "Normal",
    "adenocarcinoma": "Adenocarcinoma",
    "squamous_cell_carcinoma": "Squamous Cell Carcinoma",
    "large_cell_carcinoma": "Large Cell Carcinoma",
}


def canonicalize_folder_name(name: str):
    n = name.lower().replace("_", ".")
    if "adenocarcinoma" in n:
        return "adenocarcinoma"
    if "squamous" in n and "carcinoma" in n:
        return "squamous_cell_carcinoma"
    if "large.cell.carcinoma" in n or ("large" in n and "carcinoma" in n):
        return "large_cell_carcinoma"
    if "normal" in n:
        return "normal"
    return None


def collect_from_split(split_dir: str) -> Tuple[List[str], List[str]]:
    image_paths = []
    labels = []

    if not os.path.isdir(split_dir):
        return image_paths, labels

    for class_dir_name in os.listdir(split_dir):
        class_dir = os.path.join(split_dir, class_dir_name)
        if not os.path.isdir(class_dir):
            continue

        canonical = canonicalize_folder_name(class_dir_name)
        if canonical is None:
            continue

        for fname in os.listdir(class_dir):
            if not fname.lower().endswith(ALLOWED_EXTS):
                continue
            image_paths.append(os.path.join(class_dir, fname))
            labels.append(canonical)

    return image_paths, labels


def build_file_lists():
    train_dir = os.path.join(DATASET_DIR, "train")
    test_dir = os.path.join(DATASET_DIR, "test")
    valid_dir = os.path.join(DATASET_DIR, "valid")

    x_train, y_train = collect_from_split(train_dir)
    x_test, y_test = collect_from_split(test_dir)
    x_valid, y_valid = collect_from_split(valid_dir)

    x_val = x_test + x_valid
    y_val = y_test + y_valid

    if not x_train:
        # Fallback: use all split folders together then stratified split.
        all_x = x_train + x_test + x_valid
        all_y = y_train + y_test + y_valid
        if not all_x:
            raise FileNotFoundError("No images found in dataset/train, dataset/test, dataset/valid.")
        x_train, x_val, y_train, y_val = train_test_split(
            all_x,
            all_y,
            test_size=0.2,
            random_state=SEED,
            stratify=all_y,
        )

    if not x_val:
        x_train, x_val, y_train, y_val = train_test_split(
            x_train,
            y_train,
            test_size=0.2,
            random_state=SEED,
            stratify=y_train,
        )

    class_order = [
        c for c in ["normal", "adenocarcinoma", "squamous_cell_carcinoma", "large_cell_carcinoma"]
        if c in set(y_train + y_val)
    ]
    if len(class_order) < 2:
        raise ValueError("Need at least 2 classes to train.")

    class_to_index = {name: idx for idx, name in enumerate(class_order)}
    y_train_idx = [class_to_index[y] for y in y_train]
    y_val_idx = [class_to_index[y] for y in y_val]

    print(f"Train samples: {len(x_train)} | Validation samples: {len(x_val)}")
    print("Class order:", [DISPLAY_LABELS[c] for c in class_order])

    return x_train, y_train_idx, x_val, y_val_idx, class_order


def decode_and_resize(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


def make_dataset(paths, labels, batch_size, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=min(len(paths), 5000), seed=SEED)

    ds = ds.map(decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_cnn_model(input_shape=(128, 128, 3), num_classes=3):
    augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(0.1, 0.1),
        ],
        name="augmentation",
    )

    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            augmentation,
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train lung cancer CNN model.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--fast", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    epochs = args.epochs
    batch_size = args.batch_size
    max_train_batches = args.max_train_batches
    max_val_batches = args.max_val_batches
    patience = 3

    if args.fast:
        epochs = min(epochs, 5)
        batch_size = max(batch_size, 64)
        max_train_batches = max_train_batches or 200
        max_val_batches = max_val_batches or 50
        patience = 2

    x_train, y_train, x_val, y_val, class_order = build_file_lists()
    train_ds = make_dataset(x_train, y_train, batch_size=batch_size, training=True)
    val_ds = make_dataset(x_val, y_val, batch_size=batch_size, training=False)

    if max_train_batches > 0:
        train_ds = train_ds.take(max_train_batches)
    if max_val_batches > 0:
        val_ds = val_ds.take(max_val_batches)

    model = build_cnn_model(input_shape=(128, 128, 3), num_classes=len(class_order))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=1)

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Validation loss: {val_loss:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

    payload = {
        "canonical_classes": class_order,
        "display_classes": [DISPLAY_LABELS[c] for c in class_order],
    }
    with open(CLASS_MAP_PATH, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    print(f"Class order saved to: {CLASS_MAP_PATH}")


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    main()
