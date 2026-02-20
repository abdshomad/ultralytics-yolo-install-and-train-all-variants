"""
Global configuration settings for Ultralytics YOLO training on chicken detection dataset.
Uses YOLO format dataset structure with separate images/ and labels/ directories.
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.resolve()

# Dataset configuration
DATASET_CONFIG = {
    # Dataset directory (YOLO format with train/images, train/labels, val/images, val/labels)
    "dataset_dir": "chicken-detection-labelme-format/yolo-format",
    # Paths (relative to project root)
    "train_images": "chicken-detection-labelme-format/yolo-format/train/images",
    "train_labels": "chicken-detection-labelme-format/yolo-format/train/labels",
    "val_images": "chicken-detection-labelme-format/yolo-format/val/images",
    "val_labels": "chicken-detection-labelme-format/yolo-format/val/labels",
    # "test_images": "chicken-detection-labelme-format/yolo-format/test/images",
    # "test_labels": "chicken-detection-labelme-format/yolo-format/test/labels",
    # Dataset settings
    "num_classes": 2,  # chicken, not-chicken
    "class_names": ["chicken", "not-chicken"],
}

# YOLO variant to model name mapping
YOLO_VARIANTS = {
    "n": "yolo26n",
    "s": "yolo26s",
    "m": "yolo26m",
    "l": "yolo26l",
    "x": "yolo26x",
}

# Default training settings
DEFAULT_TRAINING_CONFIG = {
    "epochs": 100,
    "batch_size": 16,
    "imgsz": 640,
    "lr": 0.01,
    "seed": 0,
}

# Output directories for models (relative to project root)
OUTPUT_DIRS = {
    "n": "yolo-models/yolo-n",
    "s": "yolo-models/yolo-s",
    "m": "yolo-models/yolo-m",
    "l": "yolo-models/yolo-l",
    "x": "yolo-models/yolo-x",
}

# Log directories for training logs (relative to project root)
LOG_DIRS = {
    "n": "yolo-models/yolo-n/logs",
    "s": "yolo-models/yolo-s/logs",
    "m": "yolo-models/yolo-m/logs",
    "l": "yolo-models/yolo-l/logs",
    "x": "yolo-models/yolo-x/logs",
}
