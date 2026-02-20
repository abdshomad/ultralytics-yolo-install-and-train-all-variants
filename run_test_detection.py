#!/usr/bin/env python3
"""
Run YOLO detection on chicken-detection-labelme-format/yolo-format/test/images
using a fine-tuned checkpoint and draw results with supervision. Saves to test-result.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import supervision as sv
from ultralytics import YOLO

import configs

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
TEST_IMAGE_DIR = PROJECT_ROOT / "chicken-detection-labelme-format/yolo-format/test/images"
OUTPUT_DIR = PROJECT_ROOT / "chicken-detection-labelme-format/yolo-format/test-result"

CLASS_NAMES = configs.DATASET_CONFIG["class_names"]  # ["chicken", "not-chicken"]
CONFIDENCE_THRESHOLD = 0.25
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Variant to YOLO model name mapping
VARIANT_MODELS = {
    "n": "yolo26n",
    "s": "yolo26s",
    "m": "yolo26m",
    "l": "yolo26l",
    "x": "yolo26x",
}


def get_latest_checkpoint(checkpoint_dir: Path) -> Path:
    """Prefer best.pt, else last.pt, else newest .pt."""
    for name in ["best.pt", "last.pt"]:
        p = checkpoint_dir / name
        if p.exists():
            return p
    pts = list(checkpoint_dir.glob("*.pt"))
    if not pts:
        raise FileNotFoundError(f"No .pt checkpoint found in {checkpoint_dir}")
    return max(pts, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLO on test images, draw with supervision"
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=list(VARIANT_MODELS),
        default="s",
        help="Model variant (n, s, m, l, x)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path (default: latest in model output dir)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help="Confidence threshold",
    )
    args = parser.parse_args()

    output_dir = configs.OUTPUT_DIRS[args.variant]
    checkpoint_dir = PROJECT_ROOT / output_dir
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).resolve()
    else:
        if not checkpoint_dir.exists():
            print(f"Checkpoint dir not found: {checkpoint_dir}")
            print("Train a model first with: uv run train-yolo-s.py")
            sys.exit(1)
        checkpoint_path = get_latest_checkpoint(checkpoint_dir)

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    print(f"Using checkpoint: {checkpoint_path}")

    if not TEST_IMAGE_DIR.exists():
        print(f"Test image dir not found: {TEST_IMAGE_DIR}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(checkpoint_path))
    print(f"Loaded YOLO26{args.variant.upper()}")

    image_paths = [
        p for p in TEST_IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    image_paths.sort()
    print(f"Found {len(image_paths)} images in {TEST_IMAGE_DIR}")

    color_palette = sv.ColorPalette([
        sv.Color.from_hex("#00FF00"),  # chicken
        sv.Color.from_hex("#0000FF"),  # not-chicken
    ])
    box_annotator = sv.BoxAnnotator(thickness=2, color=color_palette)

    for i, image_path in enumerate(image_paths):
        im_pil = Image.open(image_path).convert("RGB")
        frame = np.array(im_pil)

        # YOLO prediction
        results = model.predict(str(image_path), conf=args.conf, verbose=False)
        
        # Convert YOLO results to supervision format
        detections = sv.Detections.from_ultralytics(results[0])
        
        if len(detections) > 0:
            frame = box_annotator.annotate(scene=frame, detections=detections)

        out_path = OUTPUT_DIR / image_path.name
        Image.fromarray(frame).save(out_path)
        if (i + 1) % 50 == 0 or (i + 1) == len(image_paths):
            print(f"  Saved {i + 1}/{len(image_paths)} -> {out_path.name}")

    print(f"Done. Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
