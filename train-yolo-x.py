#!/usr/bin/env python3
"""
Train YOLO26x model on chicken detection dataset.

Usage:
    uv run train-yolo-x.py [--epochs N] [--batch-size N] [--imgsz N] [--resume PATH] [--output-dir DIR]
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import torch

import configs
from ultralytics import YOLO


def get_checkpoint_epoch(ckpt_path: Path) -> int:
    """Return completed epoch (0-indexed) from checkpoint, or -1 if unknown."""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        epoch = int(ckpt.get("epoch", -1))
        if epoch >= 0:
            return epoch
        # Fallback: if epoch=-1, assume completed train_args.epochs (e.g. finished run)
        train_args = ckpt.get("train_args") or {}
        if isinstance(train_args, dict):
            return int(train_args.get("epochs", 100)) - 1  # 0-indexed
        return -1
    except Exception:
        return -1


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO26x on chicken detection dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=configs.DEFAULT_TRAINING_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=configs.DEFAULT_TRAINING_CONFIG["batch_size"])
    parser.add_argument("--imgsz", type=int, default=configs.DEFAULT_TRAINING_CONFIG["imgsz"])
    parser.add_argument("--lr", type=float, default=configs.DEFAULT_TRAINING_CONFIG["lr"])
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--fresh", action="store_true", help="Force fresh training (ignore existing weights)")
    parser.add_argument("--new", action="store_true", help="Force fresh training (ignore existing weights, alias for --fresh)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=configs.DEFAULT_TRAINING_CONFIG["seed"])
    args = parser.parse_args()

    variant = "x"
    dataset_dir = configs.PROJECT_ROOT / configs.DATASET_CONFIG["dataset_dir"]
    log_dir = configs.PROJECT_ROOT / configs.LOG_DIRS[variant]
    output_dir = Path(args.output_dir) if args.output_dir else (configs.PROJECT_ROOT / configs.OUTPUT_DIRS[variant])
    
    # Create dataset YAML file for YOLO
    dataset_yaml = configs.PROJECT_ROOT / "dataset.yaml"
    create_dataset_yaml(dataset_yaml, dataset_dir)

    print("\n" + "=" * 60)
    print("YOLO26x Training")
    print("=" * 60)
    print(f"Dataset: {dataset_dir}")
    print(f"Log directory: {log_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Epochs: {args.epochs}, batch_size: {args.batch_size}, imgsz: {args.imgsz}")
    print("=" * 60 + "\n")

    # Handle resume/continue: load checkpoint as weights (avoids incompatible checkpoint args)
    if args.fresh or args.new:
        print("Fresh training requested (--fresh/--new), starting from scratch")
        model = YOLO(f'{configs.YOLO_VARIANTS[variant]}.pt')
        epochs = args.epochs
    elif args.resume:
        weights_path = Path(args.resume)
        if weights_path.exists():
            print(f"Loading checkpoint as weights (continue training): {weights_path}")
            model = YOLO(str(weights_path))
            ckpt_epoch = get_checkpoint_epoch(weights_path)
            epochs = max(1, args.epochs - ckpt_epoch - 1) if ckpt_epoch >= 0 else args.epochs
            if ckpt_epoch >= 0:
                print(f"Checkpoint at epoch {ckpt_epoch + 1}, training {epochs} more epochs (target total: {args.epochs})")
        else:
            print(f"Checkpoint not found: {weights_path}, starting from base model")
            model = YOLO(f'{configs.YOLO_VARIANTS[variant]}.pt')
            epochs = args.epochs
    else:
        weights_path = log_dir / "train" / "weights" / "last.pt"
        if weights_path.exists():
            print(f"Found existing weights, loading as model (continue training): {weights_path}")
            model = YOLO(str(weights_path))
            ckpt_epoch = get_checkpoint_epoch(weights_path)
            epochs = max(1, args.epochs - ckpt_epoch - 1) if ckpt_epoch >= 0 else args.epochs
            if ckpt_epoch >= 0:
                print(f"Checkpoint at epoch {ckpt_epoch + 1}, training {epochs} more epochs (target total: {args.epochs})")
        else:
            print("No existing weights found, starting fresh training")
            model = YOLO(f'{configs.YOLO_VARIANTS[variant]}.pt')
            epochs = args.epochs

    train_args = {
        "data": str(dataset_yaml),
        "epochs": epochs,
        "batch": args.batch_size,
        "imgsz": args.imgsz,
        "lr0": args.lr,
        "project": str(log_dir),
        "name": "train",
        "seed": args.seed,
        "device": "cuda",
    }

    # Train the model
    results = model.train(**train_args)
    
    # Copy best and last weights to output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = log_dir / "train" / "weights"
    if weights_dir.exists():
        if (weights_dir / "best.pt").exists():
            date_str = datetime.now().strftime("%Y-%m-%d")
            version = "26"  # YOLO26 version
            best_filename = f"yolo{version}{variant}-{args.epochs}e-best-{date_str}.pt"
            shutil.copy2(weights_dir / "best.pt", output_dir / best_filename)
        if (weights_dir / "last.pt").exists():
            shutil.copy2(weights_dir / "last.pt", output_dir / "last.pt")
        print(f"\nModel weights copied to {output_dir}")


def create_dataset_yaml(yaml_path: Path, dataset_dir: Path):
    """Create a YOLO dataset YAML file pointing to YOLO format dataset."""
    train_images = dataset_dir / "images" / "train"
    val_images = dataset_dir / "images" / "val"
    
    yaml_content = f"""# YOLO Dataset Configuration
# Dataset: Chicken Detection (YOLO format)

path: {dataset_dir.absolute()}
train: images/train
val: images/val

# Classes
nc: {configs.DATASET_CONFIG["num_classes"]}
names: {configs.DATASET_CONFIG["class_names"]}
"""
    yaml_path.write_text(yaml_content)
    print(f"Created dataset YAML: {yaml_path}")


if __name__ == "__main__":
    main()
