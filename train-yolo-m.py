#!/usr/bin/env python3
"""
Train YOLO26m model on chicken detection dataset.

Usage:
    uv run train-yolo-m.py [--epochs N] [--batch-size N] [--imgsz N] [--resume PATH] [--output-dir DIR]
"""

import argparse
from pathlib import Path

import configs
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO26m on chicken detection dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=configs.DEFAULT_TRAINING_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=configs.DEFAULT_TRAINING_CONFIG["batch_size"])
    parser.add_argument("--imgsz", type=int, default=configs.DEFAULT_TRAINING_CONFIG["imgsz"])
    parser.add_argument("--lr", type=float, default=configs.DEFAULT_TRAINING_CONFIG["lr"])
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=configs.DEFAULT_TRAINING_CONFIG["seed"])
    args = parser.parse_args()

    variant = "m"
    dataset_dir = configs.PROJECT_ROOT / configs.DATASET_CONFIG["dataset_dir"]
    log_dir = configs.PROJECT_ROOT / configs.LOG_DIRS[variant]
    output_dir = Path(args.output_dir) if args.output_dir else (configs.PROJECT_ROOT / configs.OUTPUT_DIRS[variant])
    
    # Create dataset YAML file for YOLO
    dataset_yaml = configs.PROJECT_ROOT / "dataset.yaml"
    create_dataset_yaml(dataset_yaml, dataset_dir)

    print("\n" + "=" * 60)
    print("YOLO26m Training")
    print("=" * 60)
    print(f"Dataset: {dataset_dir}")
    print(f"Log directory: {log_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Epochs: {args.epochs}, batch_size: {args.batch_size}, imgsz: {args.imgsz}")
    print("=" * 60 + "\n")

    model = YOLO(f'{configs.YOLO_VARIANTS[variant]}.pt')
    
    train_args = {
        "data": str(dataset_yaml),
        "epochs": args.epochs,
        "batch": args.batch_size,
        "imgsz": args.imgsz,
        "lr0": args.lr,
        "project": str(log_dir),
        "name": "train",
        "seed": args.seed,
        "device": "cuda",
    }
    
    if args.resume:
        train_args["resume"] = args.resume
    
    # Train the model
    results = model.train(**train_args)
    
    # Copy best and last weights to output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = log_dir / "train" / "weights"
    if weights_dir.exists():
        import shutil
        if (weights_dir / "best.pt").exists():
            shutil.copy2(weights_dir / "best.pt", output_dir / "best.pt")
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
