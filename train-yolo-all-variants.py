#!/usr/bin/env python3
"""
Train all YOLO26 variants (n, s, m, l, x) on chicken detection dataset.

Runs training sequentially for each variant.

Usage:
    uv run train-yolo-all-variants.py [--variants n s m l x] [--epochs N] [--batch-size N]
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch

import configs

VARIANTS = ["n", "s", "m", "l", "x"]
VARIANT_SCRIPTS = {
    "n": "train-yolo-n.py",
    "s": "train-yolo-s.py",
    "m": "train-yolo-m.py",
    "l": "train-yolo-l.py",
    "x": "train-yolo-x.py",
}


def main():
    parser = argparse.ArgumentParser(
        description="Train all YOLO26 variants (n, s, m, l, x) on chicken detection dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--epochs", type=int, default=configs.DEFAULT_TRAINING_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=configs.DEFAULT_TRAINING_CONFIG["batch_size"])
    parser.add_argument("--imgsz", type=int, default=configs.DEFAULT_TRAINING_CONFIG["imgsz"])
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    project_root = configs.PROJECT_ROOT
    num_gpus = torch.cuda.device_count()

    print("\n" + "=" * 60)
    print("YOLO26 All Variants Training")
    print("=" * 60)
    print(f"Variants: {args.variants}")
    print(f"GPUs: {num_gpus} available")
    print(f"Epochs: {args.epochs}, batch_size: {args.batch_size}, imgsz: {args.imgsz}")
    print("=" * 60 + "\n")

    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    
    results = []
    for variant in args.variants:
        script = VARIANT_SCRIPTS[variant]
        script_path = project_root / script
        if not script_path.exists():
            print(f"Error: Script not found: {script_path}")
            results.append((variant, 1))
            continue

        cmd = [
            sys.executable,
            str(script_path),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--imgsz",
            str(args.imgsz),
        ]
        if args.resume:
            cmd.extend(["--resume", args.resume])

        # Create variant-specific log directory and log file
        variant_log_dir = project_root / configs.LOG_DIRS[variant]
        variant_log_dir.mkdir(parents=True, exist_ok=True)
        log_file = variant_log_dir / f"train-yolo-{variant}-{timestamp}.log"
        
        print("\n" + "-" * 60)
        print(f"Training YOLO26{variant.upper()}")
        print("-" * 60)
        print(f"Running: {' '.join(cmd)}")
        print(f"Log file: {log_file}\n")

        # Redirect output to variant-specific log file
        with open(log_file, "w") as log:
            result = subprocess.run(
                cmd,
                cwd=str(project_root),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True
            )
        
        results.append((variant, result.returncode))
        
        # Print summary to stdout
        if result.returncode == 0:
            print(f"✓ YOLO26{variant.upper()} completed successfully. Log: {log_file}")
        else:
            print(f"✗ YOLO26{variant.upper()} failed with code {result.returncode}. Log: {log_file}")

    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    for variant, code in results:
        status = "OK" if code == 0 else f"FAILED ({code})"
        print(f"  YOLO26{variant.upper()}: {status}")
    print("=" * 60 + "\n")

    exit_code = next((c for _, c in results if c != 0), 0)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
