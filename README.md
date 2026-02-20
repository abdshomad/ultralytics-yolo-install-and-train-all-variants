# Ultralytics YOLO Trainer

Training setup for Ultralytics YOLO on chicken detection dataset using YOLO format annotations.

## Overview

This repository provides a streamlined setup for training Ultralytics YOLO26 models on a custom chicken detection dataset. The dataset contains 2 classes: `chicken` and `not-chicken`, formatted in YOLO annotation format. YOLO26 is a state-of-the-art object detection model developed by Ultralytics, offering multiple size variants for different speed/accuracy trade-offs.

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU(s)
- Git submodules initialized

### Setup

1. **Initialize git submodules:**
   ```bash
   git submodule update --init --recursive
   ```

2. **Create virtual environment and install dependencies:**
   ```bash
   uv venv
   uv sync
   ```

3. **Activate virtual environment (optional):**
   ```bash
   source .venv/bin/activate
   ```

### Dataset Structure

The dataset is located in `chicken-detection-labelme-format/yolo-format/`:

```
yolo-format/
├── train/
│   ├── images/
│   │   └── [training images]
│   └── labels/
│       └── [.txt label files]
├── val/
│   ├── images/
│   │   └── [validation images]
│   └── labels/
│       └── [.txt label files]
└── test/
    ├── images/
    │   └── [test images]
    └── labels/
        └── [.txt label files]
```

**Dataset Configuration:**
- **Training images:** `chicken-detection-labelme-format/yolo-format/train/images/`
- **Training labels:** `chicken-detection-labelme-format/yolo-format/train/labels/`
- **Validation images:** `chicken-detection-labelme-format/yolo-format/val/images/`
- **Validation labels:** `chicken-detection-labelme-format/yolo-format/val/labels/`
- **Number of classes:** 2 (chicken, not-chicken)

**YOLO Label Format:**
Each image has a corresponding `.txt` file with the same name. Each line in the label file contains:
```
class_id center_x center_y width height
```
All coordinates are normalized (0-1).

## Training

### Train All Variants (Recommended)

Train all five model variants (n, s, m, l, x) sequentially:

```bash
# Using the shell script
./train-yolo-all-variants.sh

# Or directly with uv
uv run train-yolo-all-variants.py
```

Options:
- `--variants n s m l x` — Train specific variants only (default: all)
- `--epochs N` — Number of training epochs (default: 100)
- `--batch-size N` — Batch size (default: 16)
- `--imgsz N` — Image size (default: 640)

### Single Variant Training

Train any of the five model variants:

```bash
# Nano variant (fastest, smallest)
uv run train-yolo-n.py

# Small variant
uv run train-yolo-s.py

# Medium variant
uv run train-yolo-m.py

# Large variant
uv run train-yolo-l.py

# XLarge variant (slowest, most accurate)
uv run train-yolo-x.py
```

### Command-Line Options (single-variant scripts)

| Option | Description | Default |
|--------|-------------|---------|
| `--epochs N` | Number of training epochs | 100 |
| `--batch-size N` | Batch size per GPU | 16 |
| `--imgsz N` | Image size for training | 640 |
| `--lr` | Learning rate | 0.01 |
| `--resume PATH` | Resume training from checkpoint | None |
| `--output-dir DIR` | Output directory for checkpoints | From configs.py |
| `--seed N` | Random seed | 0 |

### Multi-GPU Training

YOLO automatically supports multi-GPU training. Set `CUDA_VISIBLE_DEVICES` to specify GPUs:

```bash
# Example: Train Small variant on 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run train-yolo-s.py --epochs 100
```

### Examples

#### Train All Variants
```bash
./train-yolo-all-variants.sh
```

#### Train Specific Variants Only
```bash
uv run train-yolo-all-variants.py --variants n s m
```

#### Resume Training
```bash
uv run train-yolo-m.py --resume yolo-models/yolo-m/last.pt
```

## Inference / Test Detection

Run detection on test images using a trained model:

```bash
# Default: use YOLO26s, latest checkpoint
uv run run_test_detection.py

# Specify variant and checkpoint
uv run run_test_detection.py --variant m --checkpoint yolo-models/yolo-m/best.pt

# Adjust confidence threshold
uv run run_test_detection.py --variant s --conf 0.3
```

Results are saved to `chicken-detection-labelme-format/yolo-format/test-result/`.

## Configuration

### Global Configuration (`configs.py`)

The `configs.py` file contains global settings used by all training scripts:

- **Dataset paths:** Dataset directory and annotation files
- **Number of classes:** 2 (chicken, not-chicken)
- **YOLO variants:** n, s, m, l, x
- **Output directories:** Where checkpoints are saved
- **Log directories:** Where training logs are saved (separate per variant)
- **Default training settings:** epochs, batch_size, imgsz, lr

### Output Directories

Trained models and checkpoints are saved to:
- **Nano:** `yolo-models/yolo-n/`
- **Small:** `yolo-models/yolo-s/`
- **Medium:** `yolo-models/yolo-m/`
- **Large:** `yolo-models/yolo-l/`
- **XLarge:** `yolo-models/yolo-x/`

Checkpoint files:
- `best.pt` — Best model based on validation metrics
- `last.pt` — Most recent checkpoint (for resuming)

### Log Directories

Training logs are separated by variant:
- **Nano:** `yolo-models/yolo-n/logs/train/`
- **Small:** `yolo-models/yolo-s/logs/train/`
- **Medium:** `yolo-models/yolo-m/logs/train/`
- **Large:** `yolo-models/yolo-l/logs/train/`
- **XLarge:** `yolo-models/yolo-x/logs/train/`

Each log directory contains:
- `weights/` — Training checkpoints
- `results.csv` — Training metrics
- `confusion_matrix.png` — Confusion matrix
- `results.png` — Training curves
- TensorBoard logs (if enabled)

## Project Structure

```
ultralytics-yolo-install/
├── configs.py                    # Global configuration
├── train-yolo-all-variants.py    # Train all variants
├── train-yolo-all-variants.sh    # Shell wrapper
├── train-yolo-n.py               # Nano variant
├── train-yolo-s.py               # Small variant
├── train-yolo-m.py               # Medium variant
├── train-yolo-l.py               # Large variant
├── train-yolo-x.py               # XLarge variant
├── run_test_detection.py         # Inference on test images
├── dataset.yaml                  # Dataset configuration (auto-generated)
├── chicken-detection-labelme-format/  # Dataset submodule
│   └── yolo-format/              # YOLO format dataset
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       ├── val/
│       │   ├── images/
│       │   └── labels/
│       └── test/
│           ├── images/
│           └── labels/
├── yolo-models/                  # Trained models (gitignored)
│   ├── yolo-n/
│   ├── yolo-s/
│   ├── yolo-m/
│   ├── yolo-l/
│   └── yolo-x/
├── logs/                         # Training logs (gitignored)
│   ├── yolo-n/
│   ├── yolo-s/
│   ├── yolo-m/
│   ├── yolo-l/
│   └── yolo-x/
└── pyproject.toml                # Python dependencies
```

## Dependencies

Dependencies are managed via `uv` and `pyproject.toml`. Install with:

```bash
uv sync
```

Key dependencies: `ultralytics`, `supervision`.

## Contributing

This repository uses git submodules. When updating submodules:

```bash
git submodule update --remote
```

**Important:** Do not modify files within submodule directories directly. See `AGENTS.md` for more information.

## References

- [Ultralytics YOLO GitHub](https://github.com/ultralytics/ultralytics)
- [Ultralytics YOLO Documentation](https://docs.ultralytics.com)
- [YOLO26 Paper](https://arxiv.org/abs/2305.09972)

## License

Ultralytics YOLO is licensed under AGPL-3.0. See the [Ultralytics repository](https://github.com/ultralytics/ultralytics) for details.
