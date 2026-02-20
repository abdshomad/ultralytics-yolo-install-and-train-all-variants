# Installation Guide

Complete step-by-step installation guide for Ultralytics YOLO training setup on chicken detection dataset.

## Prerequisites

Before starting, ensure you have:

- **Python 3.10+** installed
- **CUDA-capable GPU(s)** with NVIDIA drivers installed
- **Git** installed
- **`uv`** package manager installed ([Installation guide](https://github.com/astral-sh/uv))
- **nvidia-smi** available (for GPU monitoring)

### Verify Prerequisites

```bash
# Check Python version
python3 --version  # Should be 3.10 or higher

# Check CUDA availability
nvidia-smi  # Should show GPU information

# Check uv installation
uv --version
```

## Installation Steps

### Step 1: Clone Repository and Initialize Submodules

```bash
# If cloning the repository for the first time
git clone <repository-url>
cd ultralytics-yolo-install-and-train-all-variants

# Initialize git submodules (dataset)
git submodule update --init --recursive
```

**Important:** The repository uses git submodules for:
- `chicken-detection-labelme-format/` - The dataset

### Step 2: Set Up Python Virtual Environment

```bash
# Create virtual environment using uv
uv venv

# Sync dependencies from pyproject.toml
uv sync
```

This will:
- Create a virtual environment in `.venv/`
- Install `ultralytics`, `supervision`, and all transitive dependencies (PyTorch, etc.)

### Step 3: Activate Virtual Environment (Optional)

While `uv run` can execute commands directly, you can also activate the environment:

```bash
source .venv/bin/activate
# On Windows: .venv\Scripts\activate
```

### Step 4: Verify Configuration

Verify that all configuration and data files are in place:

```bash
# Check dataset files exist
ls chicken-detection-labelme-format/yolo-format/train/images
ls chicken-detection-labelme-format/yolo-format/train/labels
ls chicken-detection-labelme-format/yolo-format/val/images
ls chicken-detection-labelme-format/yolo-format/val/labels

# Check training scripts exist
ls train-yolo-*.py
```

### Step 5: Verify Dataset Structure

Ensure your dataset follows this structure:

```
chicken-detection-labelme-format/
└── yolo-format/
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

## Testing Installation

### Quick Test

Run a quick test to verify everything is set up correctly:

```bash
# Test import of key modules
uv run python -c "
import torch
from ultralytics import YOLO
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('Ultralytics YOLO OK')
"
```

### Test Training Script

Verify the training script can load:

```bash
# This should print configuration (will fail at dataset load if dataset missing)
uv run python train-yolo-s.py --epochs 1 2>&1 | head -30
```

## First Training Run

### Free GPU Memory (Optional)

Before training, you may want to free GPU memory:

```bash
# If free_gpu.py exists
uv run python free_gpu.py --kill
```

### Start Training

```bash
# Train Small variant (100 epochs by default)
uv run train-yolo-s.py

# Or with custom epochs
uv run train-yolo-s.py --epochs 50 --batch-size 8

# Train all variants sequentially
./train-yolo-all-variants.sh
```

## Troubleshooting

### Common Issues

#### 1. Git Submodules Not Initialized

**Error:** Dataset directory not found

**Solution:**
```bash
git submodule update --init --recursive
```

#### 2. CUDA Out of Memory (OOM)

**Error:** `torch.OutOfMemoryError: CUDA out of memory`

**Solutions:**
1. Reduce `--batch-size` (e.g., 2 or 1)
2. Increase `--grad-accum-steps` to maintain effective batch size
3. Use a smaller model variant (nano or small)

#### 3. Missing Python Dependencies

**Error:** `ModuleNotFoundError: No module named 'ultralytics'`

**Solution:**
```bash
uv sync
```

#### 4. Dataset Format Error

**Error:** `Could not detect dataset format`

**Solution:** Ensure YOLO format structure exists with `train/images/` and `train/labels/` directories. Each image should have a corresponding `.txt` label file with normalized coordinates.

### Getting Help

If you encounter issues not covered here:

1. Check the error message carefully
2. Verify all prerequisites are met
3. Ensure submodules and dependencies are installed
4. Check GPU memory with `nvidia-smi`
5. Review [Ultralytics YOLO documentation](https://docs.ultralytics.com)

## Dependency Management

### Adding Dependencies

```bash
uv add <package-name>
```

### Updating Dependencies

```bash
uv sync
```

## Installation Checklist

- [ ] Python 3.10+ installed
- [ ] CUDA and GPU drivers installed (`nvidia-smi` works)
- [ ] `uv` package manager installed
- [ ] Repository cloned
- [ ] Git submodules initialized (`chicken-detection-labelme-format/`)
- [ ] Python virtual environment created (`uv venv`)
- [ ] Dependencies installed (`uv sync`)
- [ ] Dataset files present (YOLO format with images/ and labels/ directories)
- [ ] Training scripts present (`train-yolo-*.py`)

## Next Steps

1. **Start training:** `uv run train-yolo-s.py`
2. **Monitor training:** Training logs are saved in `yolo-models/yolo-{variant}/logs/train/` directory
3. **Run inference:** `uv run run_test_detection.py --variant s`
