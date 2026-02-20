#!/bin/bash
# Train all YOLO variants with separate log files for each variant
# Each variant's training output will be saved to logs/train-yolo-{variant}-{timestamp}.log
# The summary output will be displayed on stdout and can be redirected if needed

uv run train-yolo-all-variants.py "$@"
