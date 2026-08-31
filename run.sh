#!/usr/bin/env bash
set -euo pipefail

# Activate your Python environment before running this script.

python src/train.py -cfg cfg/data/base.yaml \
  --task-type video \
  --train-csv /path/to/train.csv \
  --test-csv /path/to/test.csv \
  --use-memory
# For independent images, use: --task-type image --no-memory
