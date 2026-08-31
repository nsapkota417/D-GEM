#!/usr/bin/env bash
set -euo pipefail

# Activate your Python environment before running this script.

python src/train.py -cfg cfg/data/base.yaml \
  --task-type video \
  --test-csv /path/to/test.csv \
  --use-memory
