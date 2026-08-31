#!/usr/bin/env bash
set -euo pipefail

# Activate your Python environment before running this script.

python src/infer.py -cfg cfg/data/base.yaml \
  --task-type video \
  --support-csv /path/to/support.csv \
  --test-csv /path/to/test.csv \
  --weights /path/to/model.pt \
  --save-preds
