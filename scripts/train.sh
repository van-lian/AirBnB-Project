#!/usr/bin/env bash
set -euo pipefail
python "$(dirname "$0")/../src/training/train_model.py" --holdout_months "${1:-12}"

