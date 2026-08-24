#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$THIS_DIR/../.." && pwd)"
OUTPUT_DIR="${1:-$PROJECT_ROOT/data/augmented/v2}"

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m charm.car.build_dataset "$OUTPUT_DIR" \
  --rules-per-family 500 \
  --train-pairs 500 \
  --eval-pairs 4 \
  --height 16 \
  --width 16 \
  --seed 0 \
  --workers "${WORKERS:-1}"
