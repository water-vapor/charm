#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${CHARM_DATA_DIR:-$PROJECT_ROOT/data}}"
AUGMENTED_DATA_ROOT="${AUGMENTED_DATA_ROOT:-$DATA_ROOT/augmented/v2}"

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export DATA_ROOT
export AUGMENTED_DATA_ROOT
export WANDB_MODE="${WANDB_MODE:-disabled}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"

if [[ -z "${NUM_GPUS:-}" ]]; then
  NUM_GPUS="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
fi

if [[ "$NUM_GPUS" -lt 1 ]]; then
  echo "No CUDA GPUs detected. Set NUM_GPUS explicitly if running under a scheduler." >&2
  exit 1
fi

run_charm() {
  local module="$1"
  shift
  WANDB_MODE="$WANDB_MODE" OMP_NUM_THREADS="$OMP_NUM_THREADS" \
    torchrun --nproc-per-node "$NUM_GPUS" --module "$module" "$@"
}
