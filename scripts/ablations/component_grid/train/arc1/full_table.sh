#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$THIS_DIR/../../_common.sh"

SEED="${SEED:-0}"
if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
  echo "SEED must be a non-negative integer, got: $SEED" >&2
  exit 2
fi

RUN_NAME="${RUN_NAME:-full-table-seed${SEED}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$COMPONENT_GRID_TRAIN_OUTPUT_ROOT/arc-agi-1/$RUN_NAME}"
PROJECT_NAME="${PROJECT_NAME:-arc1-component-grid-train}"

run_charm charm.train \
  "data_paths=[$AUGMENTED_DATA_ROOT/arc1_training_aug1000.parquet,$AUGMENTED_DATA_ROOT/concept_aug1000.parquet,$AUGMENTED_DATA_ROOT/arc1_eval_aug1000.parquet]" \
  "data_paths_test=[$AUGMENTED_DATA_ROOT/arc1_eval_aug1000.parquet]" \
  "+checkpoint_path=$CHECKPOINT_DIR" \
  arch=urm_v2 \
  arch.mlp_type=convswiglu \
  arch.use_smart_embed=False \
  train_steps_override=600000 \
  eval_interval_steps_override=10000 \
  checkpoint_every_eval=True \
  optimizer=muon \
  seed="$SEED" \
  "+run_name=$RUN_NAME" \
  "+project_name=$PROJECT_NAME" \
  ema=True \
  "$@"
