#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$THIS_DIR/../_common.sh"

run_charm charm.train \
  "data_paths=[$AUGMENTED_DATA_ROOT/arc2_training_aug1000.parquet,$AUGMENTED_DATA_ROOT/concept_aug1000.parquet,$AUGMENTED_DATA_ROOT/arc2_eval_aug1000.parquet]" \
  "data_paths_test=[$AUGMENTED_DATA_ROOT/arc2_eval_aug1000.parquet]" \
  "+checkpoint_path=$PROJECT_ROOT/outputs/runs/baseline-arc-agi-2-urm-muon" \
  arch=urm \
  train_steps_override=1000000 \
  eval_interval_steps_override=10000 \
  optimizer=muon \
  +run_name=baseline-arc-agi-2-urm-muon \
  +project_name=charm \
  ema=True \
  "$@"
