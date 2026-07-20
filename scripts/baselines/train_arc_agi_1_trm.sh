#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$THIS_DIR/../_common.sh"

run_charm charm.train \
  "data_paths=[$AUGMENTED_DATA_ROOT/arc1_training_aug1000.parquet,$AUGMENTED_DATA_ROOT/concept_aug1000.parquet,$AUGMENTED_DATA_ROOT/arc1_eval_aug1000.parquet]" \
  "data_paths_test=[$AUGMENTED_DATA_ROOT/arc1_eval_aug1000.parquet]" \
  "+checkpoint_path=$PROJECT_ROOT/outputs/runs/baseline-arc-agi-1-trm" \
  arch=trm \
  arch.L_layers=2 \
  arch.H_cycles=3 \
  arch.L_cycles=4 \
  train_steps_override=600000 \
  eval_interval_steps_override=10000 \
  +run_name=baseline-arc-agi-1-trm \
  +project_name=charm \
  ema=True \
  "$@"
