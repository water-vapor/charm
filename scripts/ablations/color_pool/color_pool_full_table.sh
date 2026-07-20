#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$THIS_DIR/../../_common.sh"

run_charm charm.train \
  "data_paths=[$AUGMENTED_DATA_ROOT/arc1_training_aug1000_cppool1000.parquet]" \
  "data_paths_test=[$AUGMENTED_DATA_ROOT/arc1_training_aug1000_cppool1000.parquet]" \
  "+checkpoint_path=$PROJECT_ROOT/outputs/runs/ablation-colorpool-full-table" \
  arch=trm \
  arch.L_layers=2 \
  arch.H_cycles=3 \
  arch.L_cycles=4 \
  +run_name=ablation-colorpool-full-table \
  +project_name=charm \
  ema=True \
  "$@"
