#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$THIS_DIR/../../_common.sh"

run_charm charm.train \
  "data_paths=[$AUGMENTED_DATA_ROOT/arc1_training_aug1000.parquet]" \
  "data_paths_test=[$AUGMENTED_DATA_ROOT/arc1_training_aug1000.parquet]" \
  "+checkpoint_path=$PROJECT_ROOT/outputs/runs/ablation-backbone-arc1-l2-r12-h12" \
  arch=urm_v2 \
  arch.num_layers=2 \
  arch.recurrent_steps=12 \
  arch.tbptt_layer_steps=12 \
  arch.early_stop_training=False \
  arch.loops=1 \
  arch.mlp_type=convswiglu \
  checkpoint_every_eval=False \
  train_steps_override=70000 \
  eval_interval_steps_override=10000 \
  +run_name=ablation-backbone-arc1-l2-r12-h12 \
  +project_name=charm \
  ema=True \
  "$@"
