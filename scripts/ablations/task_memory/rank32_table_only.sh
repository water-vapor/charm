#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$THIS_DIR/../../_common.sh"

run_charm charm.train \
  "data_paths=[$AUGMENTED_DATA_ROOT/arc1_training_aug1000.parquet,$AUGMENTED_DATA_ROOT/rearc_aug100.parquet,$AUGMENTED_DATA_ROOT/concept_aug1000.parquet,$AUGMENTED_DATA_ROOT/arc1_eval_aug1000.parquet]" \
  "+data_path_multiplicities=[1,1,2,2]" \
  "data_paths_test=[$AUGMENTED_DATA_ROOT/arc1_eval_aug1000.parquet]" \
  "+checkpoint_path=$PROJECT_ROOT/outputs/runs/ablation-task-memory-rank32-table-only" \
  arch=urm_v2 \
  arch.mlp_type=convswiglu \
  arch.trm_embedding_mode=lowrank \
  arch.puzzle_emb_lowrank_dim=32 \
  train_steps_override=600000 \
  eval_interval_steps_override=10000 \
  optimizer=muon \
  +run_name=ablation-task-memory-rank32-table-only \
  +project_name=charm \
  ema=True \
  "$@"
