#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$THIS_DIR/../../_common.sh"

run_charm charm.train \
  "data_paths=[$AUGMENTED_DATA_ROOT/arc1_training_aug1000.parquet,$AUGMENTED_DATA_ROOT/rearc_aug100.parquet,$AUGMENTED_DATA_ROOT/concept_aug1000.parquet,$AUGMENTED_DATA_ROOT/arc1_eval_aug1000.parquet]" \
  "+data_path_multiplicities=[1,1,2,2]" \
  "data_paths_test=[$AUGMENTED_DATA_ROOT/arc1_eval_aug1000.parquet]" \
  "+checkpoint_path=$PROJECT_ROOT/outputs/runs/ablation-task-memory-charm-rank32-gate" \
  arch=urm_v2 \
  arch.mlp_type=convswiglu \
  arch.use_smart_embed=True \
  arch.smart_embed_source=slotperm \
  arch.smart_embed_strategy=film_concat \
  arch.smart_embed_interaction_mode=instance_residual \
  arch.smart_embed_interaction_rank=32 \
  arch.smart_embed_interaction_gate=True \
  train_steps_override=600000 \
  eval_interval_steps_override=10000 \
  optimizer=muon \
  +run_name=ablation-task-memory-charm-rank32-gate \
  +project_name=charm \
  ema=True \
  "$@"
