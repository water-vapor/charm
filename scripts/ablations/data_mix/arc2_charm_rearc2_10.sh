#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$THIS_DIR/../../_common.sh"

run_charm charm.train \
  "data_paths=[$AUGMENTED_DATA_ROOT/arc2_training_aug1000.parquet,$AUGMENTED_DATA_ROOT/rearc2_aug10.parquet,$AUGMENTED_DATA_ROOT/concept_aug1000.parquet,$AUGMENTED_DATA_ROOT/arc2_eval_aug1000.parquet]" \
  "+data_path_multiplicities=[1,1,2,2]" \
  "data_paths_test=[$AUGMENTED_DATA_ROOT/arc2_eval_aug1000.parquet]" \
  "+checkpoint_path=$PROJECT_ROOT/outputs/runs/ablation-data-arc2-charm-rearc2-10" \
  arch=urm_v2 \
  arch.mlp_type=convswiglu \
  arch.use_smart_embed=True \
  arch.smart_embed_source=slotperm \
  arch.smart_embed_strategy=film_concat \
  arch.smart_embed_interaction_mode=instance_residual \
  arch.smart_embed_interaction_rank=32 \
  arch.smart_embed_interaction_gate=False \
  train_steps_override=1000000 \
  eval_interval_steps_override=10000 \
  optimizer=muon \
  +run_name=ablation-data-arc2-charm-rearc2-10 \
  +project_name=charm \
  ema=True \
  "$@"
