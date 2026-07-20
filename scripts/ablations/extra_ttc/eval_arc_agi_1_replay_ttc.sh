#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$THIS_DIR/../../_common.sh"

run_charm charm.replay_eval \
  "data_paths=[$AUGMENTED_DATA_ROOT/arc1_training_aug1000.parquet,$AUGMENTED_DATA_ROOT/rearc_aug100.parquet,$AUGMENTED_DATA_ROOT/concept_aug1000.parquet,$AUGMENTED_DATA_ROOT/arc1_eval_aug1000.parquet]" \
  "+data_path_multiplicities=[1,1,2,2]" \
  "data_paths_test=[$AUGMENTED_DATA_ROOT/arc1_eval_aug1000.parquet]" \
  "+checkpoint_path=$PROJECT_ROOT/outputs/replay/arc-agi-1-16-24-hl10" \
  "+replay_checkpoint_source=$PROJECT_ROOT/checkpoints/arc-agi-1" \
  arch=urm_v2 \
  arch.mlp_type=convswiglu \
  arch.use_smart_embed=True \
  arch.smart_embed_source=slotperm \
  arch.smart_embed_strategy=film_concat \
  arch.smart_embed_interaction_mode=instance_residual \
  arch.smart_embed_interaction_rank=32 \
  arch.smart_embed_interaction_gate=True \
  "+replay_halt_max_step_deltas=[0,8]" \
  +replay_start_step=300000 \
  +replay_end_step=600001 \
  eval_exp_decay=True \
  eval_decay_half_life=10.0 \
  +run_name=arc-agi-1-replay-16-24-hl10 \
  +project_name=charm \
  ema=True \
  "$@"
