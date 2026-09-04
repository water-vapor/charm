#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$THIS_DIR/../../_common.sh"

CHECKPOINT_SOURCE="${CHECKPOINT_SOURCE:-$PROJECT_ROOT/checkpoints/ablations/component_grid_repeats/arc-agi-1/rearc-cose-gate-run1}"
RUN_NAME="${RUN_NAME:-${CHECKPOINT_SOURCE##*/}}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$COMPONENT_GRID_EVAL_OUTPUT_ROOT/arc-agi-1/$RUN_NAME}"
PROJECT_NAME="${PROJECT_NAME:-arc1-repeat-eval}"

run_charm charm.replay_eval \
  "data_paths=[$AUGMENTED_DATA_ROOT/arc1_training_aug1000.parquet,$AUGMENTED_DATA_ROOT/rearc_aug100.parquet,$AUGMENTED_DATA_ROOT/concept_aug1000.parquet,$AUGMENTED_DATA_ROOT/arc1_eval_aug1000.parquet]" \
  "data_paths_test=[$AUGMENTED_DATA_ROOT/arc1_eval_aug1000.parquet]" \
  "+data_path_multiplicities=[1,1,2,2]" \
  "+checkpoint_path=$EVAL_OUTPUT_DIR" \
  "+replay_checkpoint_source=$CHECKPOINT_SOURCE" \
  arch=urm_v2 \
  arch.mlp_type=convswiglu \
  arch.use_smart_embed=True \
  arch.smart_embed_source=slotperm \
  arch.smart_embed_strategy=film_concat \
  arch.smart_embed_interaction_mode=instance_residual \
  arch.smart_embed_interaction_rank=32 \
  arch.smart_embed_interaction_gate=True \
  +replay_start_step=510000 \
  +replay_end_step=600000 \
  +replay_save_submissions=True \
  +replay_submission_subdir=replay_eval \
  "+replay_halt_max_step_deltas=[]" \
  +replay_run_name_suffix= \
  optimizer=muon \
  "+run_name=$RUN_NAME" \
  "+project_name=$PROJECT_NAME" \
  ema=True \
  "$@"
