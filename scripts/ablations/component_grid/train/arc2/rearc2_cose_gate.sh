#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$THIS_DIR/../../_common.sh"

SEED="${SEED:-0}"
if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
  echo "SEED must be a non-negative integer, got: $SEED" >&2
  exit 2
fi

RUN_NAME="${RUN_NAME:-rearc2-cose-gate-seed${SEED}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$COMPONENT_GRID_TRAIN_OUTPUT_ROOT/arc-agi-2/$RUN_NAME}"
PROJECT_NAME="${PROJECT_NAME:-arc2-component-grid-train}"

run_charm charm.train \
  "data_paths=[$AUGMENTED_DATA_ROOT/arc2_training_aug1000.parquet,$AUGMENTED_DATA_ROOT/rearc2_aug100.parquet,$AUGMENTED_DATA_ROOT/concept_aug1000.parquet,$AUGMENTED_DATA_ROOT/arc2_eval_aug1000.parquet]" \
  "data_paths_test=[$AUGMENTED_DATA_ROOT/arc2_eval_aug1000.parquet]" \
  "+data_path_multiplicities=[1,1,2,2]" \
  "+checkpoint_path=$CHECKPOINT_DIR" \
  arch=urm_v2 \
  arch.mlp_type=convswiglu \
  arch.use_smart_embed=True \
  arch.smart_embed_source=slotperm \
  arch.smart_embed_strategy=film_concat \
  arch.smart_embed_interaction_mode=instance_residual \
  arch.smart_embed_interaction_rank=32 \
  arch.smart_embed_interaction_gate=True \
  train_steps_override=1000000 \
  eval_interval_steps_override=10000 \
  checkpoint_every_eval=True \
  optimizer=muon \
  seed="$SEED" \
  "+run_name=$RUN_NAME" \
  "+project_name=$PROJECT_NAME" \
  ema=True \
  "$@"
