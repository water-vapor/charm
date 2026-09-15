#!/usr/bin/env bash
set -euo pipefail

COSE_CL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$COSE_CL_DIR/../_common.sh"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PROJECT_ROOT/arc1-cl-base-muon-600k}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-600000}"
CL_DATA_DIR="${CL_DATA_DIR:-$AUGMENTED_DATA_ROOT}"
CL_OUTPUT_ROOT="${CL_OUTPUT_ROOT:-$PROJECT_ROOT/outputs/cose_cl}"

usage() {
  cat >&2 <<'EOF'
usage: run.sh STREAM VARIANT METHOD [trainer args...]

STREAM:  arc2_training | arc2_eval
VARIANT: cose_lowrank | cose_fulltable | table_lowrank | table_fullrank | compo_only
         cose_lowrank_gate | cose_lowrank_t32 | compo_only_t32
BASE:    CHECKPOINT_DIR defaults to PROJECT_ROOT/arc1-cl-base-muon-600k;
         CHECKPOINT_STEP defaults to 600000.
METHOD:  frozen | naive | reset | joint | joint_all
         ewc_all | ewc_comp | l2sp_all | l2sp_comp | rehearsal_all | rehearsal_comp
EOF
}

run_cose_cl() {
  if (( $# < 3 )); then
    usage
    return 2
  fi

  local stream="$1"
  local variant="$2"
  local method="$3"
  shift 3

  local seed="${SEED:-0}"
  local order_seed="${ORDER_SEED:-$seed}"
  [[ "$seed" =~ ^[0-9]+$ ]] || {
    echo "SEED must be a non-negative integer" >&2
    return 2
  }
  [[ "$order_seed" =~ ^[0-9]+$ ]] || {
    echo "ORDER_SEED must be a non-negative integer" >&2
    return 2
  }

  local campaign stream_name default_project
  case "$stream" in
    arc2_training)
      campaign=arc2train
      stream_name=arc2_training_arc1_unseen_aug1000.parquet
      default_project=charm-cose-cl
      ;;
    arc2_eval)
      campaign=arc2eval
      stream_name=arc2_eval_aug1000.parquet
      default_project=charm-cose-cl-arc2eval
      ;;
    *)
      echo "unknown stream: $stream" >&2
      return 2
      ;;
  esac

  local checkpoint_name config_name
  case "$variant" in
    cose_lowrank)
      checkpoint_name=arc1-default-step_${CHECKPOINT_STEP}.pt
      config_name=default.yaml
      ;;
    cose_fulltable)
      checkpoint_name=arc1-default-fulltb-step_${CHECKPOINT_STEP}.pt
      config_name=default-fulltb.yaml
      ;;
    table_lowrank)
      checkpoint_name=arc1-lowrank-step_${CHECKPOINT_STEP}.pt
      config_name=lowrank.yaml
      ;;
    table_fullrank)
      checkpoint_name=arc1-fulltable-step_${CHECKPOINT_STEP}.pt
      config_name=full-table.yaml
      ;;
    compo_only)
      checkpoint_name=arc1-compo-only-step_${CHECKPOINT_STEP}.pt
      config_name=compo-only.yaml
      ;;
    cose_lowrank_gate)
      checkpoint_name=arc1-default-with-gate-step_${CHECKPOINT_STEP}.pt
      config_name=default-with-gate.yaml
      ;;
    cose_lowrank_t32)
      checkpoint_name=arc1-default-t32-step_${CHECKPOINT_STEP}.pt
      config_name=default-t32.yaml
      ;;
    compo_only_t32)
      checkpoint_name=arc1-compo-only-t32-step_${CHECKPOINT_STEP}.pt
      config_name=compo-only-t32.yaml
      ;;
    *)
      echo "unknown checkpoint variant: $variant" >&2
      return 2
      ;;
  esac

  local module
  local -a method_args
  case "$method" in
    frozen|naive|reset|joint|joint_all)
      module=charm.cose_cl.train
      method_args=(--arm "$method")
      ;;
    ewc_all|ewc_comp|l2sp_all|l2sp_comp|rehearsal_all|rehearsal_comp)
      module=charm.cose_cl.cl_baselines
      method_args=(--algo "${method%%_*}" --scope "${method##*_}")
      ;;
    *)
      echo "unknown method: $method" >&2
      return 2
      ;;
  esac

  local checkpoint="$CHECKPOINT_DIR/$checkpoint_name"
  local checkpoint_config="$CHECKPOINT_DIR/$config_name"
  local stream_parquet="${CL_STREAM_PARQUET:-$CL_DATA_DIR/$stream_name}"
  local method_tag="${method//_/-}"
  local variant_tag="${variant//_/-}"
  local run_name="${RUN_NAME:-cose-cl-$campaign-$variant_tag-$method_tag-seed$seed}"
  local output_dir="${CL_OUTPUT_DIR:-$CL_OUTPUT_ROOT/$campaign/$variant_tag/$method_tag/seed$seed}"

  run_charm "$module" \
    --ckpt "$checkpoint" \
    --ckpt_config "$checkpoint_config" \
    --stream_parquet "$stream_parquet" \
    --data_dir "$CL_DATA_DIR" \
    --out_dir "$output_dir" \
    --device cuda \
    --project_name "${WANDB_PROJECT:-$default_project}" \
    --run_name "$run_name" \
    --seed "$seed" \
    --order_seed "$order_seed" \
    --stage_steps 1000 \
    --batch_size 128 \
    --warmup_steps 20 \
    --embed_lr 1e-2 \
    --sparse_lr 1e-2 \
    --sparse_weight_decay 0.1 \
    --ft_lr 1e-4 \
    --ft_weight_decay 0.1 \
    --beta1 0.9 \
    --beta2 0.95 \
    --no_translation_ratio 0.2 \
    --eval_batch_size 256 \
    --eval_augs 128 \
    --eval_every 15 \
    --arc1_eval_puzzles 60 \
    --arc1_eval_augs 64 \
    "${method_args[@]}" \
    "$@"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  run_cose_cl "$@"
fi
