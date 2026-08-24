#!/usr/bin/env bash
set -euo pipefail

CAR_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$CAR_SCRIPT_DIR/../_common.sh"

CAR_DATA_DIR="${CAR_DATA_DIR:-$AUGMENTED_DATA_ROOT}"
CAR_OUTPUT_ROOT="${CAR_OUTPUT_ROOT:-$PROJECT_ROOT/outputs/car}"

usage() {
  cat >&2 <<'EOF'
usage: run.sh DATASET ARM [trainer args...]

DATASET: single_horizon | all_horizons
ARM:     full_table | lowrank_table
         two_factor_composition | two_factor_cose
         three_factor_composition | three_factor_cose
EOF
}

run_car() {
  if (( $# < 2 )); then
    usage
    return 2
  fi

  local dataset="$1"
  local arm="$2"
  shift 2

  local train_file eval_file train_steps
  case "$dataset" in
    single_horizon)
      train_file=car_single_horizon_train.parquet
      train_steps=500000
      case "$arm" in
        two_factor_composition|two_factor_cose)
          eval_file=car_single_horizon_eval.parquet
          ;;
        full_table|lowrank_table|three_factor_composition|three_factor_cose)
          eval_file=car_all_horizons_eval.parquet
          ;;
        *) ;;
      esac
      ;;
    all_horizons)
      train_file=car_all_horizons_train.parquet
      eval_file=car_all_horizons_eval.parquet
      train_steps=292000
      ;;
    *)
      echo "unknown CAR dataset: $dataset" >&2
      return 2
      ;;
  esac

  case "$arm" in
    full_table|lowrank_table|two_factor_composition|two_factor_cose|three_factor_composition|three_factor_cose) ;;
    *)
      echo "unknown CAR arm: $arm" >&2
      return 2
      ;;
  esac

  local train_path="$CAR_DATA_DIR/$train_file"
  local eval_path="$CAR_DATA_DIR/$eval_file"
  for path in "$train_path" "$eval_path"; do
    if [[ ! -f "$path" ]]; then
      echo "missing CAR dataset: $path" >&2
      return 2
    fi
  done

  local seed="${SEED:-0}"
  [[ "$seed" =~ ^[0-9]+$ ]] || {
    echo "SEED must be a non-negative integer" >&2
    return 2
  }
  local global_batch_size="${GLOBAL_BATCH_SIZE:-1536}"
  if (( global_batch_size < 1 || global_batch_size % NUM_GPUS != 0 )); then
    echo "GLOBAL_BATCH_SIZE must be positive and divisible by NUM_GPUS=$NUM_GPUS" >&2
    return 2
  fi

  local run_name="${RUN_NAME:-car-$dataset-$arm-seed$seed}"
  local project_name="${WANDB_PROJECT:-charm-car-$dataset}"
  local output_dir="${CAR_OUTPUT_DIR:-$CAR_OUTPUT_ROOT/$dataset/$arm/seed$seed}"

  run_charm charm.car.train \
    arch=car \
    "arch.car_task_memory_mode=$arm" \
    "data_paths=[$train_path]" \
    "data_paths_test=[$eval_path]" \
    "+pair_types=[train]" \
    "train_steps_override=$train_steps" \
    eval_interval_steps_override=20000 \
    checkpoint_every_eval=false \
    "global_batch_size=$global_batch_size" \
    grad_accum_steps=1 \
    +dataloader_num_workers=2 \
    no_translation_ratio=0.2 \
    eval_passes=1 \
    "seed=$seed" \
    "+project_name=$project_name" \
    "+run_name=$run_name" \
    "+checkpoint_path=$output_dir" \
    "$@"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  run_car "$@"
fi
