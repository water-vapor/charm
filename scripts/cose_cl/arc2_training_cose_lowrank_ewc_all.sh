#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$THIS_DIR/run.sh"

run_cose_cl arc2_training cose_lowrank ewc_all \
  --reg_lambda 1e-2 \
  --fisher_batches 64 \
  "$@"
