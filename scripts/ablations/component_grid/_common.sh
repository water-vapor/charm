#!/usr/bin/env bash
set -euo pipefail

COMPONENT_GRID_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$COMPONENT_GRID_DIR/../../_common.sh"

# Canonical output roots for the manuscript component-grid jobs.
COMPONENT_GRID_TRAIN_OUTPUT_ROOT="${COMPONENT_GRID_TRAIN_OUTPUT_ROOT:-$PROJECT_ROOT/outputs/ablations/component_grid/train}"
COMPONENT_GRID_EVAL_OUTPUT_ROOT="${COMPONENT_GRID_EVAL_OUTPUT_ROOT:-$PROJECT_ROOT/outputs/ablations/component_grid/eval}"
