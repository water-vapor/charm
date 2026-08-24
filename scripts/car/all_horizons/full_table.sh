#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$THIS_DIR/../run.sh"
run_car all_horizons full_table "$@"
