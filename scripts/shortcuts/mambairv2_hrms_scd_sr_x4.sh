#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"

shortcut_entry \
  scripts/train/sr.py \
  configs/runs/sr/mambairv2/hrms_scd_x4.yaml \
  "$@"
