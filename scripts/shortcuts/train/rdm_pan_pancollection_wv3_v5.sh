#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_common.sh"

shortcut_entry \
  scripts/train/rdm_refsr.py \
  configs/runs/rdm_refsr/pancollection_wv3_pan_v5_x4.yaml \
  "$@"
