#!/usr/bin/env bash
set -euo pipefail

# 用途：训练 HRMS-SCD x4 的 MASA-SR 真实参考图 RefSR 基线。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_common.sh"

shortcut_entry \
  scripts/train/refsr.py \
  configs/runs/refsr/masa_sr/hrms_scd_x4.yaml \
  "$@"
