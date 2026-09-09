#!/usr/bin/env bash
set -euo pipefail

# 用途：训练 HRMS-SCD x4 的 RefSRWKV-SR 模型，使用 LR 上采样自参考。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_common.sh"

shortcut_entry \
  scripts/train/refsrwkv.py \
  configs/runs/refsrwkv/hrms_scd_sr_x4.yaml \
  "$@"
