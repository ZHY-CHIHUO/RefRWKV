#!/usr/bin/env bash
set -euo pipefail

# 用途：从头训练 HRMS-SCD x4 的 RefSRWKV native-grid 三分支模型。
# LR 先双三次上采样到 HR；真实 Ref 全程保持 HR 网格，避免早期折叠细节。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_common.sh"

shortcut_entry \
  scripts/train/refsrwkv.py \
  configs/runs/refsrwkv/hrms_scd_trefsr_hr_native_x4.yaml \
  "$@"
