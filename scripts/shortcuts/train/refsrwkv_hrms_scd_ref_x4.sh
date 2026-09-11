#!/usr/bin/env bash
set -euo pipefail

# 用途：从头训练 HRMS-SCD x4 的新 RefSRWKV spectral_detail 模型，读取配对真实 Ref/。
# 使用独立 run 名称，避免自动恢复旧 legacy checkpoint。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_common.sh"

shortcut_entry \
  scripts/train/refsrwkv.py \
  configs/runs/refsrwkv/hrms_scd_trefsr_spectral_detail_x4.yaml \
  "$@"
