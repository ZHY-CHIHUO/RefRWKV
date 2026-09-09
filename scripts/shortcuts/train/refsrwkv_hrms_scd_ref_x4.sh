#!/usr/bin/env bash
set -euo pipefail

# 用途：训练 HRMS-SCD x4 的 RefSRWKV TRefSR 模型，读取配对真实 Ref/。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_common.sh"

shortcut_entry \
  scripts/train/refsrwkv.py \
  configs/runs/refsrwkv/hrms_scd_trefsr_x4.yaml \
  "$@"
