#!/usr/bin/env bash
set -euo pipefail

# 用途：从头训练 PanCollection WV3 的 RefSRWKV native-grid 全色融合模型。
# 8 通道 MS LR + 1 通道 PAN Ref -> 8 通道 HR；H5 根目录由数据集 YAML 指定。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_common.sh"

shortcut_entry \
  scripts/train/refsrwkv.py \
  configs/runs/refsrwkv/pancollection_wv3_hr_native_x4.yaml \
  "$@"
