#!/usr/bin/env bash
set -euo pipefail

# 用途：从头训练 PanCollection WV3 x4 的新 RefSRWKV spectral_detail 全色多光谱融合模型。
# 数据根目录由 configs/datasets/refsr/pancollection_wv3.yaml 指向远程 H5 文件。
# 使用独立 run 名称，避免误恢复其他实验 checkpoint。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_common.sh"

shortcut_entry \
  scripts/train/refsrwkv.py \
  configs/runs/refsrwkv/pancollection_wv3_spectral_detail_x4.yaml \
  "$@"
