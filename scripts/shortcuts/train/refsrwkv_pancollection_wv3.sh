#!/usr/bin/env bash
set -euo pipefail

# 用途：训练 PanCollection WV3 x4 的 RefSRWKV 全色多光谱融合模型。
# 数据根目录由 configs/datasets/refsr/pancollection_wv3.yaml 指向远程 H5 文件。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_common.sh"

shortcut_entry \
  scripts/train/refsrwkv.py \
  configs/runs/refsrwkv/pancollection_wv3.yaml \
  "$@"
