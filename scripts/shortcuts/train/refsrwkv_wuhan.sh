#!/usr/bin/env bash
set -euo pipefail

# 用途：训练 Wuhan 四通道时空参考超分模型（网络倍率 x1）。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_common.sh"

shortcut_entry \
  scripts/train/refsrwkv.py \
  configs/runs/refsrwkv/wuhan.yaml \
  "$@"
