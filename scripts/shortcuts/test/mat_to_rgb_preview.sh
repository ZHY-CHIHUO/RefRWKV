#!/usr/bin/env bash
set -euo pipefail

# 把 D:\Download\Results 里全部 .mat 转成 RGB 预览图，按 数据集/模型 分开放。
# 每个模型目录有 00.png... 和一张 _contact.png 拼图。
#
#   bash scripts/shortcuts/test/mat_to_rgb_preview.sh
#   bash scripts/shortcuts/test/mat_to_rgb_preview.sh --overwrite

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

python_bin="${REFRWKV_PYTHON:-python}"
src="${OFFICIAL_MAT_ROOT:-/mnt/d/Download/Results}"
out="${REFRWKV_OUTPUT:-experiments/vis/official_mat_previews}"

exec "$python_bin" -u scripts/test/mat_to_rgb_preview.py \
  --src "$src" \
  --output "$out" \
  "$@"
