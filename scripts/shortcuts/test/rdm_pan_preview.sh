#!/usr/bin/env bash
set -euo pipefail

# RDM-PAN 全量推理预览：test + test_hard 各 20 张，干净 RGB，无红线。
# 默认 last.ckpt，LR tile=16；overlap=8（正式评测）和 overlap=0（更容易露缝）。
#
# 用法:
#   bash scripts/shortcuts/test/rdm_pan_preview.sh
#   bash scripts/shortcuts/test/rdm_pan_preview.sh /path/to.ckpt
#   bash scripts/shortcuts/test/rdm_pan_preview.sh --overlaps 8 --no-grad
#
# 输出 experiments/test/refsr/rdm_pan_preview/
#   tile16_ov8/test/*.png                       RR 256，文件名 test_wv3_multiExm1_*
#   tile16_ov8/test/*_gt.png                    reduced-res GT
#   tile16_ov8/test_hard/*.png                  FR 512，文件名 test_wv3_OrigScale_multiExm1_*
#
# 注意：checkpoint 里曾把 test_hard 指到 reduced 256。Python 预览会强制
# OrigScale H5；第一张若不是 lr=128 / pan=512 会直接报错。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

python_bin="${REFRWKV_PYTHON:-python}"
ckpt="${RDM_PAN_CKPT:-experiments/train/refsr/rdm_pan/pancollection_wv3/x4/pancollection_wv3_rdm_pan_x4/checkpoints/last.ckpt}"
out="${REFRWKV_OUTPUT:-experiments/test/refsr/rdm_pan_preview}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '2,22p' "$0"
  exit 0
fi
if [[ $# -gt 0 && "${1#-}" == "$1" ]]; then
  ckpt="$1"
  shift
fi
if [[ ! -f "$ckpt" ]]; then
  printf '找不到 checkpoint：%s\n' "$ckpt" >&2
  exit 2
fi
if ! command -v "$python_bin" >/dev/null 2>&1 && [[ ! -x "$python_bin" ]]; then
  printf '找不到 Python：%s。请先激活 rwkv7，或设置 REFRWKV_PYTHON。\n' "$python_bin" >&2
  exit 127
fi

# 清掉误用 reduced 256 的旧 FR 预览，避免和 OrigScale 512 混在一起。
rm -rf "$out"/tile16_ov8/test_hard "$out"/tile16_ov0/test_hard

exec "$python_bin" -u scripts/test/rdm_pan_preview.py \
  --checkpoint "$ckpt" \
  --output "$out" \
  "$@"
