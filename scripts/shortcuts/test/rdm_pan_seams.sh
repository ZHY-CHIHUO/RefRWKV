#!/usr/bin/env bash
set -euo pipefail

# 用 rdm_pan 权重做分块推理，保存 RGB / 网格 / 接缝图，方便肉眼看有没有接缝。
# 默认 last.ckpt，LR tile=16；各跑 overlap=8（当前评测）和 overlap=0（最容易露缝）。
#
# 用法:
#   bash scripts/shortcuts/test/rdm_pan_seams.sh
#   bash scripts/shortcuts/test/rdm_pan_seams.sh --max-samples 20
#   bash scripts/shortcuts/test/rdm_pan_seams.sh /path/to/xxx.ckpt --max-samples 4
#
# 输出:
#   experiments/test/refsr/rdm_pan_seams/ov8/{test,test_hard}/*_rgb.png
#   experiments/test/refsr/rdm_pan_seams/ov0/{test,test_hard}/*_rgb.png
#   *_grid.png   红线是分块步长
#   *_seam.png   梯度图，接缝会在红线上发亮
#   seam_report.json  格线梯度 / 内部梯度，ratio 明显 >1 才像接缝

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

python_bin="${REFRWKV_PYTHON:-python}"
ckpt="${RDM_PAN_CKPT:-experiments/train/refsr/rdm_pan/pancollection_wv3/x4/pancollection_wv3_rdm_pan_x4/checkpoints/last.ckpt}"
out="${REFRWKV_OUTPUT:-experiments/test/refsr/rdm_pan_seams}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '2,20p' "$0"
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
  printf '找不到 Python：%s。请先激活环境，或设置 REFRWKV_PYTHON。\n' "$python_bin" >&2
  exit 127
fi

exec "$python_bin" -u scripts/test/rdm_pan_seam_preview.py \
  --checkpoint "$ckpt" \
  --output "$out" \
  "$@"
