#!/usr/bin/env bash
set -euo pipefail

# 只重跑 RDM-PAN 全分辨率预览（test_hard / OrigScale 512）。
# 之前 tile16_ov8/test_hard 误用了 reduced 256 图，文件名也是 test_wv3_multiExm1_*。
# 正确输出应是:
#   PanCollectionH5Dataset [test_hard]: ... lr=(8, 128, 128) ref=(1, 512, 512)
#   文件名 test_wv3_OrigScale_multiExm1_000000.png
#   尺寸 512x512
#
# 用法:
#   bash scripts/shortcuts/test/rdm_pan_preview_fr.sh
#   bash scripts/shortcuts/test/rdm_pan_preview_fr.sh /path/to.ckpt

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

python_bin="${REFRWKV_PYTHON:-python}"
ckpt="${1:-experiments/train/refsr/rdm_pan/pancollection_wv3/x4/pancollection_wv3_rdm_pan_x4/checkpoints/last.ckpt}"
out="${REFRWKV_OUTPUT:-experiments/test/refsr/rdm_pan_preview}"

if [[ ! -f "$ckpt" ]]; then
  printf '找不到 checkpoint：%s\n' "$ckpt" >&2
  exit 2
fi

# 清掉错误的 256 FR 预览，避免新旧文件混在一起。
rm -rf "$out/tile16_ov8/test_hard"

exec "$python_bin" -u scripts/test/rdm_pan_preview.py \
  --checkpoint "$ckpt" \
  --output "$out" \
  --splits test_hard \
  --overlaps 8 \
  --no-grad
