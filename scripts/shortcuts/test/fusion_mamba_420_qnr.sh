#!/usr/bin/env bash
set -euo pipefail

# 用途：用官方 FusionMamba 420.pth 复测 WV3 full-res QNR/Dλ/Ds。
# Ds 已改为 toolbox 1.0：MATLAB imresize 降采样 PAN，再 interp23tap。
# OrigScale 512 在 16GB 上无法整图，这里按官方 test.py 的 cut_size=256
# （LR tile=64, overlap=0）。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

python_bin="${REFRWKV_PYTHON:-python}"
ckpt="${REFRWKV_CHECKPOINT:-experiments/train/refsr/fusion_mamba/pancollection_wv3/x4/fusion_mamba_pancollection_wv3_official/checkpoints/420.ckpt}"
out="${REFRWKV_OUTPUT:-experiments/test/refsr/compare_wv3_fullimage/fusion_mamba_420_qnr_toolbox}"

if [[ "${1:-}" == "--unit" ]]; then
  exec "$python_bin" -m unittest tests.test_pansharpening -v
fi

if [[ ! -f "$ckpt" ]]; then
  printf '找不到 checkpoint：%s\n' "$ckpt" >&2
  exit 2
fi

exec "$python_bin" scripts/test/run.py \
  --config configs/test/test.yaml \
  --checkpoint "$ckpt" \
  --split test_hard \
  --metrics d_lambda d_s qnr \
  --no-save-images \
  --output "$out" \
  --overrides \
    data.eval_tile_size=64 \
    data.eval_tile_overlap=0 \
    dataset.files.test_hard=full_examples/test_wv3_OrigScale_multiExm1.h5 \
    data.files.test_hard=full_examples/test_wv3_OrigScale_multiExm1.h5
