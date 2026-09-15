#!/usr/bin/env bash
set -euo pipefail

# FusionMamba 官方推理 vs RDM-PAN 分块：参数量 / FLOPs / 单张耗时。
#
#   FusionMamba official  : test 整图；test_hard 官方 cut_size=256
#   FusionMamba tile16    : 两边都 LR tile=16 overlap=8（和 RDM-PAN 相同）
#   RDM-PAN               : LR tile=16 overlap=8
#
# 用法:
#   bash scripts/shortcuts/test/compare_pan_cost.sh
#   FUSION_MAMBA_CKPT=... RDM_PAN_CKPT=... bash scripts/shortcuts/test/compare_pan_cost.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

python_bin="${REFRWKV_PYTHON:-python}"
fm="${FUSION_MAMBA_CKPT:-experiments/train/refsr/fusion_mamba/pancollection_wv3/x4/fusion_mamba_pancollection_wv3_official/checkpoints/420.ckpt}"
rdm="${RDM_PAN_CKPT:-experiments/train/refsr/rdm_pan/pancollection_wv3/x4/pancollection_wv3_rdm_pan_x4/checkpoints/last.ckpt}"
out="${REFRWKV_OUTPUT:-experiments/test/refsr/compare_pan_cost.json}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '2,16p' "$0"
  exit 0
fi

exec "$python_bin" -u scripts/test/compare_pan_cost.py \
  --fusion-ckpt "$fm" \
  --rdm-ckpt "$rdm" \
  --output "$out" \
  "$@"
