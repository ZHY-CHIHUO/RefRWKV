#!/usr/bin/env bash
set -euo pipefail

# WV3 全指标对比：FusionMamba 与 RDM-PAN
# Reduced : PSNR / Q2n / SAM / ERGAS
# Full-res: Dλ / Ds / QNR（Ds 已对齐 toolbox 1.0 MATLAB imresize）
#
# 切块：
#   fusion_mamba reduced  PAN 256 整图（tile=1024）
#   fusion_mamba full-res 官方 test.py cut_size=256（LR tile=64）；512 整图 16GB OOM
#   rdm_pan      两边都走训练裁块 PAN 64（LR tile=16, overlap=8），CUDA WKV T=256 数值不可用
#
# 用法：
#   bash scripts/shortcuts/test/compare_wv3_pan.sh
#   bash scripts/shortcuts/test/compare_wv3_pan.sh fusion_mamba
#   bash scripts/shortcuts/test/compare_wv3_pan.sh rdm_pan

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

python_bin="${REFRWKV_PYTHON:-python}"
out_root="${REFRWKV_OUTPUT:-experiments/test/refsr/compare_wv3_pan}"
want="${1:-all}"

fm_ckpt="${FUSION_MAMBA_CKPT:-experiments/train/refsr/fusion_mamba/pancollection_wv3/x4/fusion_mamba_pancollection_wv3_official/checkpoints/last.ckpt}"
rdm_ckpt="${RDM_PAN_CKPT:-experiments/train/refsr/rdm_pan/pancollection_wv3/x4/pancollection_wv3_rdm_pan_x4/checkpoints/last.ckpt}"
orig_h5="full_examples/test_wv3_OrigScale_multiExm1.h5"

run_one() {
  local name="$1" ckpt="$2" split="$3" metrics="$4"
  shift 4
  if [[ ! -f "$ckpt" ]]; then
    printf '找不到 checkpoint：%s\n' "$ckpt" >&2
    exit 2
  fi
  printf '\n==== %s  %s ====\n' "$name" "$split"
  "$python_bin" scripts/test/run.py \
    --config configs/test/test.yaml \
    --checkpoint "$ckpt" \
    --split "$split" \
    --metrics $metrics \
    --no-save-images \
    --output "$out_root/$name" \
    --overrides "$@"
}

run_fusion_mamba() {
  run_one fusion_mamba "$fm_ckpt" test "psnr q2n sam ergas" \
    data.eval_tile_size=1024 data.eval_tile_overlap=0
  run_one fusion_mamba "$fm_ckpt" test_hard "d_lambda d_s qnr" \
    data.eval_tile_size=64 data.eval_tile_overlap=0 \
    "dataset.files.test_hard=$orig_h5" "data.files.test_hard=$orig_h5"
}

run_rdm_pan() {
  run_one rdm_pan "$rdm_ckpt" test "psnr q2n sam ergas" \
    data.eval_tile_size=16 data.eval_tile_overlap=8
  run_one rdm_pan "$rdm_ckpt" test_hard "d_lambda d_s qnr" \
    data.eval_tile_size=16 data.eval_tile_overlap=8 \
    "dataset.files.test_hard=$orig_h5" "data.files.test_hard=$orig_h5"
}

case "$want" in
  all|"")
    run_fusion_mamba
    run_rdm_pan
    ;;
  fusion_mamba|fm)
    run_fusion_mamba
    ;;
  rdm_pan|rdm)
    run_rdm_pan
    ;;
  -h|--help)
    sed -n '2,18p' "$0"
    exit 0
    ;;
  *)
    printf '未知目标 %s，可选 all / fusion_mamba / rdm_pan\n' "$want" >&2
    exit 2
    ;;
esac

printf '\n完成。结果在 %s/{fusion_mamba,rdm_pan}/{test,test_hard}/metrics.json\n' "$out_root"
