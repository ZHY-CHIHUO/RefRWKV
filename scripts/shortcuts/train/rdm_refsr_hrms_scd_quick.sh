#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_common.sh"

ROOT="$(shortcut_root)"
TRAIN_SCRIPT="scripts/train/rdm_refsr.py"
CONFIG="configs/runs/rdm_refsr/hrms_scd_x4.yaml"
OVERRIDES=(
  data.max_samples_train=256
  data.max_samples_val=32
  data.num_workers=0
  data.val_num_workers=0
  data.batch_size=1
  data.val_batch_size=1
  train.max_steps=300
  train.max_epochs=1
  train.accelerator=gpu
  train.devices=1
  train.precision=32
  train.accumulate_grad_batches=1
  train.enable_progress_bar=true
  train.num_sanity_val_steps=0
)

case "${1:-}" in
  --help|-h)
    cat <<'EOF'
用法:
  rdm_refsr_hrms_scd_quick.sh [--run|--print] [训练参数]

默认使用完整 RDMRefSR 网络，仅限制 HRMS-SCD 样本数和 300 steps。
传入的参数放在固定 quick overrides 前，例如：
  ..._quick.sh --run --resume experiments/.../last.ckpt
EOF
    ;;
  --print)
    shift
    shortcut_print "$ROOT" "$TRAIN_SCRIPT" "$CONFIG" "$@" --overrides "${OVERRIDES[@]}"
    ;;
  --run)
    shift
    shortcut_run "$ROOT" "$TRAIN_SCRIPT" "$CONFIG" "$@" --overrides "${OVERRIDES[@]}"
    ;;
  *)
    shortcut_run "$ROOT" "$TRAIN_SCRIPT" "$CONFIG" "$@" --overrides "${OVERRIDES[@]}"
    ;;
esac
