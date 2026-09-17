#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CONFIG="configs/runs/rdm_refsr/pancollection_wv3_pan_v5_x4.yaml"

echo "=== Starting RDM-PAN v5 Training on WV3 (x4) ==="
echo "Config: ${CONFIG}"

python scripts/train/run.py --config "${CONFIG}" "$@"
