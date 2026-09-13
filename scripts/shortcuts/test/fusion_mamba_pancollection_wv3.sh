#!/usr/bin/env bash
set -euo pipefail

WEIGHT="${1:-/tmp/FusionMamba2/weights/420.pth}"
shift || true
exec conda run --no-capture-output -n rwkv7 python scripts/test/fusion_mamba_wv3.py --weight "$WEIGHT" "$@"
