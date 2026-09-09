#!/usr/bin/env bash
set -euo pipefail

# 用途：使用独立 test.yaml 对训练 checkpoint 或无参数 Bicubic 基线执行测试。
# checkpoint/--training-config 后面的参数原样传给 scripts/test/run.py。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TEST_CONFIG="${REFRWKV_TEST_CONFIG:-configs/test/test.yaml}"
CHECKPOINT="${REFRWKV_CHECKPOINT:-}"
TRAINING_CONFIG="${REFRWKV_TRAINING_CONFIG:-}"

usage() {
  cat <<'EOF'
用法:
  run_test.sh <checkpoint> [测试参数...]
  run_test.sh --training-config <training.yaml> [测试参数...]

示例:
  bash scripts/shortcuts/test/run_test.sh \
    experiments/train/sr/swinir/hrms_scd/x4/hrms_scd_swinir_x4/checkpoints/last.ckpt
  bash scripts/shortcuts/test/run_test.sh <checkpoint> --split test_easy --no-save-images

环境变量:
  REFRWKV_TEST_CONFIG  测试 YAML，默认 configs/test/test.yaml
  REFRWKV_CHECKPOINT   可替代第一个位置参数提供 checkpoint
  REFRWKV_TRAINING_CONFIG  可替代 --training-config 提供 Bicubic 配置
  REFRWKV_PYTHON       Python 可执行文件，默认 python
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" == "--training-config" ]]; then
  if [[ -z "${2:-}" || "${2#-}" != "$2" ]]; then
    printf '%s\n' '--training-config 后必须提供一个训练 YAML 路径。' >&2
    usage >&2
    exit 2
  fi
  TRAINING_CONFIG="$2"
  shift 2
fi
if [[ -z "$CHECKPOINT" && $# -gt 0 && "${1#-}" == "$1" ]]; then
  CHECKPOINT="$1"
  shift
fi
if [[ -z "$CHECKPOINT" && -z "$TRAINING_CONFIG" ]]; then
  printf '缺少 checkpoint。请提供 checkpoint，或使用 --training-config 运行 Bicubic。\n' >&2
  usage >&2
  exit 2
fi

cd "$PROJECT_ROOT"
python_bin="${REFRWKV_PYTHON:-python}"
if ! command -v "$python_bin" >/dev/null 2>&1 && [[ ! -x "$python_bin" ]]; then
  printf '找不到 Python：%s。请先激活环境，或设置 REFRWKV_PYTHON。\n' "$python_bin" >&2
  exit 127
fi

command=("$python_bin" scripts/test/run.py --config "$TEST_CONFIG")
if [[ -n "$CHECKPOINT" ]]; then
  command+=(--checkpoint "$CHECKPOINT")
fi
if [[ -n "$TRAINING_CONFIG" ]]; then
  command+=(--training-config "$TRAINING_CONFIG")
fi
exec "${command[@]}" "$@"
