#!/usr/bin/env bash
set -euo pipefail

# Compatibility launcher for cluster submit commands.  The actual command
# definitions live in scripts/shortcuts/ so local and remote runs share one
# source of truth.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHORTCUT_DIR="$SCRIPT_DIR/shortcuts"
DEFAULT_SHORTCUT="${REFRWKV_SHORTCUT:-swinir_hrms_scd_sr_x4.sh}"

usage() {
  cat <<'EOF'
用法:
  submit_train.sh                              运行默认快捷脚本
  submit_train.sh <快捷脚本名>                  运行指定快捷脚本
  submit_train.sh --list                       列出可用快捷脚本
  submit_train.sh <快捷脚本名> --print          只打印训练命令

默认脚本：swinir_hrms_scd_sr_x4.sh。也可以用 REFRWKV_SHORTCUT 指定默认脚本。
其余参数（如 --resume、--load-weights、--overrides）会传给训练入口。
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" == "--list" ]]; then
  find "$SHORTCUT_DIR" -maxdepth 1 -type f -name '*.sh' \
    ! -name '_common.sh' ! -name 'show_commands.sh' \
    -printf '%f\n' | sort
  exit 0
fi

shortcut="$DEFAULT_SHORTCUT"
if [[ $# -gt 0 && "${1#-}" == "$1" ]]; then
  shortcut="$1"
  shift
fi

if [[ "$shortcut" != *.sh ]]; then
  shortcut="${shortcut}.sh"
fi
if [[ "$shortcut" == */* || "$shortcut" == .* ]]; then
  printf '快捷脚本必须是 shortcuts 目录中的文件名：%s\n' "$shortcut" >&2
  exit 2
fi

target="$SHORTCUT_DIR/$shortcut"
if [[ ! -f "$target" ]]; then
  printf '找不到快捷脚本：%s\n' "$target" >&2
  printf '可用脚本：\n' >&2
  find "$SHORTCUT_DIR" -maxdepth 1 -type f -name '*.sh' \
    ! -name '_common.sh' ! -name 'show_commands.sh' \
    -printf '  %f\n' | sort >&2
  exit 2
fi

exec bash "$target" "$@"
