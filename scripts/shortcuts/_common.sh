#!/usr/bin/env bash
set -euo pipefail

shortcut_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd
}

shortcut_usage() {
  local script_name="$1"
  cat <<EOF
用法:
  ${script_name}                 运行训练
  ${script_name} --run            运行训练
  ${script_name} --print          只打印命令，不启动训练
 ${script_name} --help           显示帮助

如果对应实验目录已有 config.yaml，将优先使用该完整配置；否则使用默认 run 配置。
训练入口之后的参数会原样传递，例如 --resume、--load-weights 或 --overrides。
EOF
}

shortcut_print() {
  local root="$1"
  local train_script="$2"
  local config="$3"
  shift 3
  local python_bin="${REFRWKV_PYTHON:-python}"
  printf 'cd %q\n' "$root"
  printf '%q %q --config %q' "$python_bin" "$train_script" "$config"
  if (($#)); then
    printf ' %q' "$@"
  fi
  printf '\n'
}

shortcut_run() {
  local root="$1"
  local train_script="$2"
  local config="$3"
  shift 3

  cd "$root"

  export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"

  # Activate the known remote environment when the caller has not activated one.
  if [[ -z "${CONDA_PREFIX:-}" ]]; then
    local conda_script="${REFRWKV_CONDA_SH:-}"
    if [[ -z "$conda_script" ]]; then
      for candidate in \
        "/mnt/sda/conda/miniforge3/etc/profile.d/conda.sh" \
        "/home/zhy/miniconda3/etc/profile.d/conda.sh"; do
        if [[ -f "$candidate" ]]; then
          conda_script="$candidate"
          break
        fi
      done
    fi
    if [[ -n "$conda_script" && -f "$conda_script" ]]; then
      # shellcheck disable=SC1090
      source "$conda_script"
      conda activate "${REFRWKV_CONDA_ENV:-rwkv7}"
    fi
  fi

  local python_bin="${REFRWKV_PYTHON:-python}"
  if ! command -v "$python_bin" >/dev/null 2>&1 && [[ ! -x "$python_bin" ]]; then
    printf '找不到 Python：%s。请先激活环境，或设置 REFRWKV_PYTHON。\n' "$python_bin" >&2
    exit 127
  fi
  exec "$python_bin" "$train_script" --config "$config" "$@"
}

shortcut_entry() {
  local train_script="$1"
  local config="$2"
  shift 2
  local root
  root="$(shortcut_root)"

  case "${1:-}" in
    --help|-h)
      shortcut_usage "$(basename "${BASH_SOURCE[1]}")"
      ;;
    --print)
      shift
      shortcut_print "$root" "$train_script" "$config" "$@"
      ;;
    --run)
      shift
      shortcut_run "$root" "$train_script" "$config" "$@"
      ;;
    *)
      shortcut_run "$root" "$train_script" "$config" "$@"
      ;;
  esac
}
