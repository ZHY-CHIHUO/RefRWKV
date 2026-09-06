#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
用法:
  show_commands.sh                 查看 HRMS-SCD 三组训练命令
  show_commands.sh 参数...         将参数附加到每条训练命令

例如：
  show_commands.sh --overrides train.max_steps=1000
EOF
  exit 0
fi

shortcuts=(
  swinir_hrms_scd_sr_x4.sh
  refsrwkv_hrms_scd_sr_x4.sh
  refsrwkv_hrms_scd_ref_x4.sh
)

for shortcut in "${shortcuts[@]}"; do
  printf '# %s\n' "$shortcut"
  bash "$SCRIPT_DIR/$shortcut" --print "$@"
  printf '\n'
done
