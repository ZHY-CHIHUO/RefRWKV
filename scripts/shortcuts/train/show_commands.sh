#!/usr/bin/env bash
set -euo pipefail

# 用途：只打印所有训练快捷脚本的命令，不启动训练。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
用法:
  show_commands.sh                 查看 HRMS-SCD 可训练对比命令
  show_commands.sh 参数...         将参数附加到每条训练命令

例如：
  show_commands.sh --overrides train.max_steps=1000
EOF
  exit 0
fi

shortcuts=(
  swinir_hrms_scd_sr_x4.sh
  edsr_hrms_scd_sr_x4.sh
  rcan_hrms_scd_sr_x4.sh
  hat_hrms_scd_sr_x4.sh
  mambairv2_hrms_scd_sr_x4.sh
  refsrwkv_hrms_scd_sr_x4.sh
  refsrwkv_hrms_scd_ref_x4.sh
  ttsr_hrms_scd_ref_x4.sh
  masa_sr_hrms_scd_ref_x4.sh
  datsr_hrms_scd_ref_x4.sh
  refsrwkv_wuhan.sh
)

for shortcut in "${shortcuts[@]}"; do
  printf '# %s\n' "$shortcut"
  bash "$SCRIPT_DIR/$shortcut" --print "$@"
  printf '\n'
done
