#!/usr/bin/env bash
set -euo pipefail

# 用途：本机检查原版双向 Bi-WKV CUDA 能否编译、前反向是否有限。
# 用法：
#   bash scripts/shortcuts/test/bi_wkv_cuda.sh
#   bash scripts/shortcuts/test/bi_wkv_cuda.sh --unittest

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  conda_script="${REFRWKV_CONDA_SH:-}"
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

cd "$PROJECT_ROOT"
python_bin="${REFRWKV_PYTHON:-python}"
if ! command -v "$python_bin" >/dev/null 2>&1 && [[ ! -x "$python_bin" ]]; then
  printf '找不到 Python：%s。请先激活 rwkv7，或设置 REFRWKV_PYTHON。\n' "$python_bin" >&2
  exit 127
fi

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'HELP'
用法:
  bash scripts/shortcuts/test/bi_wkv_cuda.sh
  bash scripts/shortcuts/test/bi_wkv_cuda.sh --unittest

检查 GPU 上的原版 Vision-RWKV 双向分段 Bi-WKV：
  1. 编译/加载 CUDA 扩展
  2. T=16/32/64 前向是否有限，并打印与 Python 参考的误差
  3. T=64 反向梯度是否有限

加 --unittest 会再跑 tests/test_rdm_refsr.py 里的两个 CUDA 用例。
HELP
  exit 0
fi

"$python_bin" - << 'PY'
import sys
import torch
from kernels.wkv.runtime import RUN_CUDA
from models.refsr.rdm_refsr.rdm_refsr import _reference_biwkv

if not torch.cuda.is_available():
    print("CUDA 不可用：torch.cuda.is_available() is False")
    sys.exit(1)

print(f"GPU: {torch.cuda.get_device_name(0)}  capability={torch.cuda.get_device_capability(0)}  torch.cuda={torch.version.cuda}")
print("编译/加载 Bi-WKV ...")
decay = (torch.rand(16, device="cuda") * 0.2 + 0.05)
first = torch.randn(16, device="cuda") * 0.1
key = torch.randn(1, 32, 16, device="cuda")
value = torch.randn(1, 32, 16, device="cuda")
out = RUN_CUDA(decay, first, key, value)
print(f"compile/run ok  shape={tuple(out.shape)}  finite={bool(torch.isfinite(out).all())}")

torch.manual_seed(7)
print("\nT    max|cuda-ref|   mean|cuda-ref|   finite  grad_ok")
for length in (16, 32, 33, 64, 65, 96):
    key = torch.randn(2, length, 16)
    value = torch.randn(2, length, 16)
    decay = torch.rand(16) * 0.2 + 0.05
    first = torch.randn(16) * 0.1
    ref = _reference_biwkv(decay, first, key, value)
    k = key.cuda().requires_grad_(True)
    v = value.cuda().requires_grad_(True)
    w = decay.cuda().requires_grad_(True)
    u = first.cuda().requires_grad_(True)
    cuda = RUN_CUDA(w, u, k, v)
    cuda.square().mean().backward()
    diff = (cuda.detach().cpu() - ref).abs()
    grads = [k.grad, v.grad, w.grad, u.grad]
    grad_ok = all(g is not None and torch.isfinite(g).all() for g in grads)
    print(
        f"{length:<4} {diff.max().item():14.6g} {diff.mean().item():14.6g}   "
        f"{str(bool(torch.isfinite(cuda).all())):<7} {grad_ok}"
    )

print("\n说明：原版 32 路双向核和 exclusive Python 公式不是逐点同一实现；看 finite/grad_ok。")
PY

if [[ "${1:-}" == "--unittest" ]]; then
  echo
  echo "==== unittest ===="
  "$python_bin" -m unittest tests.test_rdm_refsr.RDMRefSRTests.test_cuda_biwkv_is_finite -v
fi
