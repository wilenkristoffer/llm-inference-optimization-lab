#!/usr/bin/env bash
# Phase 11: the WSL training venv, ~/ft-env.
#
# A third venv beside ~/rocm-env and ~/vllm-env. transformers / peft / trl
# declare a plain "torch" dependency, and pip will happily satisfy it with a
# CUDA wheel from PyPI, silently replacing the ROCm build. So torch is
# installed first from the ROCm index, then pinned with a constraints file for
# every later install, and torch.version.hip is checked at the end.
#
# The two exports the other venvs need (memory: rocm-wsl-working-setup) are
# appended to the activate script: the rocprofiler stub and DXG detection.
#
# Run from WSL:   bash setup_ft_env.sh
# Safe to re-run; it skips the venv creation and the activate edits if done.

set -euo pipefail

ENV="$HOME/ft-env"
HERE="$(cd "$(dirname "$0")" && pwd)"
TORCH="torch==2.14.0+rocm7.2"
TRITON="triton-rocm==3.8.0"
ROCM_INDEX="https://download.pytorch.org/whl/rocm7.2"

if [ ! -d "$ENV" ]; then
    python3 -m venv "$ENV"
fi
if ! grep -q norocprof "$ENV/bin/activate"; then
    echo 'export LD_PRELOAD=$HOME/lab/norocprof.so' >> "$ENV/bin/activate"
    echo 'export HSA_ENABLE_DXG_DETECTION=1' >> "$ENV/bin/activate"
fi
# shellcheck disable=SC1091
source "$ENV/bin/activate"

pip install -q --upgrade pip
pip install -q "$TORCH" "$TRITON" --index-url "$ROCM_INDEX"

printf '%s\n%s\n' "$TORCH" "$TRITON" > "$ENV/constraints.txt"
pip install -q -c "$ENV/constraints.txt" -r "$HERE/requirements-ft.txt"

python - <<'EOF'
import torch, transformers, peft, trl, datasets, accelerate
assert torch.version.hip, "torch was replaced by a non-ROCm build"
print("torch", torch.__version__, "hip", torch.version.hip)
for m in (transformers, peft, trl, datasets, accelerate):
    print(m.__name__, m.__version__)
print("gpu", torch.cuda.is_available() and torch.cuda.get_device_name(0))
EOF
