#!/usr/bin/env bash
# Build the pinned SAM3 environment once on a cluster login node.
set -euo pipefail

TRAIN_ENV="${TRAIN_ENV:-/nfs/turbo/umms-ajitj/conda_envs/comet-sam3}"
SAM3_REPO="${SAM3_REPO:-/nfs/turbo/umms-ajitj/software/sam3-660a5e9}"
TRAINING_ROOT="${TRAINING_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
TORCH_VERSION="${TORCH_VERSION:-2.10.0}"
CUDA_WHEEL="${CUDA_WHEEL:-cu128}"
SAM3_COMMIT="660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7"

eval "$(conda shell.bash hook)"
if [[ ! -x "$TRAIN_ENV/bin/python" ]]; then
  conda create -p "$TRAIN_ENV" -c conda-forge -y python=3.12 pip git libstdcxx-ng
fi
conda activate "$TRAIN_ENV"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

TRAIN_PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$TRAIN_PYTHON_VERSION" != "3.12" ]]; then
  echo "Existing training environment uses Python $TRAIN_PYTHON_VERSION; SAM3Training requires Python 3.12." >&2
  echo "Choose a new empty TRAIN_ENV path, or recreate this environment with Python 3.12." >&2
  exit 1
fi

python -m pip install --upgrade pip wheel "setuptools==80.9.0"
python -m pip install \
  "torch==$TORCH_VERSION" torchvision \
  --index-url "https://download.pytorch.org/whl/$CUDA_WHEEL"

if [[ ! -d "$SAM3_REPO/.git" ]]; then
  git clone https://github.com/facebookresearch/sam3.git "$SAM3_REPO"
fi
git -C "$SAM3_REPO" fetch origin "$SAM3_COMMIT"
git -C "$SAM3_REPO" checkout --detach "$SAM3_COMMIT"

python -m pip install -e "$SAM3_REPO[train]"
python -m pip install einops pycocotools psutil
python -m pip install -e "$TRAINING_ROOT[test]" nd2

python - <<'PY'
import torch
print("torch", torch.__version__, "CUDA build", torch.version.cuda)
print("SAM3 environment installed; CUDA availability is checked in a GPU job.")
PY

cat <<EOF

Environment: $TRAIN_ENV
SAM3 repo:   $SAM3_REPO

Next:
  conda activate "$TRAIN_ENV"
  hf auth login
  SAM3_REPO="$SAM3_REPO" bash scripts/download_checkpoint.sh
  sbatch sbatch/preflight.sbatch
EOF
