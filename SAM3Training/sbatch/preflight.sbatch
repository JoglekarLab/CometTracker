#!/bin/bash
#SBATCH --job-name=comet-sam3-check
#SBATCH --output=preflight_%j.out
#SBATCH --error=preflight_%j.err
#SBATCH --partition=spgpu,gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mem=64GB
#SBATCH --account=ajitj99

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

TRAIN_ENV="${TRAIN_ENV:-/nfs/turbo/umms-ajitj/conda_envs/comet-sam3}"
export SAM3_REPO="${SAM3_REPO:-/nfs/turbo/umms-ajitj/software/sam3-660a5e9}"
export SAM3_CHECKPOINT="${SAM3_CHECKPOINT:-/nfs/turbo/umms-ajitj/checkpoints/sam3/sam3.pt}"

eval "$(conda shell.bash hook)"
conda activate "$TRAIN_ENV"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$SLURM_SUBMIT_DIR:$SAM3_REPO:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda build", torch.version.cuda)
print("available", torch.cuda.is_available(), "count", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; refusing a CPU fallback")
print("gpu", torch.cuda.get_device_name(0))
PY

python scripts/build_manifest.py --config configs/campaign.yaml --audit-only
python scripts/audit_training_data.py \
  --config configs/campaign.yaml \
  --output runs/comet_sam3_final/data_audit.json
python -m comet_sam3.preflight --config configs/campaign.yaml
