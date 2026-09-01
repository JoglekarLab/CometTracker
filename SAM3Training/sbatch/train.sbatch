#!/bin/bash
#SBATCH --job-name=comet-sam3-train
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err
#SBATCH --partition=spgpu,gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --mem=96GB
#SBATCH --account=ajitj99
#SBATCH --signal=USR1@300

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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable; refusing CPU fallback")
print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 2**30, "GiB")
PY

if [[ ! -s training_data/manifests/train.jsonl || ! -s training_data/manifests/val.jsonl ]]; then
  python scripts/build_manifest.py --config configs/campaign.yaml
fi
python scripts/build_manifest.py --config configs/campaign.yaml --audit-only

# Refuse to start the expensive campaign unless the exact current configuration
# passed both real SAM3 optimizer/memory regimes on a GPU.
PREFLIGHT_REPORT="${PREFLIGHT_REPORT:-runs/comet_sam3_final/preflight.json}"
python - "$PREFLIGHT_REPORT" <<'PY'
import json
import sys
from pathlib import Path

from comet_sam3.checkpointing import config_fingerprint
from comet_sam3.config import load_config

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"missing GPU preflight report: {path}")
report = json.loads(path.read_text())
if report.get("status") != "passed":
    raise SystemExit(f"GPU preflight did not pass: {path}")
expected = config_fingerprint(load_config("configs/campaign.yaml"))
if report.get("config_fingerprint") != expected:
    raise SystemExit("GPU preflight was run with a different campaign configuration")
regimes = {case.get("regime") for case in report.get("cases", [])}
if regimes != {"frozen", "unfrozen"}:
    raise SystemExit(f"GPU preflight is missing a freeze regime: {sorted(regimes)}")
print(f"validated GPU preflight: {path}")
PY

read -r -a TRAIN_ARGV <<< "${TRAIN_ARGS:-}"
python -m comet_sam3.train --config configs/campaign.yaml "${TRAIN_ARGV[@]}"
