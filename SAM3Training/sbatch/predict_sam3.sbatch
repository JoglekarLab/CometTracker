#!/bin/bash
# Six-movie SAM3 prediction run, matching the ilastik/U-Net comparison set.
#
#   ADAPTER=runs/comet_sam3_final/best.pt sbatch sbatch/predict_sam3.sbatch
#
# Override any of ADAPTER, DATA, OUT, ONLY, EXTRA on the command line.
# EXTRA goes through verbatim, so a first smoke run is:
#
#   ADAPTER=runs/comet_sam3_final/best.pt \
#   ONLY=20260710_EB3WT_0.25DOX-ON_003 \
#   EXTRA="--limit-frames 8 --progress 1" \
#     sbatch sbatch/predict_sam3.sbatch
#
# Then copy $OUT down to the laptop and open it beside the other two models
# with ModelComparison/compare3/view3.py.
#SBATCH --job-name=comet-sam3-predict
#SBATCH --output=predict_%j.out
#SBATCH --error=predict_%j.err
# spgpu ONLY, deliberately. The gpu partition holds 16GB pre-Ampere V100s:
# a third of the memory training had, and no flash attention, so SAM3 falls
# back to a math kernel that materializes the full attention matrix. A pair of
# 1008x1008 images then wants ~6.4GB in one allocation and the job dies in the
# first forward pass. Override on the command line if you know what you are
# taking:  sbatch --partition=gpu ...  with BATCH_TILES=1.
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=96GB
#SBATCH --account=ajitj99

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

TRAIN_ENV="${TRAIN_ENV:-/nfs/turbo/umms-ajitj/conda_envs/comet-sam3}"
export SAM3_REPO="${SAM3_REPO:-/nfs/turbo/umms-ajitj/software/sam3-660a5e9}"
export SAM3_CHECKPOINT="${SAM3_CHECKPOINT:-/nfs/turbo/umms-ajitj/checkpoints/sam3/sam3.pt}"

ADAPTER="${ADAPTER:-runs/comet_sam3_final/best.pt}"
DATA="${DATA:-../Data}"
OUT="${OUT:-../ModelComparison/SAM3_Predictions}"
# The same six movies the ilastik and U-Net arms were run on. A model that
# predicted a different set is not a comparison.
# Tiles per forward pass. Each tile is TWO 1008x1008 images (t and t+1), so
# this is 2*BATCH_TILES images through the ViT at once. 2 is comfortable on a
# 48GB A40; drop to 1 on anything smaller.
BATCH_TILES="${BATCH_TILES:-2}"
# Cuts allocator fragmentation on long runs, as the OOM message itself suggests.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

ONLY="${ONLY:-20260710_EB3WT_0.25DOX-ON_001,20260710_EB3WT_0.25DOX-ON_002,20260710_EB3WT_0.25DOX-ON_003,20260710_pAJV103_0.25DOX-ON_010,20260710_pAJV103_0.25DOX-ON_015,20260715_pAJV101_0.25DOX_ON_017}"

eval "$(conda shell.bash hook)"
conda activate "$TRAIN_ENV"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$SLURM_SUBMIT_DIR:$SAM3_REPO:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

if [[ ! -f "$ADAPTER" ]]; then
  echo "no adapter checkpoint at $ADAPTER" >&2
  echo "set ADAPTER=... ; runs/comet_sam3_final/ holds best.pt and last.pt" >&2
  exit 1
fi

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable; the pinned SAM3 wrapper refuses CPU")
name = torch.cuda.get_device_name(0)
gib = torch.cuda.get_device_properties(0).total_memory / 2**30
major = torch.cuda.get_device_capability(0)[0]
print(f"{name}  {gib:.1f} GiB  sm_{major}x")
# Say it here, before an hour of queueing turns into an OOM traceback.
if gib < 30:
    print(f"WARNING: {gib:.0f} GiB is small for 1008x1008 SAM3 inference. "
          f"Use BATCH_TILES=1 if this job OOMs.", flush=True)
if major < 8:
    print("WARNING: pre-Ampere GPU - no flash attention, so attention is "
          "materialized in full and memory use is much higher than training.",
          flush=True)
PY

read -r -a EXTRA_ARGV <<< "${EXTRA:-}"
python scripts/sam3_export.py \
  --config configs/campaign.yaml \
  --adapter "$ADAPTER" \
  --data "$DATA" \
  --out "$OUT" \
  --only "$ONLY" \
  --batch-tiles "$BATCH_TILES" \
  "${EXTRA_ARGV[@]}"

echo
echo "wrote $OUT"
ls -la "$OUT"
