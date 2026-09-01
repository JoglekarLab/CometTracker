# SAM3 comet training package guide

Updated: 2026-08-30

## Readiness verdict

The code, configuration, labels, manifests, and local data are ready to transfer
to the cluster and run the GPU preflight.

The 72-hour training job should **not** be submitted until the cluster preflight
has passed. The official SAM3 checkout and checkpoint are deliberately external
to this package, and the real SAM3 API, BF16 execution, gradients, optimizer
memory, and GPU capacity cannot be verified on this Mac.

The training submission script now enforces this gate: it refuses to start
unless it finds a passing preflight report for the exact current configuration
and both the frozen and upper-ViT-unfrozen regimes.

## Where the files are

The local package is:

```text
/Users/linamp/Desktop/Claude-CometTrack/SAM3Training/
```

The package cannot be copied by itself because the manifests contain relative
recipes rather than image arrays. Preserve this minimum layout on the cluster:

```text
Claude-CometTrack/
├── SAM3Training/
├── Data/
├── TrajectoryAxisLabeling/
│   └── v4_test_session/
└── HeadLabeling/
    └── session_001/
        └── queue.csv
```

The referenced image and mask files are under `Data/`. There are 99 unique
ND2/TIFF files referenced by the manifests, totaling approximately 3.60 GiB.
The complete local `Data/` folder is approximately 3.9 GiB.

## What is inside `SAM3Training/`

### Top-level files

| Path | Purpose |
|---|---|
| `README.md` | Authoritative description of the input contract, data, 25-epoch schedule, optimizer, validation, cluster setup, and outputs. |
| `TRAINING_PACKAGE_GUIDE.md` | This operational map and launch checklist. |
| `configs/campaign.yaml` | Single source of truth for paths, preprocessing, synthetic generation, targets, losses, schedule, validation, and freezing. |
| `pyproject.toml` | Python 3.12 package metadata and scientific dependencies. |
| `preview_training_data.py` | Optional local preview tool; it is not used by the cluster training loop. |

### Model and training code: `comet_sam3/`

| File or folder | Purpose |
|---|---|
| `model.py` | Wraps the pinned official SAM3 image model. Adds the learned single-class token, head branch, tracking embeddings, and link scorer. |
| `schema.py` | Defines and validates comet instances, pair samples, heads, axes, IDs, links, exhaustive labels, and partial negatives. |
| `preprocessing.py` | Constructs the conservative causal RGB pair from temporal-median background subtraction and the raw/residual blend. |
| `geometry.py` | Applies the same lossless D8 rotation/reflection to both images and all coordinates. |
| `targets.py` | Builds the 3-source-pixel uniform axis target and Gaussian head heatmap. |
| `losses.py` | Hungarian matching plus partial-label-aware presence, box, axis, head, and link losses. |
| `metrics.py` | Certified-background false positives, head error, link accuracy, and axis centerline distance. |
| `optim.py` | AdamW groups, three learning-rate tiers, no-decay rules, warmup, and cosine scheduling. |
| `curriculum.py` | Deterministic epoch sizes and exact source proportions. |
| `checkpointing.py` | Atomic resume checkpoints and compact adapters. Validates the SAM3 commit, base-checkpoint SHA256, trained keys, and all semantic configuration including procedural settings. |
| `preflight.py` | Runs real forward, matching, multitask loss, backward, optimizer, gradient, BF16, and GPU-memory checks in both freeze regimes. |
| `train.py` | Resumable single-GPU BF16 training, validation, metrics, best-model selection, and signal-safe checkpointing. |
| `data/current_annotations.py` | Loads current V4 accepted pairs and explicitly drawn partial-background rectangles. |
| `data/unet_masks.py` | Derives axes and consecutive links from old U-Net masks and pastes certified comet crops into real backgrounds without inventing heads. |
| `data/synthetic.py` | Generates deterministic seven-frame movies with exact heads, axes, IDs, links, branches, merges, decay-length mixture, slow blobs, and wiggling filament backgrounds. |
| `data/sources.py` | Reads manifests and samples current, U-Net-paste, and procedural sources according to the curriculum. |

### Preparation and launch helpers

| Path | Purpose |
|---|---|
| `scripts/setup_cluster_env.sh` | Creates the Python 3.12 environment, installs Torch 2.10/cu128, checks out the pinned SAM3 commit, and installs SAM3 plus this package. |
| `scripts/download_checkpoint.sh` | Downloads the gated `facebook/sam3` checkpoint after Hugging Face login. |
| `scripts/build_manifest.py` | Rebuilds portable train/validation recipes and the leakage audit atomically. |
| `scripts/audit_training_data.py` | Checks every referenced file and materializes representative examples from every data source. |
| `sbatch/preflight.sbatch` | Thirty-minute, one-GPU fail-fast job with 4 CPUs and 64 GB system RAM. |
| `sbatch/train.sbatch` | Resumable 72-hour, one-GPU training job with 8 CPUs and 96 GB system RAM. It requires a matching passed preflight report. |

### Manifests and audits

| Path | Contents |
|---|---|
| `training_data/manifests/train.jsonl` | 6,404 training recipes. |
| `training_data/manifests/val.jsonl` | 144 frozen validation recipes. |
| `training_data/manifests/leakage_audit.json` | Movie-level train/validation leakage and pAJV101 policy audit. |
| `training_data/data_audit_local.json` | Latest local path and representative-materialization audit. |
| `tests/` | Configuration, preprocessing, data-source, synthetic-event, loss, target, optimizer, checkpoint, and launch-gate tests. |

## Current data inventory

Training manifest, 6,404 recipes:

- 158 current accepted pairs;
- 310 current certified-background regions;
- 5,936 U-Net-mask paste recipes, produced from 742 safe base pairs and all
  eight D8 transforms.

Validation manifest, 144 frozen recipes:

- 15 current accepted pairs;
- 65 current certified-background regions;
- 64 fixed procedural recipes.

Across train and validation there are 173 accepted current pairs and 375
certified-background regions. Procedural **training** scenes are generated on
the fly and therefore do not occupy thousands of manifest rows.

There is intentionally no test split. The two frozen validation movies are:

- `20260710_pAJV103_0.25DOX-ON_010`
- `20260716_N271_0.25DOX_ON_002`

## Local verification completed

The following checks passed on 2026-08-30:

- 42 automated tests;
- compilation of all Python modules;
- shell syntax for both setup scripts and both Slurm scripts;
- all 24,292 manifest file references resolve locally;
- 6,548 manifest IDs are unique;
- representative current-positive, current-background, U-Net-paste, and
  procedural samples materialize correctly as 192×192 causal RGB pairs;
- no train/validation movie overlap;
- both held-out movies appear only in validation;
- pAJV101 contributes background only: 33 of 310 training-background records,
  or 10.65%, below the 15% cap;
- checkpoint fingerprints change when procedural settings change;
- training cannot start with a missing, stale, or one-regime-only preflight.

A small non-blocking data-quality note: five of the 375 manually drawn
background rectangles are extremely thin, so they may occasionally produce no
negative query. Their effect is negligible, but they can be filtered in a
future cleanup if another campaign is run.

## Exact cluster sequence

Run these commands from the cluster copy of
`Claude-CometTrack/SAM3Training/`.

### 1. Create the pinned environment and SAM3 checkout

```bash
bash scripts/setup_cluster_env.sh
```

Default locations:

```text
Training environment: /nfs/turbo/umms-ajitj/conda_envs/comet-sam3
SAM3 checkout:       /nfs/turbo/umms-ajitj/software/sam3-660a5e9
Pinned commit:       660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7
```

The login node must be able to install packages and clone the official
repository. Override `TRAIN_ENV` or `SAM3_REPO` if different paths are
required.

### 2. Authenticate and download the gated checkpoint

```bash
conda activate /nfs/turbo/umms-ajitj/conda_envs/comet-sam3
hf auth login
SAM3_REPO=/nfs/turbo/umms-ajitj/software/sam3-660a5e9 \
  bash scripts/download_checkpoint.sh
```

Default checkpoint:

```text
/nfs/turbo/umms-ajitj/checkpoints/sam3/sam3.pt
```

### 3. Rebuild and audit the transferred manifests

```bash
python scripts/build_manifest.py --config configs/campaign.yaml
python scripts/build_manifest.py --config configs/campaign.yaml --audit-only
```

### 4. Run the mandatory GPU preflight

```bash
sbatch sbatch/preflight.sbatch
```

After the job finishes, inspect its Slurm output/error files and:

```bash
cat runs/comet_sam3_final/preflight.json
```

Do not continue unless the top-level status is `passed` and the report
contains both `frozen` and `unfrozen` cases. The GPU must support native BF16,
and the optimizer-step memory test must remain below the safety limit.

### 5. Start training

```bash
sbatch sbatch/train.sbatch
```

The script independently verifies the matching preflight again before loading
SAM3. It uses account `ajitj99`, partitions `spgpu,gpu`, one GPU, 8 CPUs,
96 GB RAM, and a 72-hour wall time.

To inspect only the first five-epoch phase:

```bash
TRAIN_ARGS="--stop-after-epoch 5" sbatch sbatch/train.sbatch
```

To resume after wall time or interruption, submit the same training command
again. The default `--resume auto` loads `last.pt`.

## Expected outputs

Training writes under `SAM3Training/runs/comet_sam3_final/`:

- `preflight.json`: two-regime GPU/API/gradient/memory report;
- `data_audit.json`: cluster-side path and source materialization audit;
- `source_inventory.json`: data-source counts and sampling settings;
- `metrics.jsonl`: training and validation metrics;
- `last.pt`: full atomic resume state;
- `best.pt`: best compact adapter;
- `epoch_XX_adapter.pt`: compact per-epoch snapshots.

The `runs/` directory does not exist locally yet because no cluster preflight
or training job has been run.

