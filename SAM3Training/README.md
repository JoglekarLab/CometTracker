# Final comet SAM3 training campaign

For the file-by-file package map, transfer requirements, readiness evidence,
and exact launch checklist, see `TRAINING_PACKAGE_GUIDE.md`.

This folder is ready for the last training round on the cluster. It fine-tunes
the official, pinned SAM3 image detector for four outputs per comet:

1. object presence/confidence;
2. a uniformly widened central-axis mask (3 source pixels, no taper and no
   comet-body mask);
3. a head heatmap and subpixel head coordinate; and
4. a learned identity-link score from frame `t` to `t+1`.

The literal text prompt is bypassed. One learned class token replaces
`microtubule plus-end comet`, so the model does not depend on whether that
phrase has useful linguistic meaning to SAM3.

## Frozen input contract

Every target is seen through a causal pseudo-RGB pair:

```text
X_t   = [R=I(t-2), G=I(t-1), B=I(t)]
X_t+1 = [R=I(t-1), G=I(t),   B=I(t+1)]
```

The overlap is intentional. The repeated underlying frames are numerically
identical in both inputs, while each input is centered on the frame whose
head/axis is supervised. Inputs use the approved conservative preprocessing:
movie-level temporal-median background, positive residual clipping, a 50/50
raw/residual blend, and one joint robust normalization. The source tile is
192×192; SAM3 then resizes it to its native 1008×1008 input. Coordinates and
the 3-pixel axis width remain defined in the 192×192 source coordinate system.

## Audited data

The manifests contain recipes and relative provenance paths, not duplicated
image arrays. Training-time geometry augmentation is restricted to D8
rotations/reflections and integer translation. There is no added Gaussian or
Poisson noise, intensity multiplier, blur, or bleaching augmentation.

- Current V4 labels: 173 accepted pairs total. Of these, 172 have heads
  (344 head points), 113 have axes (226 axes), and one is axis-only. The train
  split has 158 accepted pairs plus 310 background rectangles; validation has
  15 accepted pairs plus 65 background rectangles.
- Old U-Net masks: 742 safe consecutive axis/link pairs, expanded into 5,936
  certified D8 copy/paste recipes. Zero outside a painted mask stays unknown;
  no head label is invented.
- Procedural source: generated on the fly with exact heads, tail-to-head axes,
  and persistent identities. Within this source, 80% are moving positives,
  10% are empty negatives, and 10% contain frozen comet-shaped hard negatives.
  Each comet independently draws its exponential decay length from a four-part
  mixture: 5% at 3–5 px, 70% at 5–18 px, 20% at 18–24 px, and 5% at 24–28 px;
  sampling is uniform inside the selected interval.
  There are no paused positive tracks. Eight percent of positive scenes branch;
  half of those forks use a shallow 10–14 degree total opening and begin before
  the supervised pair so both daughters are resolvable and linkable. Five
  percent of positive scenes contain a two-parent-to-one-child merge. Association
  loss is masked only across an ambiguous split or merge transition.
  Every scene has weak analytic background drift; 45% also contain slowly
  appearing/disappearing or drifting blobs, and 35% contain a crowded field of
  thin, slowly wiggling microtubule-like curves. These structures are rendered
  directly and receive no comet labels. They do not use post-hoc pixel noise,
  blur, intensity scaling, or bleaching.
- Current-source sampling is 75% accepted pairs and 25% certified background
  regions, preventing the more numerous rectangles from swamping the positive
  annotations.

Background rectangles are **partial negatives**. Only queries centered inside
the hand-certified rectangle receive a negative presence target; the rest of
the crop remains unknown. Generic rejects and uncertain reviews are never
treated as background. `pAJV101` contributes background only (10.65% of the
training background pool) and is excluded from positives and validation.

There is no test split. The small validation set is frozen at the movie level:

- `20260710_pAJV103_0.25DOX-ON_010`
- `20260716_N271_0.25DOX_ON_002`

## Training schedule

Each row lasts five epochs. Counts are exact per epoch.

| Epochs | Pair samples/epoch | Procedural | Old U-Net paste | Current V4 |
|---|---:|---:|---:|---:|
| 1–5 | **6,000** | **5,400 (90%)** | 480 (8%) | 120 (2%) |
| 6–10 | 5,000 | 3,250 (65%) | 1,250 (25%) | 500 (10%) |
| 11–15 | 4,000 | 1,600 (40%) | 1,400 (35%) | 1,000 (25%) |
| 16–20 | 3,000 | 600 (20%) | 900 (30%) | 1,500 (50%) |
| 21–25 | 2,500 | 125 (5%) | 500 (20%) | **1,875 (75%)** |

The campaign contains 102,500 pair samples: 54,875 procedural, 22,650 old-mask
paste, and 24,975 current-label draws. Epochs 1–5 keep the ViT trunk frozen and
train the FPN neck, SAM encoder/decoder/axis branch, head branch, link branch,
and learned class token. Starting at epoch 6, only the upper four ViT blocks
and final ViT normalization are additionally unfrozen at `2e-6`.

Other optimizer settings:

- physical batch: 1 pair/GPU;
- gradient accumulation: 8 pairs;
- new-head LR: `1e-4`;
- SAM decoder/FPN LR: `2e-5`;
- upper-vision LR: `2e-6`;
- AdamW, 1,000-step warmup, cosine decay, gradient norm cap 1.0;
- no weight decay on biases, normalization parameters, or embeddings.

The best checkpoint is selected lexicographically: fewest false positives in
certified background, then lowest head p90 error, highest link accuracy, then
lowest axis centerline distance. Full validation runs after epochs 5, 10, 15,
20, and every epoch from 21–25; other epochs use a 32-pair quick validation.

## Cluster setup

From `SAM3Training/` on Great Lakes (or the equivalent cluster checkout):

```bash
bash scripts/setup_cluster_env.sh
conda activate /nfs/turbo/umms-ajitj/conda_envs/comet-sam3
hf auth login
SAM3_REPO=/nfs/turbo/umms-ajitj/software/sam3-660a5e9 \
  bash scripts/download_checkpoint.sh
python scripts/build_manifest.py --config configs/campaign.yaml
sbatch sbatch/preflight.sbatch
```

The official SAM3 checkpoint is gated on Hugging Face. `hf auth login` stores
credentials in the account cache; no token is written into these scripts.
Override `TRAIN_ENV`, `SAM3_REPO`, or `SAM3_CHECKPOINT` as environment variables
if the cluster paths differ.

Do not start the long job until `preflight.json` reports `status: passed` for
both the frozen and upper-ViT-unfrozen cases. The preflight performs the real
Hungarian/multitask loss, backward pass, AdamW step (including moment-memory
allocation), gradient checks for every output branch, and GPU allocated/reserved
memory checks.

Then submit:

```bash
sbatch sbatch/train.sbatch
```

The 72-hour job is resumable. It overwrites `runs/comet_sam3_final/last.pt`
atomically during and after epochs. If wall time or a signal stops it, submit
the same command again; `--resume auto` is the default and continues from the
saved epoch/pair index. To stop deliberately after one phase:

```bash
TRAIN_ARGS="--stop-after-epoch 5" sbatch sbatch/train.sbatch
```

Outputs are under `runs/comet_sam3_final/`:

- `preflight.json`: both memory/API/gradient audits;
- `data_audit.json`: real path, causal-channel, target, and leakage checks;
- `metrics.jsonl`: step and epoch metrics;
- `last.pt`: full resume state;
- `best.pt`: best compact adapter;
- `epoch_XX_adapter.pt`: compact epoch snapshots; and
- `source_inventory.json`: manifest and sampling inventory.

The checkpoint records and validates the exact SAM3 git commit, base-checkpoint
SHA256, configuration fingerprint, trained parameter keys, optimizer/scheduler,
and RNG state. A wrong or partially loaded official checkpoint is a hard error.

## Validation: axis agreement

![Hand-annotated comet axes overlaid on the predicted axis mask](docs/sam3_axis_validation.png)

Causal RGB input on the left, predicted axis mask at `p >= 0.5` on the right,
with hand-annotated axis pixels drawn on top — white inside the mask, orange
outside. The percentage is the share of annotated axis pixels the mask covers.
