# CometTracker V7

A plusTipTracker-shaped tracker for EB3 comets, driven by SAM3 masks. Written
from scratch — no code shared with V1–V6, though their *measurements* set most of
the defaults (each one is cited in `config.py`).

Read [`DESIGN.md`](DESIGN.md) for why it is built this way and what has and has
not been verified.

## Install and run

```bash
pip install -r requirements.txt      # numpy scipy scikit-image tifffile
```

```python
from comet_tracker_v7.pipeline import run_folder
from comet_tracker_v7.config import Config

r = run_folder("../ModelComparison/SAM3_Predictions",
               "20260710_pAJV103_0.25DOX-ON_010", Config())

print(len(r.segments), "segments ->", len(r.compounds), "compound tracks")
print(r.stats["growth_speed_um_min_mean"])
for t in r.tracks[:5]:
    print(t["compound_id"], t["motion_class"], t["growth_speed_um_min"])
```

It reads the same `<stem>_labels.tif` / `<stem>_prob.tif` layout that
`sam3_export.py` and the ilastik/U-Net arms all write, so all three detectors go
through it unchanged.

## What it does, in order

| stage | module | what happens |
|---|---|---|
| masks → table | `detections.py` | one row per instance: position, axis, axis uncertainty, shape, score |
| frame to frame | `link.py` | Kalman + LAP with birth/death, gated by the **mask-axis corridor** |
| gap closing | `gapclose.py` | global LAP over ends × starts, each join labelled **fgap** (pause) or **bgap** (catastrophe) |
| classify | `motion.py` | MSS: immobile / confined / Brownian / directed |
| numbers | `summarize.py` | growth speed, lifetime, gap frequencies — using plusTip's own definitions |

## The one idea

A SAM3 mask is a rendered centreline (`soft_uniform_axis`, "deliberately no
taper"), so its principal axis is the growth axis. Measured on real predictions:
**5.4°** median from the direction of travel, and the across-axis component of a
single-frame step has a **p99 of 2.16 px**. The corridor gate spends that, and
because it comes from shape rather than motion it gates a track's *first* link —
which is what u-track's three-pass scheme exists to work around.

## Things to know before trusting a number

- **No head is used anywhere.** SAM3's head branch gives 1.37 heads per comet
  (up to 8), so V7 tracks the mask centroid. That costs a known bias: mask length
  drifts a median 6.8 px over a track, so centroid displacement understates
  plus-end displacement by about half of that. Reported per track as
  `length_drift_px`, not corrected.
- **Presence does nothing.** Kept as scaffolding, honestly labelled — see
  `config.py`. AUC 0.608 for separating moving comets from static junk.
- **Gap frequencies are not yet biology.** Median segment length is 4 frames, so
  plusTip's mean-of-1/duration definition mostly measures linker fragmentation.
- **Nothing is scored against hand-labelled real tracks.** The synthetic
  benchmark's clutter density and miss rate are V6's estimates.

## Tests and benchmark

```bash
python -m pytest tests/ -q          # 47 tests
```

`tests/test_geometry.py::test_axis_unit_matches_regionprops` is the load-bearing
one: it pins the `regionprops.orientation` convention that everything else
depends on. During development the wrong unit vector made the mask axis look
perpendicular to travel (median 84.7°) and would have inverted the corridor gate
into a filter keeping only impossible links.

```python
from comet_tracker_v7.synthbench import run_benchmark
print(run_benchmark())   # 3 seeds, 120 comets x 30 frames, known provenance
```

## Videos

[A rendered example is hosted here](https://claude.ai/code/artifact/02bb8675-9613-4740-87d2-49a7cb4d425f): raw frames on the left, tracks on
the right, coloured by motion class. The mp4s are not committed — they are
~3 MB each and rebuild in about 20 seconds:

```bash
python tools/render_video.py \
  --nd2 ../Data/2EB3/<movie>.nd2 \
  --pred ../ModelComparison/SAM3_Predictions \
  --stem <movie stem> \
  --out videos/track.mp4
```

A track is drawn only while it is alive: from its first frame to its last,
showing the path travelled so far, then gone. Passing `--hold N` keeps it on
screen for N frames after it ends. Requires `nd2` and `ffmpeg`.
