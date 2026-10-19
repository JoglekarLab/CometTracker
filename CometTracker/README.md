# CometTracker v9

The tracker that [`CometPhotometry/`](../CometPhotometry/) runs on. It links SAM3
masks into growth tracks with the same method as v7. For how and why that
method works, see [`CometTracker_v7/DESIGN.md`](../CometTracker_v7/DESIGN.md).

## What changed since v7

**v8** ([`CHANGELOG_v8.md`](CHANGELOG_v8.md))
- Pixel size is set **per movie** from `zoom_table.csv`: 1× = 0.1589 µm/px,
  1.5× = 0.1059 µm/px. The .nd2 files never record the 1.5× optovar.
- The 488 nm channel is read **by name** (`movie.read_movie`), so two-channel
  files are handled correctly.
- The motion label uses the `diffusion` criterion. It compares how far a track gets from its start with how far random jiggling with the same step size would get: D = net distance ÷ (rms step × √steps). A track is directed when D ≥ 1.73, which random motion reaches only 5% of the time.

**v9** ([`CHANGELOG_v9.md`](CHANGELOG_v9.md))
- Per-comet background from a box turned with the comet, 5–9 px to each side.
- An EMG fit along each comet gives its plus end (μ) and brightest point (max).
- Two more speeds per track from those positions: EMG μ speed and EMG max speed.

Linking and gap closing are the same as v8. Defaults: `max_gap = 0`, tracks of
at least 3 frames.

## Install

```bash
pip install -r requirements.txt      # numpy scipy scikit-image tifffile
pip install nd2                      # to read the .nd2 movies
```

## Use

```python
from comet_tracker_v9.pipeline import run_folder
from comet_tracker_v9.config import Config

r = run_folder("path/to/SAM3_Predictions", "<movie stem>", Config())
print(len(r.compounds), "tracks")
```

It reads the `<stem>_labels.tif` / `<stem>_prob.tif` folder that SAM3 writes.

The per-movie summary (speeds, lengths, photometry) can also be run from the
command line. It writes one JSON per movie:

```bash
python -m comet_tracker_v9.postprocess --pred PRED_DIR --nd2-root Data \
    --zoom-table zoom_table.csv --photometry --no-heads --out OUT_DIR
```

- `--nd2-root` expects `Data/<design>/<movie>.nd2` or
  `Data/<design>/<YYYYMMDD>/<movie>.nd2`, and takes the design from the folder.
  For one flat folder of movies use `--nd2 FOLDER` instead.
- If a movie's .nd2 is not found, the tracks are still written but without
  photometry, so check the `group` and `channel_used` fields in the output.
- Head speed needs `SAM3Training/` on the Python path; `--no-heads` skips it.

## Files

| | |
|---|---|
| `comet_tracker_v9/` | the package |
| `tests/` | `python -m pytest tests -q` |
| `zoom_table.csv` | 1× or 1.5× for every movie. Movies not listed are taken as 1.5× |
| `CHANGELOG_v8.md`, `CHANGELOG_v9.md` | what changed, and what it was checked on |
