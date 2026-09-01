"""Run a trained comet-SAM3 checkpoint over movies and write a prediction
folder the ilastik/U-Net comparison tools already know how to read.

    # on the cluster, from SAM3Training/
    python scripts/sam3_export.py \
        --config configs/campaign.yaml \
        --adapter runs/comet_sam3_final/best.pt \
        --data ../Data --out ../ModelComparison/SAM3_Predictions \
        --only 20260710_EB3WT_0.25DOX-ON_001,...

    # then copy --out down to the laptop and open it next to the other two:
    #   ModelComparison/compare3/view3.py

WHY IT LOOKS LIKE THIS

  * SAM3 is a query detector. It emits 200 object queries per frame, each with
    a presence score, an axis mask and a head heatmap - not the dense
    probability map ilastik and the U-Net produce. Every downstream tool
    (comparemodels.py, trackcompare.py, harvest.py) re-thresholds a
    `_prob.tif`, so the queries are collapsed into one:

        P(pixel) = max over accepted queries of  presence(q) * P_axis(q, pixel)

    A pixel still means "confidence this pixel is comet", which is the only
    property those tools rely on. `comet_sam3/export.py` holds that arithmetic
    and is tested without a GPU.

  * The 192-pixel tile is not a tuning knob. Training saw 192x192 source tiles
    resized to SAM3's 1008x1008 input - a 5.25x magnification. Feeding a whole
    512x512 frame instead is a 1.97x magnification, so every comet arrives
    2.7x smaller than anything the model was trained on. `--tile` therefore
    defaults to the configured `input.source_tile_size` and refuses to differ
    from it without `--allow-scale-change`, which exists only so the failure
    can be demonstrated rather than guessed at.

  * Each tile is normalized on its own, because that is what training did:
    build_current_pair_sample slices the movie AND the temporal-median
    background to the 192x192 tile and only then calls causal_rgb_pair, so the
    1st/99.7th percentiles setting the intensity map are the tile's. Doing it
    once on the whole frame would be cheaper and wrong.

  * The input contract is causal: X_t = [I(t-2), I(t-1), I(t)]. Frames 0, 1
    and the last frame cannot be a center. They stay zero in `_prob.tif` and
    the count is printed. Nothing is interpolated in from a neighbour - a
    frame the model never saw must not look like one it did.

  * Heads and links are written to their own files. The other two models have
    no notion of a plus-end or an identity link, so putting them in
    `_points.csv` would silently change a file three other scripts parse.

WHAT WAS TESTED, AND WHAT WAS NOT
    `tests/test_export.py` covers all of the decode arithmetic against known
    arrays, and asserts the shared `points_and_labels` is element-identical to
    `ilastik_export.points_and_labels`, so the three models are reduced to
    detections by the same rule.

    The torch half below has NOT been run: this was written on a machine with
    no CUDA, no SAM3 checkout and no trained checkpoint. Every call it makes
    into the model mirrors `comet_sam3/metrics.py`, which validation ran every
    epoch - the presence sigmoid, the bilinear axis interpolation, the head
    softargmax and `pairwise_link_logits` are the same operations in the same
    order. Run `--limit-frames 4` on one movie first and look at the printed
    detections/frame before launching the full six.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from comet_sam3 import export as ex
from comet_sam3.config import load_config, required_environment
from comet_sam3.preprocessing import causal_rgb_pair, temporal_median_background


def find_movies(data_dir: str, only: list[str], ext: str = ".nd2") -> list[str]:
    """Same selection rule as ilastik_export.find_movies."""
    import glob

    paths = sorted(glob.glob(os.path.join(data_dir, "**", "*" + ext), recursive=True))
    only = [o for o in only if o]
    if only:
        paths = [p for p in paths if any(o in os.path.basename(p) for o in only)]
    return paths


def load_movie(path: str) -> np.ndarray:
    """(T, Y, X). Reuses the package's own reader so the axis handling -
    which channel is taken, which z is projected - matches what training saw
    rather than being a second opinion about the same file."""
    from comet_sam3.data.current_annotations import _load_movie

    return _load_movie(Path(path))


def build_model(config: dict, adapter: str, device: str):
    from comet_sam3.checkpointing import load_checkpoint
    from comet_sam3.model import CometSAM3

    _, base_checkpoint = required_environment(config)
    model_cfg = config["model"]
    model = CometSAM3.from_checkpoint(
        str(base_checkpoint),
        sam_input_size=int(config["input"]["sam_input_size"]),
        track_embedding_dim=int(model_cfg["track_embedding_dim"]),
        disable_dac_query_duplication=bool(model_cfg["disable_dac_query_duplication"]),
        device=device,
    )
    # restore_rng=False: this is inference, and restoring the training RNG
    # would silently reseed the process.
    payload = load_checkpoint(adapter, model, config=config, restore_rng=False)
    model.eval()
    return model, payload


def _tile_batch(images: np.ndarray, torch, device):
    """(N, H, W, 3) float32 -> (N, 3, H, W) tensor on device."""
    array = np.ascontiguousarray(np.moveaxis(images, -1, 1))
    return torch.from_numpy(array).to(device=device, dtype=torch.float32)


def _frame_arrays(frame_predictions, batch_index, keep, tile, torch, F):
    """presence, axis probability at tile resolution, and head maps, for the
    queries in ``keep``. Mirrors metrics.MetricAccumulator.update."""
    presence = frame_predictions.presence_logits[batch_index].sigmoid()
    if keep is None:
        keep = torch.arange(presence.shape[0], device=presence.device)
    if keep.numel() == 0:
        empty = np.zeros((0, tile, tile), np.float32)
        return np.zeros((0,), np.float32), empty, empty
    axis = F.interpolate(
        frame_predictions.axis_logits[batch_index, keep, None].float(),
        size=(tile, tile), mode="bilinear", align_corners=False,
    )[:, 0].sigmoid()
    head = F.interpolate(
        frame_predictions.head_logits[batch_index, keep, None].float(),
        size=(tile, tile), mode="bilinear", align_corners=False,
    )[:, 0]
    return (
        presence[keep].float().cpu().numpy(),
        axis.cpu().numpy().astype(np.float32),
        head.cpu().numpy().astype(np.float32),
    )


def export_movie(model, movie: np.ndarray, a, config: dict) -> dict:
    """One movie -> dense probability, head rows, link rows."""
    import contextlib

    import torch
    from torch.nn import functional as F

    tile = int(a.tile)
    n_frames, height, width = movie.shape
    if tile > height or tile > width:
        # Same refusal, same wording, as current_annotations._crop_origin.
        # Padding the shortfall would feed the model a band of black it never
        # saw in training and call the result a prediction.
        raise SystemExit(
            f"tile {tile} does not fit movie shape {(height, width)}")
    # Median over the whole movie, exactly as _load_background does, then
    # sliced per tile below - NOT recomputed per tile.
    background = temporal_median_background(movie)
    blend = float(config["input"]["background_blend"])

    origins = [
        (y0, x0)
        for y0 in ex.tile_origins(height, tile, a.stride)
        for x0 in ex.tile_origins(width, tile, a.stride)
    ]
    centers = list(ex.predictable_centers(n_frames))
    prob = np.zeros((n_frames, height, width), np.float32)
    head_rows: list[dict] = []
    link_rows: list[dict] = []

    def amp():
        """A FRESH autocast per forward. Reusing one instance across a long
        loop is asking a context manager to be re-entrant for 1400 frames."""
        if a.device.startswith("cuda") and config["training"]["precision"] == "bf16":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def inputs_for(center, y0, x0):
        """The causal pseudo-RGB pair for ONE tile.

        Cropped before normalizing, not after. build_current_pair_sample slices
        the movie and the background to the tile and only then calls
        causal_rgb_pair, so the 1st/99.7th percentiles that set the intensity
        map are the tile's own. Normalizing the whole 512x512 frame instead
        would let a bright structure anywhere else in the field dim every tile,
        and the model would be seeing intensities it was never trained on.
        """
        return causal_rgb_pair(
            movie[:, y0:y0 + tile, x0:x0 + tile],
            center=center,
            background=background[y0:y0 + tile, x0:x0 + tile],
            background_blend=blend,
        )

    started = time.time()
    for step, center in enumerate(centers):
        frame_heads: list[dict] = []
        frame_head_yx: list[tuple[float, float]] = []
        frame_head_score: list[float] = []
        raw_links: list[tuple[int, float, float, float]] = []

        for start in range(0, len(origins), a.batch_tiles):
            chunk = origins[start:start + a.batch_tiles]
            pairs = [inputs_for(center, y0, x0) for y0, x0 in chunk]
            tiles_t = np.stack([pair[0] for pair in pairs])
            tiles_p = np.stack([pair[1] for pair in pairs])
            with torch.no_grad(), amp():
                predictions = model(
                    _tile_batch(tiles_t, torch, a.device),
                    _tile_batch(tiles_p, torch, a.device),
                )

            for b, (y0, x0) in enumerate(chunk):
                score_t = predictions.t.presence_logits[b].sigmoid()
                keep_t = torch.nonzero(score_t >= a.presence, as_tuple=False).flatten()
                presence, axis, head_maps = _frame_arrays(
                    predictions.t, b, keep_t, tile, torch, F
                )
                ex.paste_max(prob[center], ex.dense_axis_map(presence, axis), y0, x0)

                local_index: dict[int, int] = {}
                for k in range(presence.shape[0]):
                    hy, hx = ex.softargmax_yx(head_maps[k])
                    y, x = hy + y0, hx + x0
                    if not (0 <= y < height and 0 <= x < width):
                        continue
                    local_index[k] = len(frame_heads)
                    frame_heads.append(dict(
                        frame=center,
                        y=round(float(y), 3), x=round(float(x), 3),
                        presence=round(float(presence[k]), 4),
                        axis_peak=round(float(axis[k].max()), 4),
                        query=int(keep_t[k]),
                        tile_y=int(y0), tile_x=int(x0),
                    ))
                    frame_head_yx.append((y, x))
                    frame_head_score.append(float(presence[k]))

                if a.link_thresh > 1.0 or not local_index:
                    continue
                score_p = predictions.tp1.presence_logits[b].sigmoid()
                keep_p = torch.nonzero(score_p >= a.presence, as_tuple=False).flatten()
                if keep_p.numel() == 0:
                    continue
                _, _, head_maps_p = _frame_arrays(
                    predictions.tp1, b, keep_p, tile, torch, F
                )
                with torch.no_grad():
                    scores = model.pairwise_link_logits(
                        predictions.t.track_embeddings[b, keep_t][None].float(),
                        predictions.tp1.track_embeddings[b, keep_p][None].float(),
                    )[0].sigmoid().cpu().numpy()
                for i, j, score in ex.match_links(scores, a.link_thresh):
                    if i not in local_index:
                        continue
                    ny, nx = ex.softargmax_yx(head_maps_p[j])
                    raw_links.append((local_index[i], ny + y0, nx + x0, score))

        keep_heads = ex.dedupe_points(
            np.asarray(frame_head_yx, np.float64).reshape(-1, 2),
            np.asarray(frame_head_score, np.float64),
            a.min_distance,
        )
        survived = set(int(index) for index in keep_heads)
        head_rows.extend(frame_heads[int(index)] for index in keep_heads)
        for raw_index, ny, nx, score in raw_links:
            if raw_index not in survived:
                continue
            row = frame_heads[raw_index]
            link_rows.append(dict(
                frame=center,
                y0=row["y"], x0=row["x"],
                y1=round(float(ny), 3), x1=round(float(nx), 3),
                score=round(float(score), 4),
            ))

        if a.progress and (step % a.progress == 0 or step == len(centers) - 1):
            elapsed = time.time() - started
            print(f"    frame {center} ({step + 1}/{len(centers)}), "
                  f"{len(head_rows)} heads, {elapsed:.0f}s", flush=True)

    return dict(prob=prob, heads=head_rows, links=link_rows,
                centers=centers, n_frames=n_frames)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="configs/campaign.yaml")
    p.add_argument("--adapter", required=True,
                   help="a comet-sam3-adapter-v2 checkpoint, e.g. "
                        "runs/comet_sam3_final/best.pt")
    p.add_argument("--data", required=True, help="folder tree with the movies")
    p.add_argument("--out", required=True, help="prediction folder to write")
    p.add_argument("--only", default="", help="comma-separated stem substrings")
    p.add_argument("--ext", default=".nd2")
    p.add_argument("--device", default="cuda")

    p.add_argument("--tile", type=int, default=0,
                   help="source tile size. 0 = the configured "
                        "input.source_tile_size, which is the only value the "
                        "model was trained at.")
    p.add_argument("--allow-scale-change", action="store_true",
                   help="permit --tile to differ from the trained tile size. "
                        "Every comet then arrives at the wrong scale; this "
                        "exists to demonstrate that, not to tune it.")
    p.add_argument("--stride", type=int, default=128,
                   help="tile stride. Smaller = more overlap, more compute, "
                        "and more duplicate heads for --min-distance to remove.")
    p.add_argument("--batch-tiles", type=int, default=2,
                   help="tiles per forward pass. Each tile costs TWO 1008x1008 "
                        "images (t and t+1). Raise until the GPU is full.")

    p.add_argument("--presence", type=float, default=0.20,
                   help="queries below this presence score contribute nothing "
                        "to _prob.tif and produce no head. Deliberately low: "
                        "the real threshold is applied downstream, and a map "
                        "that cannot represent a weak detection cannot be "
                        "re-thresholded fairly against the other two models.")
    p.add_argument("--thresh", type=float, default=0.5,
                   help="threshold for _labels.tif / _points.csv only")
    p.add_argument("--min-area", type=int, default=6)
    p.add_argument("--min-distance", type=float, default=3.0,
                   help="suppress heads closer than this. Without it the head "
                        "count is a function of --stride.")
    p.add_argument("--link-thresh", type=float, default=0.5,
                   help="minimum link score to write. >1 disables link export.")

    p.add_argument("--limit-frames", type=int, default=0,
                   help="only the first N frames. Use it for the first run.")
    p.add_argument("--progress", type=int, default=10,
                   help="print every N centers; 0 is silent")
    a = p.parse_args(argv)

    config = load_config(a.config)
    trained_tile = int(config["input"]["source_tile_size"])
    if a.tile <= 0:
        a.tile = trained_tile
    if a.tile != trained_tile and not a.allow_scale_change:
        raise SystemExit(
            f"--tile {a.tile} but the model was trained on {trained_tile}-pixel "
            f"tiles resized to {config['input']['sam_input_size']}. A different "
            f"tile changes the magnification every comet is seen at. Pass "
            f"--allow-scale-change if that is really what you want.")
    if a.stride > a.tile:
        raise SystemExit(f"--stride {a.stride} > --tile {a.tile} leaves gaps")

    paths = find_movies(a.data, [o.strip() for o in a.only.split(",")], a.ext)
    if not paths:
        raise SystemExit(f"no {a.ext} movies under {a.data!r} matching {a.only!r}")
    print(f"{len(paths)} movies -> {a.out}", flush=True)

    model, payload = build_model(config, a.adapter, a.device)
    print(f"adapter epoch {payload.get('epoch')}, "
          f"step {payload.get('global_step')}, "
          f"tile {a.tile} stride {a.stride}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        movie = load_movie(path)
        if a.limit_frames:
            movie = movie[:a.limit_frames]
        print(f"  {stem}  {movie.shape}", flush=True)
        result = export_movie(model, movie, a, config)

        rows, labels = ex.points_and_labels(result["prob"], a.thresh, a.min_area)
        ex.write_outputs(a.out, stem, result["prob"], rows, labels)
        ex.write_heads(a.out, stem, result["heads"])
        ex.write_links(a.out, stem, result["links"])

        predicted = len(result["centers"])
        skipped = result["n_frames"] - predicted
        rate = len(rows) / predicted if predicted else 0.0
        print(f"  {stem}: {len(rows)} detections ({rate:.1f}/frame), "
              f"{len(result['heads'])} heads, {len(result['links'])} links, "
              f"{skipped} frames not predictable (causal input needs t-2..t+1)",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
