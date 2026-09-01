#!/usr/bin/env python3
"""Build an interactive audit of the three SAM3 training-data sources.

The preview uses the same causal contract planned for training:

    X_t   = [R=I(t-2), G=I(t-1), B=I(t)]
    X_t+1 = [R=I(t-1), G=I(t),   B=I(t+1)]

It deliberately performs no post-hoc intensity, blur, Gaussian-noise, or
Poisson-noise augmentation.  Real examples receive only lossless D8 spatial
transforms.  Old mask tracks are pasted additively, at unit intensity, onto
certified real-background clips.
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation
from skimage.draw import line
from skimage.morphology import skeletonize


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from head_labeler import crop_context, load_movie, read_queue  # noqa: E402
from trajectory_axis_labeler_v4 import (  # noqa: E402
    blend_rgb,
    causal_rgb_endpoints,
    temporal_median_background,
)
from comet_tracker_v3.comet_tracker_v3.procgen import _draw_comet  # noqa: E402
from comet_tracker_v3.comet_tracker_v3.synthesize import _soft_alpha  # noqa: E402


DATA = PROJECT / "Data"
QUEUE = PROJECT / "HeadLabeling/session_001/queue.csv"
CURRENT_LABELS = PROJECT / "TrajectoryAxisLabeling/v4_test_session/labels.json"
SIZE = 96
CONTEXT = 5
BLEND = 0.5


@dataclass(frozen=True)
class Donor:
    movie: Path
    background_mask: Path


DONORS = (
    Donor(
        DATA / "EB3-GW16/20260716_GW16_0.25DOX_ON_001.nd2",
        DATA / "EB3-GW16/20260716_GW16_0.25DOX_ON_001_background.tif",
    ),
    Donor(
        DATA / "EB3-GW16/20260716_GW16_0.25DOX_ON_002.nd2",
        DATA / "EB3-GW16/20260716_GW16_0.25DOX_ON_002_background.tif",
    ),
)


OLD_SPECS = (
    ("2EB3", "20260710_pAJV103_0.25DOX-ON_001", 39, 19, 1, True),
    ("EB3-GW16", "20260716_GW16_0.25DOX_ON_008", 1, 53, 2, False),
    ("EB3-N271", "20260716_N271_0.25DOX_ON_003", 2, 16, 3, True),
    ("EB3WT", "20260713_EB3WT_0.25DOX_001", 1, 4, 1, False),
)


def load_tiff(path: Path) -> np.ndarray:
    import tifffile

    return np.asarray(tifffile.imread(path))


def normalize_rgb_pair(
    clip: np.ndarray,
    background: np.ndarray,
    center: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Use the V4 labeler's exact joint-stretch and conservative blend."""
    frames = list(range(center - CONTEXT, center + CONTEXT + 1))
    raw, residual = causal_rgb_endpoints(
        clip, background, frames, 0, 0, clip.shape[1], clip.shape[2]
    )
    rgb = blend_rgb(raw, residual, BLEND)
    return rgb[CONTEXT], rgb[CONTEXT + 1]


def best_certified_crop(mask: np.ndarray, t0: int, n: int, rank: int = 0) -> tuple[int, int]:
    """Choose a deterministic 96px crop with maximal certified-background coverage."""
    coverage = np.asarray(mask[t0:t0 + n], np.float32).mean(axis=0)
    H, W = coverage.shape
    integral = np.pad(coverage.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    scored = []
    for y in range(0, H - SIZE + 1, 8):
        for x in range(0, W - SIZE + 1, 8):
            total = (
                integral[y + SIZE, x + SIZE]
                - integral[y, x + SIZE]
                - integral[y + SIZE, x]
                + integral[y, x]
            )
            scored.append((float(total), y, x))
    scored.sort(reverse=True)
    # Spread examples among several near-perfect areas rather than returning
    # exactly the same crop for every seed.
    return scored[min(rank * 7, len(scored) - 1)][1:]


def donor_clip(index: int, length: int = 15) -> tuple[np.ndarray, np.ndarray, dict]:
    donor = DONORS[index % len(DONORS)]
    movie = load_movie(donor.movie)
    mask = load_tiff(donor.background_mask)
    t0 = 8 + (index * 13) % (len(movie) - length - 8)
    y0, x0 = best_certified_crop(mask, t0, length, rank=index % 5)
    clip = movie[t0:t0 + length, y0:y0 + SIZE, x0:x0 + SIZE].astype(np.float32)
    background = temporal_median_background(movie)[y0:y0 + SIZE, x0:x0 + SIZE]
    coverage = float(mask[t0:t0 + length, y0:y0 + SIZE, x0:x0 + SIZE].mean())
    meta = {
        "movie": donor.movie.stem,
        "frames": f"{t0}–{t0 + length - 1}",
        "crop": f"y={y0}:{y0 + SIZE}, x={x0}:{x0 + SIZE}",
        "certified": f"{100.0 * coverage:.1f}%",
    }
    return clip, background, meta


def d8(array: np.ndarray, k: int, flip: bool, spatial_axes: tuple[int, int] = (-2, -1)) -> np.ndarray:
    out = np.rot90(array, int(k), axes=spatial_axes)
    if flip:
        out = np.flip(out, axis=spatial_axes[1])
    return np.ascontiguousarray(out)


def d8_yx(y: float, x: float, n: int, k: int, flip: bool) -> tuple[float, float]:
    if k % 4 == 0:
        yy, xx = y, x
    elif k % 4 == 1:
        yy, xx = n - 1 - x, y
    elif k % 4 == 2:
        yy, xx = n - 1 - y, n - 1 - x
    else:
        yy, xx = x, n - 1 - y
    if flip:
        xx = n - 1 - xx
    return float(yy), float(xx)


def exact_axis(shape: tuple[int, int], head: tuple[float, float], tail: tuple[float, float]) -> np.ndarray:
    out = np.zeros(shape, bool)
    yy, xx = line(
        int(round(head[0])), int(round(head[1])),
        int(round(tail[0])), int(round(tail[1])),
    )
    keep = (yy >= 0) & (yy < shape[0]) & (xx >= 0) & (xx < shape[1])
    out[yy[keep], xx[keep]] = True
    return out


def clean_axis(mask: np.ndarray) -> np.ndarray | None:
    skel = skeletonize(np.asarray(mask, bool))
    if not skel.any():
        return None
    # The audited corpus is 99.2% a single unbranched skeleton. Reject rather
    # than silently repairing the remaining cases.
    padded = np.pad(skel.astype(np.uint8), 1)
    neighbors = np.zeros_like(skel, np.uint8)
    for dy in range(3):
        for dx in range(3):
            if dy == 1 and dx == 1:
                continue
            neighbors += padded[dy:dy + skel.shape[0], dx:dx + skel.shape[1]]
    endpoints = skel & (neighbors == 1)
    if int(endpoints.sum()) != 2 or np.any(skel & (neighbors > 2)):
        return None
    return skel


def encode_png(rgb: np.ndarray, scale: int = 2) -> str:
    arr = np.clip(np.asarray(rgb) * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(arr, "RGB")
    image = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
    buf = io.BytesIO()
    image.save(buf, format="WEBP", quality=72, method=6)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def overlay_image(
    rgb: np.ndarray,
    axes: np.ndarray | None = None,
    heads: list[tuple[float, float]] | None = None,
    scale: int = 2,
) -> str:
    arr = np.clip(np.asarray(rgb) * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(arr, "RGB").resize(
        (arr.shape[1] * scale, arr.shape[0] * scale), Image.Resampling.NEAREST
    )
    draw = ImageDraw.Draw(image)
    if axes is not None and np.any(axes):
        shown = binary_dilation(np.asarray(axes, bool), iterations=1)
        yy, xx = np.nonzero(shown)
        for y, x in zip(yy, xx):
            draw.rectangle(
                (x * scale, y * scale, x * scale + scale - 1, y * scale + scale - 1),
                fill=(255, 45, 210),
            )
    for y, x in heads or []:
        cy, cx = y * scale, x * scale
        r = 1.35 * scale
        draw.ellipse((cx - r - 1, cy - r - 1, cx + r + 1, cy + r + 1), fill=(20, 20, 20))
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(255, 235, 40))
    buf = io.BytesIO()
    image.save(buf, format="WEBP", quality=72, method=6)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def target_image(
    axes: np.ndarray | None,
    heads: list[tuple[float, float]] | None,
    shape: tuple[int, int] = (SIZE, SIZE),
) -> str:
    canvas = np.zeros((*shape, 3), np.float32)
    if axes is not None:
        canvas[..., 0] = np.maximum(canvas[..., 0], np.asarray(axes, np.float32))
        canvas[..., 2] = np.maximum(canvas[..., 2], np.asarray(axes, np.float32))
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    for y, x in heads or []:
        h = np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2.0 * 1.5 ** 2))
        canvas[..., 0] = np.maximum(canvas[..., 0], h)
        canvas[..., 1] = np.maximum(canvas[..., 1], 0.92 * h)
    return encode_png(canvas)


def panel(label: str, rgb: np.ndarray, axes=None, heads=None, note: str = "") -> dict:
    return {
        "label": label,
        "base": encode_png(rgb),
        "overlay": overlay_image(rgb, axes, heads),
        "note": note,
    }


def synthetic_examples() -> list[dict]:
    examples = []
    settings = (
        (0, 1.6, 0.20, 10.0, 5.8),
        (1, 2.4, 1.15, 7.0, 7.0),
        (2, 3.4, 2.25, 6.0, 6.4),
        (3, 1.9, 4.95, 13.0, 5.3),
    )
    center = 7
    for i, speed, angle, length, snr_scale in settings:
        clip, background, donor_meta = donor_clip(i)
        pristine = clip.copy()
        dy, dx = math.sin(angle), math.cos(angle)
        start = np.array([48.0 - dy * speed * center, 48.0 - dx * speed * center])
        noise = 1.4826 * np.median(np.abs(pristine - np.median(pristine, axis=0)))
        amplitude = max(float(noise) * snr_scale, 90.0)
        heads_by_frame: dict[int, tuple[float, float]] = {}
        axes_by_frame: dict[int, np.ndarray] = {}
        for frame in range(len(clip)):
            head = start + np.array([dy, dx]) * speed * frame
            _draw_comet(clip[frame], float(head[0]), float(head[1]), dy, dx, length, 1.25, amplitude)
            tail = head - np.array([dy, dx]) * (-length * math.log(0.16))
            heads_by_frame[frame] = (float(head[0]), float(head[1]))
            axes_by_frame[frame] = exact_axis((SIZE, SIZE), heads_by_frame[frame], (float(tail[0]), float(tail[1])))
        clip = np.clip(clip, 0, 65535)
        xt, xp = normalize_rgb_pair(clip, background, center)
        bt, _ = normalize_rgb_pair(pristine, background, center)
        h0, h1 = [heads_by_frame[center]], [heads_by_frame[center + 1]]
        a0, a1 = axes_by_frame[center], axes_by_frame[center + 1]
        examples.append({
            "title": f"Procedural track {i + 1}",
            "subtitle": "Exact head, exact central axis, exact positive t→t+1 identity",
            "badges": ["on-the-fly", "exact targets", "real certified background"],
            "facts": [
                ["motion", f"{speed:.1f} px/frame"],
                ["axis decay length", f"{length:.1f} px"],
                ["donor", donor_meta["movie"]],
                ["background coverage", donor_meta["certified"]],
                ["post-hoc appearance aug", "none"],
            ],
            "panels": [
                panel("Certified donor before paste", bt, note="Same real background; no synthetic comet"),
                panel("Training input Xₜ", xt, a0, h0, "R=t−2, G=t−1, B=t"),
                panel("Training input Xₜ₊₁", xp, a1, h1, "R=t−1, G=t, B=t+1"),
                {"label": "Supervision at t / t+1", "base": target_image(a0 | a1, h0 + h1),
                 "overlay": target_image(a0 | a1, h0 + h1),
                 "note": "Yellow=head heatmaps; magenta=axis heatmaps; same-track link=positive"},
            ],
        })
    return examples


def source_track_arrays(folder: str, stem: str, label: int, decision: int):
    movie_path = DATA / folder / f"{stem}.nd2"
    mask_path = DATA / folder / f"{stem}_comet_masks.tif"
    movie = load_movie(movie_path)
    masks = load_tiff(mask_path)
    source_bg = temporal_median_background(movie)
    frames = list(range(max(0, decision - 7), min(len(movie), decision + 8)))
    union = np.any(np.stack([masks[f] == label for f in frames]), axis=0)
    yy, xx = np.nonzero(union)
    y0, y1 = max(0, int(yy.min()) - 7), min(movie.shape[1], int(yy.max()) + 8)
    x0, x1 = max(0, int(xx.min()) - 7), min(movie.shape[2], int(xx.max()) + 8)
    signals, local_masks = {}, {}
    for f in frames:
        m = masks[f, y0:y1, x0:x1] == label
        if not m.any():
            continue
        alpha = _soft_alpha(m, grow=3.0)
        residual = np.clip(movie[f, y0:y1, x0:x1].astype(np.float32) - source_bg[y0:y1, x0:x1], 0, None)
        signals[f] = residual * alpha
        local_masks[f] = m
    return movie, masks, source_bg, signals, local_masks, (y0, y1, x0, x1), movie_path


def old_track_examples() -> list[dict]:
    examples = []
    center = 7
    for i, (folder, stem, label, decision, k, flip) in enumerate(OLD_SPECS):
        movie, masks, source_bg, signals, local_masks, bbox, movie_path = source_track_arrays(
            folder, stem, label, decision
        )
        clip, donor_bg, donor_meta = donor_clip(i + 4)
        transformed = {f: d8(s, k, flip, (0, 1)) for f, s in signals.items()}
        transformed_masks = {f: d8(m, k, flip, (0, 1)) for f, m in local_masks.items()}
        h, w = next(iter(transformed.values())).shape
        py = 12 + (i * 11) % max(1, SIZE - h - 20)
        px = 14 + (i * 17) % max(1, SIZE - w - 20)
        py = min(py, SIZE - h)
        px = min(px, SIZE - w)
        pasted_axes = {}
        for local_frame in range(len(clip)):
            source_frame = decision + local_frame - center
            if source_frame not in transformed:
                continue
            clip[local_frame, py:py + h, px:px + w] += transformed[source_frame]
            ax = clean_axis(transformed_masks[source_frame])
            if ax is not None:
                full = np.zeros((SIZE, SIZE), bool)
                full[py:py + h, px:px + w] = ax
                pasted_axes[local_frame] = full
        clip = np.clip(clip, 0, 65535)
        xt, xp = normalize_rgb_pair(clip, donor_bg, center)

        # Source view uses a 96px crop centered on this particular track.
        m0 = masks[decision] == label
        sy, sx = np.nonzero(m0)
        raw_crop, frames, y0, x0 = crop_context(
            movie, decision, float(sy.mean()), float(sx.mean()), SIZE, CONTEXT, 1
        )
        raw, residual = causal_rgb_endpoints(
            movie, source_bg, frames, y0, x0, raw_crop.shape[1], raw_crop.shape[2]
        )
        src_rgb = blend_rgb(raw, residual, BLEND)[CONTEXT]
        src_axis_global = clean_axis(m0)
        src_axis = np.zeros((SIZE, SIZE), bool)
        if src_axis_global is not None:
            src_axis = src_axis_global[y0:y0 + SIZE, x0:x0 + SIZE]
        a0 = pasted_axes.get(center, np.zeros((SIZE, SIZE), bool))
        a1 = pasted_axes.get(center + 1, np.zeros((SIZE, SIZE), bool))
        examples.append({
            "title": f"Real track copy/paste {i + 1}",
            "subtitle": "Axis-only supervision derived from a hand-painted U-Net mask",
            "badges": ["real comet", "unit intensity", "D8 transform", "head unknown"],
            "facts": [
                ["source", stem],
                ["persistent label", str(label)],
                ["transform", f"rotate {90 * k}°" + (" + reflect" if flip else "")],
                ["donor", donor_meta["movie"]],
                ["same-track link", "positive"],
                ["post-hoc appearance aug", "none"],
            ],
            "panels": [
                panel("Original labeled track at t", src_rgb, src_axis, None,
                      "Original mask skeleton; zero outside mask remains unlabeled"),
                panel("Augmented training input Xₜ", xt, a0, None, "Pasted additively on a different real background"),
                panel("Augmented training input Xₜ₊₁", xp, a1, None, "Same transformed track one frame later"),
                {"label": "Axis supervision at t / t+1", "base": target_image(a0 | a1, None),
                 "overlay": target_image(a0 | a1, None),
                 "note": "Magenta=derived axis. No head loss is applied to these examples."},
            ],
        })
    return examples


def queue_by_id() -> dict[str, dict]:
    return {row["candidate_id"]: row for row in read_queue(QUEUE)}


def choose_current_reviews(labels: dict, queue: dict[str, dict], count: int = 4):
    candidates = []
    for cid, review in labels["reviews"].items():
        row = queue.get(cid)
        if not row or review.get("verdict") != "both":
            continue
        if review.get("quality_warnings"):
            continue
        candidates.append((row["category"], row["movie"], cid, row, review))
    selected, used_movies, used_categories = [], set(), set()
    for category, movie, cid, row, review in sorted(candidates):
        if movie in used_movies or category in used_categories:
            continue
        selected.append((cid, row, review))
        used_movies.add(movie)
        used_categories.add(category)
        if len(selected) == count:
            return selected
    for category, movie, cid, row, review in sorted(candidates):
        if cid not in {s[0] for s in selected} and movie not in used_movies:
            selected.append((cid, row, review))
            used_movies.add(movie)
        if len(selected) == count:
            break
    return selected


def current_examples() -> list[dict]:
    labels = json.loads(CURRENT_LABELS.read_text())
    queue = queue_by_id()
    examples = []
    transforms = ((1, True), (2, False), (3, True), (1, False))
    for i, ((cid, row, review), (k, flip)) in enumerate(zip(choose_current_reviews(labels, queue), transforms)):
        movie = load_movie(row["movie_path"])
        frame = int(row["frame"])
        _crop, frames, y0, x0 = crop_context(
            movie, frame, float(row["y"]), float(row["x"]), SIZE, CONTEXT, 1
        )
        bg = temporal_median_background(movie)
        raw, residual = causal_rgb_endpoints(movie, bg, frames, y0, x0, SIZE, SIZE)
        shown = blend_rgb(raw, residual, BLEND)
        t_index = frames.index(frame)
        original_t, original_p = shown[t_index], shown[t_index + 1]
        heads = {int(p["frame"]): (float(p["y"]) - y0, float(p["x"]) - x0)
                 for p in review["head_points"]}
        axis_by_frame = {}
        for f in (frame, frame + 1):
            ax = np.zeros((SIZE, SIZE), bool)
            for p in review["axis_pixels"]:
                if int(p["frame"]) != f:
                    continue
                yy, xx = int(p["y"]) - y0, int(p["x"]) - x0
                if 0 <= yy < SIZE and 0 <= xx < SIZE:
                    ax[yy, xx] = True
            axis_by_frame[f] = ax
        aug_t = d8(original_t, k, flip, (0, 1))
        aug_p = d8(original_p, k, flip, (0, 1))
        a0 = d8(axis_by_frame[frame], k, flip, (0, 1))
        a1 = d8(axis_by_frame[frame + 1], k, flip, (0, 1))
        h0 = [d8_yx(*heads[frame], SIZE, k, flip)]
        h1 = [d8_yx(*heads[frame + 1], SIZE, k, flip)]
        examples.append({
            "title": f"Current annotation augmentation {i + 1}",
            "subtitle": "Accepted real head, axis, and same-comet t→t+1 pair",
            "badges": ["human accepted", "head + axis", "D8 transform", row["category"]],
            "facts": [
                ["movie", row["movie"]],
                ["candidate", cid],
                ["transform", f"rotate {90 * k}°" + (" + reflect" if flip else "")],
                ["head displacement", f"{review.get('head_distance_pixels_0_to_plus1', float('nan')):.2f} px"],
                ["same-track link", "positive"],
                ["post-hoc appearance aug", "none"],
            ],
            "panels": [
                panel("Original accepted Xₜ", original_t, axis_by_frame[frame], [heads[frame]],
                      "Saved one-pixel axis is widened only in this display"),
                panel("Augmented training input Xₜ", aug_t, a0, h0, "Image and targets transformed together"),
                panel("Augmented training input Xₜ₊₁", aug_p, a1, h1, "Same D8 transform applied to the pair"),
                {"label": "Supervision at t / t+1", "base": target_image(a0 | a1, h0 + h1),
                 "overlay": target_image(a0 | a1, h0 + h1),
                 "note": "Yellow=head heatmaps; magenta=canonical axes; association=positive"},
            ],
        })
    return examples


def html_fragment(payload: dict) -> str:
    data = json.dumps(payload, separators=(",", ":"))
    return f'''<div id="sam3-training-preview">
  <style>
    #sam3-training-preview {{ color: var(--foreground); font: 13px/1.4 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    #sam3-training-preview * {{ box-sizing: border-box; }}
    #sam3-training-preview .title {{ font-size: 20px; font-weight: 720; letter-spacing: -0.02em; margin: 0 0 3px; }}
    #sam3-training-preview .sub {{ color: color-mix(in srgb, var(--foreground) 68%, transparent); margin-bottom: 14px; }}
    #sam3-training-preview .stats {{ display: grid; grid-template-columns: repeat(3,minmax(0,1fr)); gap: 8px; margin-bottom: 12px; }}
    #sam3-training-preview .stat {{ border: 1px solid var(--border); border-radius: 8px; padding: 9px 10px; }}
    #sam3-training-preview .stat b {{ display:block; font-size:15px; }}
    #sam3-training-preview .stat span {{ color: color-mix(in srgb, var(--foreground) 65%, transparent); font-size:12px; }}
    #sam3-training-preview .tabs, #sam3-training-preview .toolbar {{ display:flex; flex-wrap:wrap; align-items:center; gap:6px; margin: 8px 0; }}
    #sam3-training-preview button {{ border:1px solid var(--border); background:transparent; color:var(--foreground); border-radius:7px; padding:6px 9px; cursor:pointer; font:inherit; }}
    #sam3-training-preview button[aria-pressed="true"] {{ background:color-mix(in srgb,var(--foreground) 12%,transparent); border-color:color-mix(in srgb,var(--foreground) 35%,var(--border)); }}
    #sam3-training-preview .sample-select {{ margin-left:auto; display:flex; gap:5px; align-items:center; }}
    #sam3-training-preview .sample-index {{ min-width:72px; text-align:center; font-variant-numeric:tabular-nums; }}
    #sam3-training-preview .example {{ border-top:1px solid var(--border); padding-top:12px; }}
    #sam3-training-preview .example-head {{ display:flex; flex-wrap:wrap; justify-content:space-between; gap:7px; align-items:start; margin-bottom:10px; }}
    #sam3-training-preview h3 {{ margin:0; font-size:16px; }}
    #sam3-training-preview .example-sub {{ color:color-mix(in srgb,var(--foreground) 67%,transparent); font-size:12px; }}
    #sam3-training-preview .badges {{ display:flex; flex-wrap:wrap; gap:5px; }}
    #sam3-training-preview .badge {{ border:1px solid var(--border); border-radius:99px; padding:2px 7px; font-size:11px; }}
    #sam3-training-preview .panels {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:8px; }}
    #sam3-training-preview figure {{ margin:0; min-width:0; }}
    #sam3-training-preview figure img {{ display:block; width:100%; aspect-ratio:1; object-fit:contain; image-rendering:pixelated; background:#050505; border:1px solid var(--border); border-radius:6px; }}
    #sam3-training-preview figcaption {{ padding-top:5px; }}
    #sam3-training-preview figcaption b {{ display:block; font-size:12px; }}
    #sam3-training-preview figcaption span {{ display:block; min-height:31px; color:color-mix(in srgb,var(--foreground) 63%,transparent); font-size:11px; }}
    #sam3-training-preview .facts {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:5px 12px; border-top:1px solid var(--border); margin-top:9px; padding-top:9px; }}
    #sam3-training-preview .fact {{ display:flex; justify-content:space-between; gap:8px; min-width:0; }}
    #sam3-training-preview .fact span:first-child {{ color:color-mix(in srgb,var(--foreground) 60%,transparent); }}
    #sam3-training-preview .fact span:last-child {{ text-align:right; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
    #sam3-training-preview .legend {{ display:flex; gap:14px; align-items:center; margin-left:4px; font-size:12px; color:color-mix(in srgb,var(--foreground) 72%,transparent); }}
    #sam3-training-preview .swatch {{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:4px; vertical-align:-1px; }}
    #sam3-training-preview .yellow {{ background:#ffeb28; }}
    #sam3-training-preview .magenta {{ background:#ff2dd2; border-radius:1px; }}
    @media (max-width: 760px) {{
      #sam3-training-preview .panels {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
      #sam3-training-preview .facts {{ grid-template-columns:1fr; }}
      #sam3-training-preview .stats {{ grid-template-columns:1fr; }}
      #sam3-training-preview .sample-select {{ margin-left:0; }}
    }}
  </style>
  <div class="title">SAM3 training-data audit</div>
  <div class="sub">Actual causal RGB inputs and transformed targets · R=t−2, G=t−1, B=t · conservative 50% background blend</div>
  <div class="stats" id="sam3-stats"></div>
  <div class="tabs" id="sam3-tabs" aria-label="Training data source"></div>
  <div class="toolbar">
    <button type="button" id="sam3-overlay" aria-pressed="true">Targets over input</button>
    <div class="legend"><span><i class="swatch yellow"></i>head</span><span><i class="swatch magenta"></i>axis</span></div>
    <div class="sample-select">
      <button type="button" id="sam3-prev" aria-label="Previous example">←</button>
      <span class="sample-index" id="sam3-index"></span>
      <button type="button" id="sam3-next" aria-label="Next example">→</button>
    </div>
  </div>
  <section class="example" id="sam3-example" aria-live="polite"></section>
  <script>
    (() => {{
      const root = document.getElementById('sam3-training-preview');
      const payload = {data};
      const state = {{ source: 0, sample: 0, overlay: true }};
      const stats = root.querySelector('#sam3-stats');
      const tabs = root.querySelector('#sam3-tabs');
      const example = root.querySelector('#sam3-example');
      const index = root.querySelector('#sam3-index');
      const overlay = root.querySelector('#sam3-overlay');
      stats.innerHTML = payload.stats.map(s => `<div class="stat"><b>${{s.value}}</b><span>${{s.label}}</span></div>`).join('');
      payload.sources.forEach((source, i) => {{
        const button = document.createElement('button');
        button.type = 'button'; button.textContent = source.name;
        button.setAttribute('aria-pressed', i === 0 ? 'true' : 'false');
        button.addEventListener('click', () => {{ state.source = i; state.sample = 0; render(); }});
        tabs.appendChild(button);
      }});
      function render() {{
        const source = payload.sources[state.source];
        const item = source.examples[state.sample];
        [...tabs.children].forEach((b,i) => b.setAttribute('aria-pressed', i === state.source ? 'true' : 'false'));
        overlay.setAttribute('aria-pressed', state.overlay ? 'true' : 'false');
        index.textContent = `${{state.sample + 1}} / ${{source.examples.length}}`;
        const badges = item.badges.map(b => `<span class="badge">${{b}}</span>`).join('');
        const panels = item.panels.map(p => `<figure><img src="${{state.overlay ? p.overlay : p.base}}" alt="${{p.label}}"><figcaption><b>${{p.label}}</b><span>${{p.note}}</span></figcaption></figure>`).join('');
        const facts = item.facts.map(f => `<div class="fact"><span>${{f[0]}}</span><span title="${{f[1]}}">${{f[1]}}</span></div>`).join('');
        example.innerHTML = `<div class="example-head"><div><h3>${{item.title}}</h3><div class="example-sub">${{item.subtitle}}</div></div><div class="badges">${{badges}}</div></div><div class="panels">${{panels}}</div><div class="facts">${{facts}}</div>`;
      }}
      overlay.addEventListener('click', () => {{ state.overlay = !state.overlay; render(); }});
      root.querySelector('#sam3-prev').addEventListener('click', () => {{
        const n = payload.sources[state.source].examples.length;
        state.sample = (state.sample - 1 + n) % n; render();
      }});
      root.querySelector('#sam3-next').addEventListener('click', () => {{
        const n = payload.sources[state.source].examples.length;
        state.sample = (state.sample + 1) % n; render();
      }});
      render();
    }})();
  </script>
</div>
'''


def build(output: Path) -> None:
    current = json.loads(CURRENT_LABELS.read_text())
    reviews = list(current["reviews"].values())
    head_pairs = sum(bool(r.get("head_accepted")) for r in reviews)
    axis_pairs = sum(bool(r.get("axis_accepted")) for r in reviews)
    payload = {
        "stats": [
            {"value": "On-the-fly", "label": "procedural clips with exact head, axis, and identity"},
            {"value": "131 tracks · 1,160 clean axes", "label": "old hand-painted U-Net-mask corpus"},
            {"value": f"{2 * head_pairs} heads · {2 * axis_pairs} axes", "label": "accepted current frame-level targets"},
        ],
        "sources": [
            {"name": "1 · Procedural synthetic", "examples": synthetic_examples()},
            {"name": "2 · Pasted U-Net tracks", "examples": old_track_examples()},
            {"name": "3 · Current annotations", "examples": current_examples()},
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_fragment(payload))
    print(f"wrote {output} ({output.stat().st_size / 1024:.1f} KiB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.output.resolve())


if __name__ == "__main__":
    main()
