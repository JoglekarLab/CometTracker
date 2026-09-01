"""Render a movie with V7 track paths drawn on it, as an mp4.

Side by side: the raw frame on the left, the same frame with tracks on the
right. Tracks are coloured by MOTION CLASS, because the thing worth seeing is
that the immobile population is separated from the growing one -- on these
movies that separation is what moves the measured growth rate from ~8 um/min
(everything pooled) to ~11 um/min (directed only).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comet_tracker_v7.config import MICRONS_PER_PIXEL, SECONDS_PER_FRAME, Config
from comet_tracker_v7.pipeline import run_folder

# Brighter siblings of the page's validated chart palette (#22A5A8 / #C57A2C),
# so the video and the charts use ONE encoding. Teal vs ochre rather than the
# obvious green vs red: at equal lightness green and red are indistinguishable
# under deuteranopia (measured dE 0.6), while this pair scores 15.8.
CLASS_COLOR = {
    "directed":  (69, 210, 214),
    "brownian":  (229, 160, 82),
    "confined":  (229, 160, 82),
    "immobile":  (229, 160, 82),
    None:        (150, 150, 150),
}
DIM = {"immobile": 0.45, "confined": 0.8, "brownian": 0.8, "directed": 1.0}


def stretch(frame, lo_p=1.0, hi_p=99.7):
    lo, hi = np.percentile(frame, [lo_p, hi_p])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((frame.astype(np.float32) - lo) / (hi - lo), 0, 1)


def render(nd2_path, pred_dir, stem, out_path, hold=0, fps=10, scale=1):
    import nd2

    movie = nd2.imread(nd2_path)
    result = run_folder(pred_dir, stem, Config())

    # per-frame drawing lists: (points_so_far, colour, class)
    paths = []
    for c in result.compounds:
        pts, frames = [], []
        for seg in c.segments:
            p = seg.positions()
            pts.extend([(float(x), float(y)) for y, x in p])   # (col, row)
            frames.extend(seg.frames)
        paths.append((np.asarray(frames), pts, c.motion_class))

    n_frames, h, w = movie.shape
    tmp = tempfile.mkdtemp()
    for t in range(n_frames):
        g = (stretch(movie[t]) * 255).astype(np.uint8)
        base = Image.fromarray(g).convert("RGB")
        over = base.copy()
        d = ImageDraw.Draw(over, "RGBA")
        for frames, pts, cls in paths:
            # A track is drawn only while it is ALIVE: from its first frame to
            # its last, showing the path travelled so far. Once it ends it
            # disappears (``hold`` frames later, if a brief hold is wanted).
            #
            # The previous version used a fixed 40-frame trail, which on a
            # 91-frame movie meant a dead track lingered for most of the
            # remaining playback -- so the field filled up with the history of
            # comets that were no longer there, and it was impossible to see
            # what the tracker was actually following at any given moment.
            if t < frames[0] or t > frames[-1] + hold:
                continue
            vis = np.flatnonzero(frames <= t)
            if vis.size < 2:
                if vis.size == 1:
                    x, y = pts[vis[0]]
                    col = CLASS_COLOR.get(cls, CLASS_COLOR[None])
                    d.ellipse([x - 2, y - 2, x + 2, y + 2],
                              outline=col + (200,), width=1)
                continue
            col = CLASS_COLOR.get(cls, CLASS_COLOR[None])
            a = int(255 * DIM.get(cls, 1.0))
            seq = [pts[i] for i in vis]
            d.line(seq, fill=col + (a,), width=2 if cls == "directed" else 1)
            x, y = seq[-1]
            r = 3 if cls == "directed" else 2
            d.ellipse([x - r, y - r, x + r, y + r], fill=col + (a,))

        canvas = Image.new("RGB", (w * 2 + 6, h), (16, 16, 18))
        canvas.paste(base, (0, 0))
        canvas.paste(over, (w + 6, 0))
        dd = ImageDraw.Draw(canvas)
        dd.text((6, 6), f"raw   frame {t:3d}/{n_frames - 1}   "
                       f"t = {t * SECONDS_PER_FRAME:5.1f} s", fill=(235, 235, 235))
        dd.text((w + 12, 6), "V7 tracks", fill=(235, 235, 235))
        if scale != 1:
            canvas = canvas.resize((canvas.width * scale, canvas.height * scale),
                                   Image.LANCZOS)
        canvas.save(os.path.join(tmp, f"f{t:04d}.png"))

    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
         "-i", os.path.join(tmp, "f%04d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "31",
         "-movflags", "+faststart", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
         out_path], check=True)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nd2", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = render(a.nd2, a.pred, a.stem, a.out)
    print(f"{a.out}: {len(r.compounds)} tracks, "
          f"{os.path.getsize(a.out)/1e6:.2f} MB")
