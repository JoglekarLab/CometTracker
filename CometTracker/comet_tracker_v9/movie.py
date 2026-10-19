"""Read one movie the same way everywhere: the 488 channel only, at its real scale.

Two facts about the acquisitions that the files themselves do not tell you
correctly, and that every step of the pipeline has to agree on.

THE CHANNEL. Most movies from October 2026 on carry a second channel (640 nm).
Only the 488 channel -- EB3 -- is analysed. nd2 puts the channel axis SECOND,
(T, C, Y, X), so code that assumed (T, Y, X, C) and took ``[..., 0]`` would
slice the image instead of choosing a channel. The channel is picked by NAME,
so the order it was acquired in does not matter; an unnamed multi-channel file
falls back to index 0 and says so.

THE ZOOM. The scope has a 1.5x optovar that NIS-Elements does not record:
every file says "Zoom: 1.00x" and 0.1589 um/px whatever the optovar was set to.
Proved on GW16/20261007 _011 against _012, the same cell before and after the
switch: _012 is _011 magnified by exactly 1.50 (correlation 0.93, falling off
on either side). So the metadata pixel size is wrong for every 1.5x movie and
cannot be used. The zoom comes from a table set by eye instead, and

    1.5x is the default -- the normal setting, and what every movie is
         unless the table says otherwise. Pixel = 0.158876 / 1.5 = 0.105917 um
    1x   only for the movies the table marks as 1.  Pixel = 0.158876 um

The table is a CSV with at least the columns ``stem`` and ``zoom``; lines
starting with ``#`` are comments. ``zoom_table.csv`` at the package root is
the one the cluster run uses.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field

import numpy as np

# what NIS reports for every file: the 100x objective with no optovar
MICRONS_PER_PIXEL_1X = 0.158875614664932
DEFAULT_ZOOM = 1.5
EB3_CHANNEL = "488"


@dataclass
class MovieInfo:
    n_channels: int = 1
    channel_index: int = 0
    channel_name: str | None = None
    channel_note: str | None = None
    seconds_per_frame: float | None = None      # from the metadata, if it has one
    pixel_um_metadata: float | None = None      # what NIS claims -- NOT used
    shape_tyx: tuple = ()
    extra: dict = field(default_factory=dict)


def read_movie(path: str, channel: str = EB3_CHANNEL):
    """(T, Y, X) float32 of the requested channel, and what was found.

    Raises if the file cannot be read; a damaged .nd2 is reported as such
    rather than half-read.
    """
    import nd2

    info = MovieInfo()
    with nd2.ND2File(str(path)) as f:
        sizes = dict(f.sizes)
        arr = np.asarray(f.asarray())
        try:
            info.pixel_um_metadata = float(f.voxel_size().x)
        except Exception:
            pass
        names = []
        try:
            names = [c.channel.name for c in f.metadata.channels]
        except Exception:
            pass
        try:
            ms = f.experiment[0].parameters.periodMs
            if ms and ms > 0:
                info.seconds_per_frame = float(ms) / 1000.0
        except Exception:
            pass

    axes = [str(a).upper() for a in sizes]
    if "C" in axes:
        info.n_channels = int(sizes["C"])
        hit = [i for i, n in enumerate(names) if n and channel in str(n)]
        if hit:
            info.channel_index = hit[0]
        else:
            info.channel_index = 0
            info.channel_note = (f"no channel named '{channel}' among {names or 'unnamed'};"
                                 " took index 0")
        info.channel_name = names[info.channel_index] if names else None
        arr = np.take(arr, info.channel_index, axis=axes.index("C"))
        axes.remove("C")
    else:
        info.channel_name = names[0] if names else None

    for name in ("Z",):                      # a z-stack is max-projected, as SAM3 does
        if name in axes:
            arr = arr.max(axis=axes.index(name)); axes.remove(name)
    for name in list(axes):                  # anything else exotic: first slice
        if name not in ("T", "Y", "X"):
            arr = np.take(arr, 0, axis=axes.index(name)); axes.remove(name)
    if "T" not in axes:
        arr = arr[None]; axes.insert(0, "T")
    arr = np.transpose(arr, [axes.index(a) for a in ("T", "Y", "X")])
    if arr.ndim != 3:
        raise ValueError(f"expected (T, Y, X) after channel selection, got {arr.shape}")
    info.shape_tyx = tuple(int(s) for s in arr.shape)
    return np.ascontiguousarray(arr, dtype=np.float32), info


class ZoomTable:
    """stem -> zoom, with every movie not listed taken at the default (1.5x)."""

    def __init__(self, path: str | None = None, default: float = DEFAULT_ZOOM):
        self.path = path
        self.default = float(default)
        self.zoom: dict[str, float] = {}
        if path:
            self._load(path)

    def _load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"zoom table not found: {path}")
        with open(path, newline="") as fh:
            rows = [ln for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
        rd = csv.DictReader(rows)
        if not rd.fieldnames or "stem" not in rd.fieldnames or "zoom" not in rd.fieldnames:
            raise ValueError(f"{path}: needs 'stem' and 'zoom' columns, has {rd.fieldnames}")
        for r in rd:
            stem = (r.get("stem") or "").strip()
            if not stem:
                continue
            z = float(r["zoom"])
            if z not in (1.0, 1.5):
                raise ValueError(f"{path}: zoom {z} for {stem}; expected 1 or 1.5")
            self.zoom[stem] = z

    def lookup(self, stem: str):
        """(zoom, pixel_um, source) for one movie."""
        if stem in self.zoom:
            z, src = self.zoom[stem], "table"
        else:
            z, src = self.default, "default"
        return z, MICRONS_PER_PIXEL_1X / z, src

    def n_at(self, z: float) -> int:
        return sum(1 for v in self.zoom.values() if v == z)


def probe_movie(path: str, channel: str = EB3_CHANNEL) -> dict:
    """Cheap check of one file: can it be opened and read, and where is 488.

    Reads the metadata and the first and last frames only. ``sam3_ok`` says
    whether SAM3's reader -- which takes channel INDEX 0, not a name -- gets
    the 488 channel from this file.
    """
    import nd2

    out = dict(ok=False, n_channels=None, channels=None, index_488=None,
               sam3_ok=None, n_frames=None, error=None)
    try:
        with nd2.ND2File(str(path)) as f:
            sizes = dict(f.sizes)
            try:
                names = [c.channel.name for c in f.metadata.channels]
            except Exception:
                names = []
            out["n_frames"] = int(sizes.get("T", 1))
            out["n_channels"] = int(sizes.get("C", 1))
            out["channels"] = names
            hit = [i for i, n in enumerate(names) if n and channel in str(n)]
            out["index_488"] = hit[0] if hit else None
            out["sam3_ok"] = out["index_488"] == 0
            n_seq = f.attributes.sequenceCount
            for i in (0, n_seq - 1):
                np.array(f.read_frame(i), copy=True)    # raises on a damaged file
        out["ok"] = True
    except Exception as exc:                            # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
    return out
