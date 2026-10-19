"""Per-detection photometry and a comet length fitted from the image.

    from .photometry import measure
    phot = measure(movie, labels, prob, cfg.detect)

Everything in here is measured on RAW PIXELS. Nothing reads SAM3's probability
except to reproduce the detection list, so the numbers are independent of the
detector's confidence -- which is the whole point, because ``mask_len_um`` is
not: mask major axis nearly doubles across presence bins (9.4 -> 17.6 px) while
the minor axis stays flat.

WHAT IS MEASURED, PER DETECTION

Intensity, all three background-subtracted, background being the median of the
unlabelled pixels in a padded box around the comet (so a neighbouring comet is
excluded by its own label, not by luck):

    peak        brightest pixel in the mask, minus background
    integrated  sum over the mask of (pixel - background)
    mean        mean over the mask, minus background

Length, from a profile sampled along the mask's major axis. The profile is a
stripe 3 px wide averaged across, bilinearly interpolated at 0.5 px steps, out
to the mask length plus 10 px on each side so the fit sees where the signal
actually ends rather than where the mask stops:

    gauss_fwhm_um   FWHM of a Gaussian + offset fitted to the profile
    exp_lambda_um   decay constant of an exponential fitted to the tail only,
                    the tail being whichever side of the peak decays slower

The two disagree by construction, and that is useful. An EB comet is not
symmetric -- it is sharp at the tip and decays behind -- so a Gaussian FWHM is
a compromise between the two sides while lambda describes only the decay. If
the cell lines rank the same way under both, the ranking is not an artefact of
the model.

R^2 is written for each fit. A comparison should be repeated on the subset with
a decent fit before it is believed.
"""
from __future__ import annotations

import numpy as np

__all__ = ["measure", "FIELDS"]

FIELDS = ("peak", "integrated", "mean", "bg",
          "auc", "auc_extent_px", "auc_side", "auc_extent_side_px",
          "peak_prof", "bg_side", "bg_side_n", "area",
          "peak_side", "integrated_side", "mean_side", "peak_prof_side",
          "biel_len_side_px", "biel_side_r2",
          "gauss_fwhm_px", "gauss_r2", "exp_lambda_px", "exp_r2",
          "emg_len_px", "emg_sigma_px", "emg_fwhm_px", "emg_r2",
          "biel_len_px", "biel_r2", "biel_len_off_px", "biel_off_r2",
          "profile_snr",
          # v9: background from a box turned with the comet, and the old one
          "bg_patch", "bg_box_n",
          # v9: EMG positions for speed, once for EACH way the comet could
          # point -- see _emg_positions. "_p": plus end toward +axis,
          # "_n": plus end toward -axis. The track picks one from its own
          # direction of travel.
          *[f"emg_{k}_{d}" for d in ("p", "n")
            for k in ("window", "ok", "mu_y", "mu_x", "max_y", "max_x",
                      "mu_err_px", "len_px", "sigma_px", "r2")])

_PAD = 12          # v8 only: px added around the bbox for the old background
_STRIPE = 1.5      # px half-width of the sampling stripe
_STEP = 0.5        # px along the axis
_EXTRA = 28.0      # px sampled beyond each end of the mask. 28 and not
                   # 10 because the Bieling fit needs a full 25 px behind
                   # the tip, and a window that stops short of 25 would
                   # silently shorten lambda for the longest comets.
_SHORT_EXTRA = 10.0  # px; the window the Gaussian, tail and EMG fits see.
                     # They are hurt by the long window Bieling needs: going
                     # from 10 to 28 px inflated the tail from 0.92 to 1.14 um
                     # and dropped the Gaussian from 78% to 63% convergence.
# Sideways background. The distance is fixed, the same for every comet, and was
# read off a pooled lateral profile of 520 isolated comets: the excess over the
# far field falls to +11% at 3 px, +4.4% at 5 px, +2.0% at 7 px and under 1%
# past 8. Choosing the distance per comet would make a bright comet -- which
# contaminates further out -- get a systematically different background from a
# dim one, so it is chosen once, globally, and left alone.
#
# 5-7 and not 7-10: measured on the same 520 comets, 5-7 reads +18 counts above
# the 7-10 value and costs 8% of the area under the curve -- but that 8% is FLAT
# in brightness (x0.923 for the dimmest quartile, x0.929 for the brightest), so
# it is an offset every comet pays rather than a bias that tracks the signal.
# 3-5 was rejected for exactly that reason: x0.74 dim against x0.81 bright.
_SIDE_LO = 5.0     # px from the axis
_SIDE_HI = 7.0     # px
_SIDE_MIN_PX = 20  # below this the band is too small to take a median from
_LAM_MAX = 40.0    # px; 6.4 um. Longer than any real comet decay here, so a
                   # fit that reaches it is a failure, not a measurement.
_FWHM_MAX = 45.0   # px; same idea for the Gaussian

# v9 background: a box TURNED WITH THE COMET instead of the image-aligned
# bounding box + 40 px. The old box gave a diagonal comet a huge square full of
# unrelated area -- bundles, cell edge, glass. The new one runs the length of
# the mask (+2 px) and sits beside it, 5 to 9 px out. Not closer than 5: the
# comet's own light still adds +11% at 3 px and +4.4% at 5 px (the pooled
# lateral profile that set _SIDE_LO above). Not ahead of or behind the comet:
# behind is the lattice, which carries EB.
_BOX_END = 2.0     # px past each end of the mask
_BOX_LO = 5.0      # px from the axis
_BOX_HI = 9.0      # px
_BOX_MIN_PX = 20

# v9 EMG positions for speed. Decided 2026-10-12 on 1,292 comets of
# 20260710_pAJV103_0.25DOX-ON_010:
_EMG_SHORT_PX = 4.0     # mask shorter than this: no fit, no position
_EMG_R2_REFIT = 0.7     # below this the wide fit triggers the mask-only refit
_EMG_SLACK_PX = 2.0     # max / mu may sit this far past the mask end. SAM3
                        # masks shrink at the ends as confidence drops, so the
                        # real plus end can lie just outside the mask
_EMG_MU_ERR_MAX = 1.0   # px; the fit's own 1-sigma on mu
_EMG_SD_MIN, _EMG_SD_MAX = 0.5, 4.0   # px; one comet blurs to ~1-2 px. Wider
                        # is two comets in one mask, or a comet on a bundle


def _bg_sideways(patch, free, cy, cx, theta, half_major):
    """Median intensity in a band running alongside the comet.

    Sideways, not around: behind the tip is the lattice, which carries EB and
    is signal, and ahead of the tip is empty space. Only the lateral direction
    is local cytoplasm. Pixels inside ANY mask are excluded, so a neighbour is
    dropped by its own label rather than by luck.

    Returns (median, n_pixels). NaN when the band is too small to trust.
    """
    H, W = patch.shape
    yy, xx = np.mgrid[0:H, 0:W]
    dy = yy - cy
    dx = xx - cx
    uy, ux = np.cos(theta), np.sin(theta)
    along = dy * uy + dx * ux
    perp = -dy * ux + dx * uy
    m = (free & (np.abs(along) <= half_major)
         & (np.abs(perp) >= _SIDE_LO) & (np.abs(perp) < _SIDE_HI))
    n = int(m.sum())
    if n < _SIDE_MIN_PX:
        return np.nan, n
    return float(np.median(patch[m])), n


def _bg_box(patch, own, free, cy, cx, theta, half_major):
    """Background from a box turned with the comet. Returns (median, mad_sd, n).

    ``own`` is this comet's mask in the patch, ``free`` the pixels in no mask.
    Falls back to (NaN, NaN, n) when fewer than _BOX_MIN_PX pixels qualify;
    the caller then uses the sideways band, then the patch.
    """
    H, W = patch.shape
    yy, xx = np.mgrid[0:H, 0:W]
    dy, dx = yy - cy, xx - cx
    uy, ux = np.cos(theta), np.sin(theta)
    along = dy * uy + dx * ux
    perp = -dy * ux + dx * uy
    m = (free & (np.abs(along) <= half_major + _BOX_END)
         & (np.abs(perp) >= _BOX_LO) & (np.abs(perp) <= _BOX_HI))
    n = int(m.sum())
    if n < _BOX_MIN_PX:
        return np.nan, np.nan, n
    v = patch[m]
    med = float(np.median(v))
    return med, float(1.4826 * np.median(np.abs(v - med))), n


def _emg_core(s, y, floor=None, flips=(1.0, -1.0)):
    """Best of _emg_each over ``flips`` (highest R^2), or None."""
    each = [r for r in _emg_each(s, y, floor, flips).values() if r is not None]
    return max(each, key=lambda r: r["r2"]) if each else None


def _emg_each(s, y, floor=None, flips=(1.0, -1.0)):
    """One EMG fit, each direction in ``flips`` tried. ``floor=0.0`` fixes the
    offset. flip +1 puts the tail toward +s (plus end facing -s); -1 the
    reverse.

    Returns {flip: None or a dict} with the plus end (mu_s) and the curve's top
    (max_s) in the PROFILE's own coordinate s, plus lam, sd, r2 and mu_err
    (the fit's 1-sigma on mu, px). Same validity rules as _fit_emg.
    """
    from scipy.optimize import curve_fit

    res = {f: None for f in flips}
    if s.size < 6 or not np.isfinite(y).all():
        return res
    c0 = 0.0 if floor is not None else float(np.percentile(y, 10))
    A0 = float(y.max() - c0)
    if A0 <= 0:
        return res
    mu0 = float(s[int(np.argmax(y))])
    for flip in flips:
        x = s * flip
        order = np.argsort(x)
        xx, yy = x[order], y[order]
        try:
            if floor is None:
                p, cv = curve_fit(_emg, xx, yy, p0=[A0, 4.0, mu0 * flip, 1.3, c0],
                                  maxfev=4000)
            else:
                q, cv = curve_fit(lambda t, A, L, mu, sd: _emg(t, A, L, mu, sd, floor),
                                  xx, yy, p0=[A0, 4.0, mu0 * flip, 1.3], maxfev=4000)
                p = np.r_[q, floor]
        except Exception:                                   # noqa: BLE001
            continue
        lam, sd = float(abs(p[1])), float(abs(p[3]))
        if not np.isfinite(lam) or lam <= 0.3 or lam >= _LAM_MAX:
            continue
        if not np.isfinite(sd) or sd <= 0.2 or sd > 8.0:
            continue
        if not (xx[0] <= p[2] <= xx[-1]):
            continue
        with np.errstate(all="ignore"):
            fit = _emg(xx, *p)
        ss = float(((yy - yy.mean()) ** 2).sum())
        r2 = 1.0 - float(((yy - fit) ** 2).sum()) / ss if ss > 0 else float("nan")
        if not np.isfinite(r2):
            continue
        err = float(np.sqrt(cv[2, 2])) if np.all(np.isfinite(cv)) and cv[2, 2] >= 0 \
            else float("inf")
        xf = np.linspace(xx[0], xx[-1], max(400, 20 * xx.size))
        with np.errstate(all="ignore"):
            yf = _emg(xf, *p)
        res[flip] = dict(lam=lam, sd=sd, r2=r2, mu_err=err,
                         mu_s=float(p[2] * flip),
                         max_s=float(xf[int(np.nanargmax(yf))] * flip))
    return res


def _emg_usable(r):
    return (r is not None and r["mu_err"] < _EMG_MU_ERR_MAX
            and _EMG_SD_MIN <= r["sd"] <= _EMG_SD_MAX)


_UNSET = object()


def _emg_positions(s_near, y_near, s_all, y_all, half, plus_dir, wide=_UNSET):
    """Where the plus end (mu) and the curve's top (max) are, for speed.

    The rule, as decided:
      1. mask shorter than _EMG_SHORT_PX -> no fit. The frame has no EMG
         position and is left out of the EMG speed. (window -1)
      2. the normal fit on the mask +- 10 px ("wide").
      3. if it is poor (R^2 < 0.7, or no valid fit) OR its max / mu lies more
         than _EMG_SLACK_PX past the mask end -> refit on the MASK ONLY with
         the floor fixed at 0 (= background, already subtracted). A free
         floor cannot be used there: inside the mask the tail never reaches
         background, so the fit raises the floor and shortens the comet
         (x0.66, plus end 0.6 px back on comets that fit fine wide).
      4. the frame is kept only if the fit places mu to within 1 px and its
         blur is 0.5-4 px. If the refit fails that, the wide fit is kept
         instead when IT passes and lies within the mask -- R^2 alone was
         shown to throw away dim comets whose plus end is well placed.
    ``plus_dir`` (+1 or -1) is the side of the profile the plus end is
    ASSUMED to face. The fit is only allowed that way round. Run once per
    direction; the track chooses afterwards. Letting each frame choose by R^2
    instead put the plus end on the wrong side in 30% of usable frames.
    window: 1 = wide, 2 = mask-only, 0 = no usable fit, -1 = mask too short.
    """
    flips = (-float(plus_dir),)
    out = dict(window=0.0, ok=0.0, mu_s=np.nan, max_s=np.nan, mu_err=np.nan,
               lam=np.nan, sd=np.nan, r2=np.nan)
    if 2.0 * half < _EMG_SHORT_PX:
        out["window"] = -1.0
        return out
    if wide is _UNSET:                       # not precomputed by the caller
        wide = _emg_core(s_near, y_near, flips=flips)

    def inside(r):
        return (abs(r["mu_s"]) <= half + _EMG_SLACK_PX
                and abs(r["max_s"]) <= half + _EMG_SLACK_PX)

    chosen, win = None, 0.0
    poor = wide is None or not (wide["r2"] >= _EMG_R2_REFIT)
    if not poor and inside(wide):
        if _emg_usable(wide):
            chosen, win = wide, 1.0
    else:
        m = np.abs(s_all) <= half
        mo = (_emg_core(s_all[m], y_all[m], floor=0.0, flips=flips)
              if m.sum() >= 8 else None)
        if _emg_usable(mo) and inside(mo):
            chosen, win = mo, 2.0
        elif _emg_usable(wide) and inside(wide):
            chosen, win = wide, 1.0
    if chosen is None:
        return out
    out.update(window=win, ok=1.0, mu_s=chosen["mu_s"], max_s=chosen["max_s"],
               mu_err=chosen["mu_err"], lam=chosen["lam"], sd=chosen["sd"],
               r2=chosen["r2"])
    return out


def _auc(s, y):
    """Area of the hump around the peak, for a profile already at baseline 0.

    Walks outward from the peak and stops at the background OR at a valley.
    Without the valley rule a neighbouring comet 3 um away gets swallowed into
    this comet's area: on a test comet that turned 1377 into 1789 counts*um and
    a 5.0 um extent into 7.0 um.

    Returns (auc, extent, peak_height). Units follow s.
    """
    if s.size < 5 or not np.isfinite(y).any():
        return np.nan, np.nan, np.nan
    k = int(np.nanargmax(y))
    pk = float(y[k])
    if not np.isfinite(pk) or pk <= 0:
        return np.nan, np.nan, np.nan

    def walk(step):
        i = k
        run = pk
        while True:
            n = i + step
            if n < 0 or n >= y.size:
                return i
            v = y[n]
            if not np.isfinite(v) or v <= 0:
                return i
            run = min(run, v)
            if run < 0.40 * pk and v > run + 0.15 * pk:
                return i
            i = n

    i = walk(-1)
    j = walk(+1)
    if j <= i:
        return np.nan, np.nan, pk
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return (float(trapz(y[i:j + 1], s[i:j + 1])),
            float(s[j] - s[i]), pk)


def _bilinear(img, ry, rx):
    """Sample img at float (row, col), zero outside."""
    h, w = img.shape
    y0 = np.floor(ry).astype(int); x0 = np.floor(rx).astype(int)
    dy = ry - y0; dx = rx - x0
    ok = (y0 >= 0) & (x0 >= 0) & (y0 < h - 1) & (x0 < w - 1)
    out = np.zeros(ry.shape, float)
    if not ok.any():
        return out, ok
    y0c = np.clip(y0, 0, h - 2); x0c = np.clip(x0, 0, w - 2)
    a = img[y0c, x0c]; b = img[y0c, x0c + 1]
    c = img[y0c + 1, x0c]; d = img[y0c + 1, x0c + 1]
    out = (a * (1 - dy) * (1 - dx) + b * (1 - dy) * dx
           + c * dy * (1 - dx) + d * dy * dx)
    return np.where(ok, out, 0.0), ok


def _profile(img, cy, cx, theta, half_len):
    """Mean intensity along the major axis, averaged across a 3 px stripe.

    regionprops orientation: the major-axis unit vector is (cos t, sin t) in
    (row, col). The perpendicular is (-sin t, cos t).
    """
    uy, ux = np.cos(theta), np.sin(theta)
    py, px = -ux, uy
    s = np.arange(-half_len, half_len + _STEP, _STEP)
    offs = np.linspace(-_STRIPE, _STRIPE, 3)
    acc = np.zeros((len(offs), s.size))
    okall = np.zeros(s.size, bool)
    for i, o in enumerate(offs):
        ry = cy + s * uy + o * py
        rx = cx + s * ux + o * px
        v, ok = _bilinear(img, ry, rx)
        acc[i] = v
        okall |= ok
    return s, acc.mean(axis=0), okall


def _fit_gauss(s, y):
    """A*exp(-(s-m)^2/(2 sd^2)) + c. Returns (fwhm_px, r2)."""
    from scipy.optimize import curve_fit

    def f(x, A, m, sd, c):
        return A * np.exp(-((x - m) ** 2) / (2.0 * sd * sd)) + c

    c0 = float(np.percentile(y, 10))
    A0 = float(y.max() - c0)
    if A0 <= 0:
        return float("nan"), float("nan")
    m0 = float(s[int(np.argmax(y))])
    w = np.clip(y - c0, 0, None)
    sd0 = float(np.sqrt(max((w * (s - m0) ** 2).sum() / max(w.sum(), 1e-9), .25)))
    try:
        p, _ = curve_fit(f, s, y, p0=[A0, m0, sd0, c0], maxfev=2000)
    except Exception:                                   # noqa: BLE001
        return float("nan"), float("nan")
    sd = abs(p[2])
    fwhm = 2.3548200450309493 * sd
    if not np.isfinite(sd) or sd <= 0 or fwhm > _FWHM_MAX:
        return float("nan"), float("nan")
    # a peak fitted outside the sampled window is fitting something else
    if not (s[0] <= p[1] <= s[-1]):
        return float("nan"), float("nan")
    resid = y - f(s, *p)
    ss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss if ss > 0 else float("nan")
    return float(fwhm), r2


def _fit_exp(s, y):
    """Exponential decay on the slower-decaying side of the peak.

    Which side is the tail is decided from the data, not assumed, because the
    mask axis has no head/tail orientation -- theta is defined mod pi.
    """
    from scipy.optimize import curve_fit

    c0 = float(np.percentile(y, 10))
    yy = y - c0
    k = int(np.argmax(yy))
    peak = yy[k]
    if peak <= 0:
        return float("nan"), float("nan")
    best_lam, best_r2 = float("nan"), float("nan")
    for sign in (+1, -1):
        sel = (np.arange(len(s)) - k) * sign > 0
        if sel.sum() < 6:
            continue
        x = np.abs(s[sel] - s[k])
        v = yy[sel]
        order = np.argsort(x)
        x, v = x[order], v[order]

        def f(t, A, lam, c):
            return A * np.exp(-t / lam) + c

        try:
            p, _ = curve_fit(f, x, v, p0=[peak, 4.0, 0.0], maxfev=2000,
                             bounds=([0, .3, -np.inf], [np.inf, _LAM_MAX, np.inf]))
        except Exception:                               # noqa: BLE001
            continue
        # A fit that rails at the bound means "no decay found in this window",
        # not "the comet is 6 um long". Refusing it is the difference between a
        # NaN you can filter and a number that silently inflates a median.
        if p[1] >= _LAM_MAX * 0.98 or p[1] <= 0.32:
            continue
        resid = v - f(x, *p)
        ss = float(((v - v.mean()) ** 2).sum())
        r2 = 1.0 - float((resid ** 2).sum()) / ss if ss > 0 else float("nan")
        # the tail is the side that decays more slowly -- compare lambda to
        # lambda, never to r2
        if np.isfinite(p[1]) and (not np.isfinite(best_lam) or p[1] > best_lam):
            best_lam, best_r2 = float(p[1]), r2
    return best_lam, best_r2


def _fit_bieling(s, y, npix=25, step=1.0):
    """Bieling 2007 / Farmer 2021: I = A*exp(-x/lambda), 25 px behind the tip.

    Four things differ from ``_fit_exp`` and each one matters:

      tip, not peak   x is measured from the TIP -- the maximum of the profile
                      -- and the tip pixel itself is DROPPED. Their reason is
                      that sub-pixel structure at the very tip, which the
                      imaging cannot resolve, should not be allowed to set the
                      whole fit. A is the intensity at pixel 1, not at 0.
      fixed window    exactly 25 px, not "whatever is left of the profile", so
                      lambda does not depend on how long the mask happened to
                      be.
      pixel steps     1 px, as the camera samples it, not the 0.5 px the rest
                      of this module interpolates at.
      NO OFFSET       the model goes to zero. That is the risky one: with no
                      constant term, any background left over after subtraction
                      cannot be absorbed, and the fit lengthens lambda instead
                      to keep the far end up. So the same fit is also run WITH
                      an offset and both are written out -- if they disagree,
                      the background subtraction is what is being measured.

    Returns (lam_px, r2, lam_with_offset_px, r2_with_offset).
    """
    from scipy.optimize import curve_fit

    bad = (float("nan"),) * 4
    k = int(np.argmax(y))
    if not np.isfinite(y[k]) or y[k] <= 0:
        return bad

    # the tail is the side with more room and the slower fall; take the side
    # that is not the sharp rising edge
    out = []
    for sign in (+1, -1):
        xs = s[k] + sign * np.arange(step, (npix + 1) * step, step)
        if xs.min() < s[0] or xs.max() > s[-1]:
            out.append(None); continue
        v = np.interp(xs if sign > 0 else xs[::-1], s, y)
        if sign < 0:
            v = v[::-1]
        out.append(v)
    cand = [(i, v) for i, v in enumerate(out) if v is not None]
    if not cand:
        return bad
    # pick the tail: the side whose mean is higher decays more slowly
    i, v = max(cand, key=lambda t: np.nanmean(t[1]))
    x = np.arange(1, npix + 1) * step

    def f_noff(t, A, lam):
        return A * np.exp(-t / lam)

    def f_off(t, A, lam, c):
        return A * np.exp(-t / lam) + c

    def r2_of(v, pred):
        ss = float(((v - v.mean()) ** 2).sum())
        return 1.0 - float(((v - pred) ** 2).sum()) / ss if ss > 0 else float("nan")

    res = []
    for f, p0, nb in ((f_noff, [float(v[0]), 5.0], 2),
                      (f_off, [float(v[0]), 5.0, 0.0], 3)):
        try:
            p, _ = curve_fit(f, x, v, p0=p0, maxfev=4000)
            lam = float(abs(p[1]))
            if not np.isfinite(lam) or lam <= 0.3 or lam >= _LAM_MAX:
                res.extend([float("nan"), float("nan")]); continue
            res.extend([lam, r2_of(v, f(x, *p))])
        except Exception:                                   # noqa: BLE001
            res.extend([float("nan"), float("nan")])
    return tuple(res)


def _emg(x, A, L, mu, sd, c):
    """Exponentially modified Gaussian: a one-sided exponential blurred by a
    Gaussian PSF, which is what an EB comet under a microscope actually is.

    Written with erfcx (the scaled complementary error function) rather than
    exp(...) * erfc(...): those two factors blow up and vanish against each
    other far from the peak, and multiplying them directly loses the answer to
    floating point. Matches scipy.stats.exponnorm to 2e-15 and fits 60x faster
    than going through it.
    """
    from scipy.special import erfcx
    L = abs(L) + 1e-9
    sd = abs(sd) + 1e-9
    z = (sd / L - (x - mu) / sd) / 1.4142135623730951
    return A * erfcx(z) * np.exp(-((x - mu) ** 2) / (2.0 * sd * sd)) + c


_EMG_TABLE = None


def _emg_fwhm_over_sigma(K):
    """FWHM of the EMG curve, in units of sigma, as a function of K = lam/sigma.

    lambda is NOT a width. It is the decay constant of one component of the
    model, so comparing it to a Gaussian FWHM compares two different things --
    which is why it can come out longer than the FWHM on the same comet while
    the EMG plainly fits the data better.

    The honest comparison is to take the SAME statistic off the better curve:
    the full width at half maximum of the fitted EMG itself. The EMG's shape
    depends only on K, and scales with sigma, so this is a one-dimensional
    function computed once and interpolated:

        K -> 0    2.355   the Gaussian limit, as it must be
        K = 1     2.891
        K = 4     5.025
        K -> inf  0.693 K approached from above -- the exponential limit

    Measured on real comets, the resulting width tracks the Gaussian FWHM at
    r = 0.86 against 0.75 for lambda, and the cases where the two disagree are
    the ones where the Gaussian's own R^2 is poor -- i.e. where the Gaussian is
    the estimate that should be distrusted.
    """
    global _EMG_TABLE
    if _EMG_TABLE is None:
        ks = np.concatenate([np.array([1e-3]), np.geomspace(.02, 200.0, 220)])
        vals = np.empty_like(ks)
        for i, k in enumerate(ks):
            # scipy's exponnorm, not our own _emg, for the TABLE only: the
            # erfcx form is stable across the range a real profile spans but
            # not across the 200-sigma tail this table needs, where erfcx
            # overflows exactly where exp() underflows. Built once at import,
            # so its slowness costs nothing.
            from scipy.stats import exponnorm
            x = np.linspace(-9.0, 9.0 + 12.0 * k, 12001)
            y = exponnorm.pdf(x, k, loc=0.0, scale=1.0)
            y = np.where(np.isfinite(y), y, 0.0)
            j = int(np.argmax(y)); half = y[j] / 2.0
            li = np.flatnonzero(y[:j] <= half)
            ri = np.flatnonzero(y[j:] <= half)
            if not li.size or not ri.size:
                vals[i] = np.nan; continue
            a = li[-1]
            xl = np.interp(half, [y[a], y[a + 1]], [x[a], x[a + 1]])
            b = j + ri[0]
            xr = np.interp(half, [y[b], y[b - 1]], [x[b], x[b - 1]])
            vals[i] = xr - xl
        _EMG_TABLE = (ks, vals)
    ks, vals = _EMG_TABLE
    return float(np.interp(K, ks, vals))


def _fit_emg(s, y):
    """Gaussian rise, exponential decay: the two fitted as ONE model.

    An EB comet is a one-sided exponential -- sharp where tubulin is being
    added, decaying behind as the lattice matures -- seen through a microscope
    that blurs it with a roughly Gaussian PSF. The observed profile is
    therefore their CONVOLUTION, the exponentially modified Gaussian, and
    fitting that is different from fitting either piece on its own:

      _fit_gauss   symmetric, so it splits the difference between the sharp
                   side and the decaying side and belongs to neither
      _fit_exp     fits only the tail, starting at the OBSERVED peak, which is
                   still inside the blurred region -- so the PSF is counted as
                   part of the decay and lambda comes out too long

    The EMG separates them: `sigma` absorbs the PSF and `lam` is the decay
    length with the blur taken out. Checked against simulated profiles with a
    known decay length, it recovers it to 0.5% at realistic noise and
    overestimates by 15% at noise far worse than anything here.

    Returns (lam_px, sigma_px, r2). Which side decays is decided from the
    data, since the mask axis is only defined mod pi.
    """
    from scipy.optimize import curve_fit

    c0 = float(np.percentile(y, 10))
    A0 = float(y.max() - c0)
    if A0 <= 0:
        return float("nan"), float("nan"), float("nan")
    mu0 = float(s[int(np.argmax(y))])
    best = (float("nan"), float("nan"), -np.inf)

    for flip in (1.0, -1.0):
        x = s * flip
        order = np.argsort(x)
        xx, yy = x[order], y[order]
        try:
            p, _ = curve_fit(_emg, xx, yy,
                             p0=[A0, 4.0, mu0 * flip, 1.3, c0], maxfev=4000)
        except Exception:                                   # noqa: BLE001
            continue
        lam, sd = float(abs(p[1])), float(abs(p[3]))
        if not np.isfinite(lam) or lam <= 0.3 or lam >= _LAM_MAX:
            continue
        if not np.isfinite(sd) or sd <= 0.2 or sd > 8.0:
            continue
        if not (xx[0] <= p[2] <= xx[-1]):
            continue
        resid = yy - _emg(xx, *p)
        ss = float(((yy - yy.mean()) ** 2).sum())
        r2 = 1.0 - float((resid ** 2).sum()) / ss if ss > 0 else float("nan")
        if np.isfinite(r2) and r2 > best[2]:
            best = (lam, sd, r2)
    return best if np.isfinite(best[2]) else (float("nan"), float("nan"), float("nan"))


def measure(movie, labels, prob=None, config=None):
    """Per-detection photometry, aligned to ``from_label_stack``'s ordering.

    The detection list is rebuilt here with the same filters and the same
    regionprops iteration order, so index i of every returned array is index i
    of the DetectionTable. The count is asserted against ``expect_n`` when it
    is given, so a silent misalignment becomes a loud failure.
    """
    from skimage.measure import regionprops

    from .config import DetectConfig

    cfg = config or DetectConfig()
    labels = np.asarray(labels)
    movie = np.asarray(movie)
    if movie.ndim != 3:
        # nd2 gives (T, C, Y, X) for a two-channel file, so [..., 0] would slice
        # the image, not pick a channel. Choose the channel by name upstream.
        raise ValueError(f"movie must be (T, Y, X), got {movie.shape}; read it "
                         "with comet_tracker_v9.movie.read_movie (488 only)")
    if movie.shape != labels.shape:
        raise ValueError(f"movie {movie.shape} and labels {labels.shape} differ")
    if prob is not None:
        prob = np.asarray(prob)
        if prob.dtype == np.uint8:
            prob = prob.astype(np.float32) / 255.0
        elif prob.dtype == np.uint16:
            prob = prob.astype(np.float32) / 65535.0

    out = {k: [] for k in FIELDS}
    n_done = 0

    def _pad(n):
        # every field gets exactly one value per detection; anything a branch
        # below did not write is NaN. Keeps the arrays aligned however a
        # detection exits.
        for v in out.values():
            while len(v) < n:
                v.append(np.nan)

    h, w = labels.shape[1:]
    for t in range(labels.shape[0]):
        lab = labels[t]
        if not lab.any():
            continue
        img = movie[t].astype(np.float64)
        inten = prob[t] if prob is not None else None
        for p in regionprops(lab, intensity_image=inten):
            if p.area < cfg.min_area:
                continue
            score = 1.0 if inten is None else float(p.intensity_max)
            if score < cfg.min_score:
                continue

            _pad(n_done)
            n_done += 1
            r0, c0, r1, c1 = p.bbox
            padw = _PAD + int(_EXTRA)     # wide enough to hold the profile
            R0, C0 = max(0, r0 - padw), max(0, c0 - padw)
            R1, C1 = min(h, r1 + padw), min(w, c1 + padw)
            patch = img[R0:R1, C0:C1]
            lpatch = lab[R0:R1, C0:C1]
            free = lpatch == 0
            bg_patch = float(np.median(patch[free])) if free.sum() >= 20 \
                else float(np.median(patch))
            noise_patch = float(np.std(patch[free])) if free.sum() >= 20 else np.nan

            cy0 = p.centroid[0] - R0
            cx0 = p.centroid[1] - C0
            mask_vals = img[p.coords[:, 0], p.coords[:, 1]]
            sbg, sbg_n = _bg_sideways(patch, free, cy0, cx0,
                                      float(p.orientation),
                                      0.5 * float(p.axis_major_length))
            # v9: the box turned with the comet is THE background. If it is
            # too small to trust (comet at the image edge, crowded field), the
            # sideways band, then the old patch median, stand in.
            bbg, bnoise, bbg_n = _bg_box(patch, lpatch == p.label, free, cy0, cx0,
                                         float(p.orientation),
                                         0.5 * float(p.axis_major_length))
            if np.isfinite(bbg):
                bg, noise = bbg, bnoise
            elif np.isfinite(sbg):
                bg, noise = sbg, noise_patch
            else:
                bg, noise = bg_patch, noise_patch
            out["bg_patch"].append(bg_patch)
            out["bg_box_n"].append(float(bbg_n))
            out["bg_side"].append(sbg)
            out["bg_side_n"].append(float(sbg_n))
            out["bg"].append(bg)
            out["area"].append(float(len(p.coords)))
            out["peak"].append(float(mask_vals.max() - bg))
            out["integrated"].append(float((mask_vals - bg).sum()))
            out["mean"].append(float(mask_vals.mean() - bg))
            # the same three above the sideways band. Exact, not approximate --
            # subtracting a different constant is all that changes -- but formed
            # here, per detection, because the per-track median of a difference
            # is not the difference of the per-track medians.
            if np.isfinite(sbg):
                out["peak_side"].append(float(mask_vals.max() - sbg))
                out["integrated_side"].append(float((mask_vals - sbg).sum()))
                out["mean_side"].append(float(mask_vals.mean() - sbg))
            else:
                out["peak_side"].append(np.nan)
                out["integrated_side"].append(np.nan)
                out["mean_side"].append(np.nan)

            # Sample the profile from a patch in which every OTHER comet has
            # been replaced by background. Without this a neighbour a few
            # pixels away is read as part of this comet's tail, and the
            # exponential fit happily reports a very long lambda.
            clean = patch.copy()
            other = (lpatch != 0) & (lpatch != p.label)
            if other.any():
                clean[other] = bg

            cy, cx = p.centroid
            # ONE long profile is sampled, but the fits do not all get the same
            # window. Bieling needs a full 25 px behind the tip; the others are
            # HURT by that much extra background -- measured on real data,
            # widening their window from 10 to 28 px inflated the exponential
            # tail from 0.92 to 1.14 um and dropped the Gaussian's convergence
            # from 78% to 63%. So each fit gets the window it needs.
            half_long = 0.5 * float(p.axis_major_length) + _EXTRA
            half_short = 0.5 * float(p.axis_major_length) + _SHORT_EXTRA
            s, prof, ok = _profile(clean, cy - R0, cx - C0,
                                   float(p.orientation), half_long)
            prof = prof - bg
            near = ok & (np.abs(s) <= half_short)
            if ok.sum() < 10 or near.sum() < 10:
                for k in ("auc", "auc_extent_px", "auc_side",
                          "auc_extent_side_px", "peak_prof",
                          "peak_prof_side", "biel_len_side_px", "biel_side_r2",
                          "gauss_fwhm_px", "gauss_r2", "exp_lambda_px", "exp_r2",
                          "emg_len_px", "emg_sigma_px", "emg_fwhm_px", "emg_r2", "biel_len_px",
                          "biel_r2", "biel_len_off_px", "biel_off_r2",
                          "profile_snr"):
                    out[k].append(np.nan)
                continue
            g_fwhm, g_r2 = _fit_gauss(s[near], prof[near])
            e_lam, e_r2 = _fit_exp(s[near], prof[near])
            # one wide EMG fit serves both the length fields and the positions
            wide_each = _emg_each(s[near], prof[near])
            _cands = [r for r in wide_each.values() if r is not None]
            wide = max(_cands, key=lambda r: r["r2"]) if _cands else None
            m_lam, m_sd, m_r2 = ((wide["lam"], wide["sd"], wide["r2"]) if wide
                                 else (np.nan, np.nan, np.nan))
            _uy, _ux = np.cos(float(p.orientation)), np.sin(float(p.orientation))
            _gy, _gx = p.centroid                      # image coordinates
            for d, sgn in (("p", 1.0), ("n", -1.0)):
                ep = _emg_positions(s[near], prof[near], s[ok], prof[ok],
                                    0.5 * float(p.axis_major_length), plus_dir=sgn,
                                    wide=wide_each[-sgn])
                out[f"emg_window_{d}"].append(ep["window"])
                out[f"emg_ok_{d}"].append(ep["ok"])
                out[f"emg_mu_y_{d}"].append(_gy + ep["mu_s"] * _uy)
                out[f"emg_mu_x_{d}"].append(_gx + ep["mu_s"] * _ux)
                out[f"emg_max_y_{d}"].append(_gy + ep["max_s"] * _uy)
                out[f"emg_max_x_{d}"].append(_gx + ep["max_s"] * _ux)
                out[f"emg_mu_err_px_{d}"].append(ep["mu_err"])
                out[f"emg_len_px_{d}"].append(ep["lam"])
                out[f"emg_sigma_px_{d}"].append(ep["sd"])
                out[f"emg_r2_{d}"].append(ep["r2"])
            out["gauss_fwhm_px"].append(g_fwhm); out["gauss_r2"].append(g_r2)
            out["exp_lambda_px"].append(e_lam); out["exp_r2"].append(e_r2)
            b_lam, b_r2, b_lam_o, b_r2_o = _fit_bieling(s[ok], prof[ok])
            out["emg_len_px"].append(m_lam); out["emg_sigma_px"].append(m_sd)
            out["emg_fwhm_px"].append(
                m_sd * _emg_fwhm_over_sigma(m_lam / m_sd)
                if np.isfinite(m_lam) and np.isfinite(m_sd) and m_sd > 0
                else np.nan)
            out["emg_r2"].append(m_r2)
            out["biel_len_px"].append(b_lam); out["biel_r2"].append(b_r2)
            out["biel_len_off_px"].append(b_lam_o); out["biel_off_r2"].append(b_r2_o)
            a_auc, a_ext, a_pk = _auc(s[ok], prof[ok])
            out["auc"].append(a_auc)
            out["auc_extent_px"].append(a_ext)
            out["peak_prof"].append(a_pk)
            # The same area again, above the sideways band instead. prof is
            # already whole-patch-subtracted, so shifting the baseline by the
            # difference between the two backgrounds is the whole change -- and
            # it moves the STOP POINTS too, which is the reason this cannot be
            # recovered afterwards from auc and the two background values.
            if np.isfinite(sbg):
                s_auc, s_ext, _ = _auc(s[ok], prof[ok] - (sbg - bg))
            else:
                s_auc = s_ext = np.nan
            out["auc_side"].append(s_auc)
            out["auc_extent_side_px"].append(s_ext)
            out["peak_prof_side"].append(a_pk + (bg - sbg)
                                         if np.isfinite(sbg) else np.nan)
            # the no-offset Bieling fit is the ONE length that moves with the
            # background, so it is the one that genuinely has to be run twice
            if np.isfinite(sbg):
                sb_lam, sb_r2, _, _ = _fit_bieling(s[ok], prof[ok] - (sbg - bg))
            else:
                sb_lam = sb_r2 = np.nan
            out["biel_len_side_px"].append(sb_lam)
            out["biel_side_r2"].append(sb_r2)
            out["profile_snr"].append(float(prof[ok].max() / noise)
                                      if noise and np.isfinite(noise) and noise > 0
                                      else np.nan)
    _pad(n_done)
    return {k: np.asarray(v, float) for k, v in out.items()}
