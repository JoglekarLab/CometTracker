# CometTracker V7 — design

A plusTipTracker-shaped tracker driven by SAM3 masks. Written from scratch; it
shares no code with V1–V6. What it inherits from them is their **measurements**,
recorded beside the parameters they set.

---

## The one idea

A microtubule grows along a line. A SAM3 mask is a *rendered centreline* — the
training target is `soft_uniform_axis`, a 3 px-wide soft tube around the
annotated axis, "deliberately no taper" (`comet_sam3/targets.py`). So the mask's
principal axis is not an ellipse fitted to a smear of light. It is the growth
axis, drawn.

Measured on real SAM3 predictions, 247 straight tracks across both pAJV103
movies:

| | median | within 15° | within 30° |
|---|---|---|---|
| mask axis vs net direction of travel | **5.4°** | 83.4% | 96.8% |
| mask axis vs that frame's step | 6.8° | 74.4% | 90.6% |

and the across-axis component of a single-frame step:

| |q| | median | p90 | p99 |
|---|---|---|---|---|
| px | 0.17 | 0.52 | 2.16 |

Everything below is built to spend that.

---

## What V7 tracks, and what it refuses to track

**Position is the mask centroid.** Not the head. SAM3's head branch emits 1.37
heads per comet, up to 8, scattered at a median 0.47× the mask's own length —
there is no reliable plus-end in it. Three candidates were measured by RMS
residual around a constant-velocity fit on 247 real tracks:

| estimator | RMS residual |
|---|---|
| **mask centroid** | **0.61 px** |
| bounding-box centre | 0.68 px |
| centroid + half-length along axis | 2.61 px |

The bounding-box centre was the original proposal. It loses because **90% of its
frame-to-frame motion is explained by the mask changing length**
(corr(\|bbox step\|, \|Δ major axis\|) = 0.903 and 0.880 on the two movies) while
the head is uncorrelated with it (0.024, 0.050). Reconstructing a tip is worse
still: it inherits the full end-jitter that the centroid averages down.

**The cost of that choice is stated, not hidden.** Mask length drifts a median
6.8 px over a track, so centroid displacement understates plus-end displacement
by roughly half of that. It is reported per track as `length_drift_px` and not
corrected, because correcting it needs a plus-end estimator V7 does not have.

**Mask length is never a state or a measurement.** It is partly a readout of
detector confidence — across presence bins the minor axis is flat (4.79 → 5.06
px) while the major nearly doubles (9.36 → 17.56 px). V1 filtered tail
half-length as a Kalman state; on these masks that state would mostly track the
detector's mood.

---

## Stage 1 — frame to frame

Constant-velocity Kalman filter on the centroid, LAP assignment with birth and
death blocks, three gates.

**The corridor gate** is the point. `|q| ≤ corridor_min + mult · |step| · σ_θ`,
where `q` is the step's across-axis component measured against the *candidate
detection's own mask axis*.

Three properties:

- **It exists at birth.** The axis is shape, not motion, so it gates a track's
  first link — 22.8% of all links by V6's count, of which only 3.2% are correct,
  and unreachable by any motion-derived gate. This is also why V7 does not need
  u-track's forward/backward/forward pass scheme, whose entire purpose is to
  bootstrap the missing initial direction.
- **It is independent evidence.** A motion axis is bent by the mis-links the
  gate exists to catch. Shape cannot be.
- **σ_θ makes it self-aware.** A round mask has a huge orientation uncertainty
  and gets a wide corridor rather than being silently trusted.

Benchmarked (3 seeds, 120 comets × 30 frames, measured clutter statistics):

| corridor | spurious | purity | comets recovered |
|---|---|---|---|
| off | 26.4% | 0.763 | 75.0 |
| 4.92 (V6's) | 25.4% | 0.769 | 75.0 |
| **2.5 (V7)** | **21.0%** | **0.810** | **77.7** |
| 1.5 | 19.1% | 0.821 | 78.0 |

Precision improves and recall does *not* fall — the corridor is close to free.
And it reproduces V6's finding about what it is *for*: it lets `max_disp` stay
loose without paying for it.

| max_disp | spurious, corridor on | corridor off |
|---|---|---|
| 5 | 19.3% | 20.6% |
| 7 | 21.0% | 26.4% |
| 9 | 22.1% | 30.6% |
| 12 | 23.1% | 35.3% |

Inert at 5, doing most of the work at 12 — exactly V6's table shape.

**No coasting** (`max_gap = 0`), V6's single largest measured effect. Fragments
are gap closing's problem.

**One pass.** V6 measured 1/3/5 passes as equivalent; V7 measures 3 passes as
actively *worse* (22.9% spurious and 68.7 comets vs 21.0% and 77.7). Caveat:
V7's multi-pass is a simplification of u-track's — it re-seeds initial
velocities between passes rather than carrying full Kalman state — so this
result may be limited by the implementation rather than the idea.

---

## Stage 2 — gap closing

One global LAP over every track end against every track start within
`time_window`, with each accepted join labelled:

- **fgap** — the start is ahead, same direction. Paused, then resumed.
- **bgap** — the start is behind, back down the line. Depolymerised. A
  catastrophe.

The backward wedge is 10° against the forward cone's 30°. That is physics, not
tuning: a growing tip can wander, a shrinking one retreats along the lattice it
just built and cannot deviate from it.

Measured against ground truth, fragments per comet: **1.49 → 1.27**, with the
welded-track count unchanged at 1.7. It joins without fusing.

One parameter plusTip does not have: `fwd_vel_mult`. V7 measures forward reach
from the segment's last position using its own median speed, and a gap caused by
a *detector miss* rather than a pause puts the comet at almost exactly
`speed·dt` ahead — right on the boundary. Diagnosed on real data: of 8,333
direction-agreeing candidate pairs, only 288 survived a reach of 1.0×.

---

## Stage 3 — motion classification

MSS (`trackMSSAnalysis.m`): immobile / confined / Brownian / directed, with a
straightness fallback for tracks too short for the fit, always labelled with
which method produced it.

This is not cosmetic. On real data it is what makes the growth rate come out
right:

| | movie 010 | movie 015 |
|---|---|---|
| all tracks | 8.96 µm/min | 7.55 µm/min |
| **directed only** | **11.35 µm/min** | **10.10 µm/min** |
| expected from HANDOFF's 2.36 px/frame | 11.25 µm/min | 11.25 µm/min |

385 and 688 tracks classify as immobile. That population is what was dragging
every velocity number in this project downward, and it comes out without needing
a single label.

---

## What is measured, what is guessed, what is broken

**Measured and working:** the corridor gate, the centroid-over-bbox choice, no
coasting, one pass, gap closing's fragment reduction, MSS separating the junk.

**Measured and *not* working — presence.** The design called for presence to be
a cost rather than a threshold, on the argument that a weak detection where a
strong track predicts should link while the same detection alone should not.
The structure is right and the input is not: on real data a detection inside a
straight moving track scores 0.922 against 0.898 inside a track that goes
nowhere — **AUC 0.608**, where 0.5 is no information. Benchmarked, presence as a
cost (20.9% spurious), presence ignored (20.3%) and a hard threshold at 0.7
(20.3%) are indistinguishable, and `birth_score_weight` swept over 0–120 moves
nothing. The parameters are kept and labelled, not claimed.

An earlier version of the benchmark *did* show a hard threshold cutting spurious
tracks from 21% to 7%. That was an artifact of a guessed clutter-score
distribution that real data does not have. The guess is now replaced by the
measurement.

**Still guesses:** `process_noise`, all four cost weights, `fwd_vel_mult`,
`min_frames_for_mss`, and the MSS class thresholds (u-track compares against
simulated Brownian confidence intervals; V7 uses fixed cuts).

**Not yet verified at all:** nothing here has been scored against hand-labelled
real tracks. The benchmark's clutter density (140/frame) and miss rate (25%) are
V6's *estimates*, and V6 flagged them as the parameters its results were most
sensitive to. That warning carries over unchanged.

**The known upstream defect.** `sam3_export.py` writes
`P = max_q presence(q) · P_axis(q, pixel)` and thresholds *that* at 0.5, so the
mask contour is a joint iso-contour: a query with presence ≤ 0.5 can never
produce a mask however good its axis, which silently discards ~31% of SAM3's
detections. Fixing it means thresholding `P_axis` alone and carrying `presence`
as a column — which is exactly the interface `detections.py` already takes, so
the fix is upstream-only.
