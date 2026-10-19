"""CometTracker V7 - a plusTipTracker-shaped tracker driven by SAM3 masks.

V7 is written from scratch; it shares no code with V1-V6. What it inherits is
their *measurements*, which are recorded next to the parameters they set (see
``config.py``).

The one structural difference from every earlier version: V7 never uses a
detected head. SAM3's head branch produces 1.37 heads per comet (up to 8),
scattered along the tube at a median 0.47x the mask's own length, so a head is
not a reliable plus-end. V7 tracks the mask CENTROID and takes direction from
the mask's principal axis, which was measured on real predictions to sit
5.4 degrees (median) from the direction of travel.
"""

__version__ = "0.1.0"
