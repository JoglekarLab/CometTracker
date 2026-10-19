"""Linear assignment with birth and death priced in, as in Jaqaman et al. 2008.

WHY NOT JUST HUNGARIAN ON THE COST MATRIX

V1 built an (n_tracks x n_dets) distance matrix, wrote 1e9 into every gated
cell, ran ``linear_sum_assignment``, and threw away results >= 1e9 afterwards.
That works, but it makes "do not link" a HARD GATE rather than a priced
alternative, and the difference shows up exactly where it matters: when a track
has two mediocre candidates, or when a detection is weak. A hard gate has no way
to say "this link is allowed but starting a new track would be cheaper".

Jaqaman's formulation makes not-linking a real option by solving a square
augmented problem::

         m detections      n tracks
    n [   link costs   |  death (diag)  ]
    m [  birth (diag)  |   auxiliary    ]

Every track either links to a detection or takes its own death entry; every
detection either links or takes its own birth entry. Because births and deaths
are priced, a weak detection can be cheap to LINK (its cost is dominated by how
well it fits a prediction) while being expensive to BIRTH. That asymmetry is the
whole reason V7 can drop the presence threshold that currently discards ~31% of
SAM3's detections: presence stops being a gate and becomes a cost.

The bottom-right auxiliary block is the transpose of the link block. It carries
no independent meaning; it exists so the problem is square and so an accepted
link consumes its mirrored entry rather than leaving a cheaper phantom.

BLOCKED ENTRIES are a large finite number, not inf. ``BIG`` is set above
``(n + m) * max_finite_cost``, so one blocked cell costs more than an entire
otherwise-valid assignment and can never be chosen unless nothing else exists.
Finite values keep scipy's solver on its well-tested path.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = ["Assignment", "solve"]


class Assignment:
    """Result of one LAP solve.

    matches : (k, 2) int array of (track_index, detection_index)
    dead    : int array of track indices that took their death entry
    born    : int array of detection indices that took their birth entry
    cost    : total cost of the real (non-auxiliary) part of the assignment
    """

    __slots__ = ("matches", "dead", "born", "cost")

    def __init__(self, matches, dead, born, cost):
        self.matches = matches
        self.dead = dead
        self.born = born
        self.cost = float(cost)

    def __repr__(self) -> str:
        return (f"Assignment({len(self.matches)} linked, {len(self.dead)} died, "
                f"{len(self.born)} born, cost={self.cost:.2f})")


def solve(link_cost: np.ndarray,
          birth_cost: np.ndarray,
          death_cost: np.ndarray) -> Assignment:
    """Solve the augmented assignment problem.

    Parameters
    ----------
    link_cost : (n_tracks, n_dets). Use ``np.inf`` for a gated pair.
    birth_cost : (n_dets,) cost of detection j starting a new track.
    death_cost : (n_tracks,) cost of track i ending here.

    Both birth and death costs must be finite; if they were not, a fully gated
    row or column would make the problem infeasible.
    """
    link_cost = np.asarray(link_cost, dtype=np.float64)
    if link_cost.ndim != 2:
        raise ValueError("link_cost must be 2-D")
    n, m = link_cost.shape
    birth_cost = np.asarray(birth_cost, dtype=np.float64).reshape(-1)
    death_cost = np.asarray(death_cost, dtype=np.float64).reshape(-1)
    if birth_cost.size != m or death_cost.size != n:
        raise ValueError("birth_cost must be (n_dets,) and death_cost (n_tracks,)")
    if not (np.isfinite(birth_cost).all() and np.isfinite(death_cost).all()):
        raise ValueError("birth and death costs must be finite")

    if n == 0:
        return Assignment(np.zeros((0, 2), int), np.zeros(0, int),
                          np.arange(m), float(birth_cost.sum()))
    if m == 0:
        return Assignment(np.zeros((0, 2), int), np.arange(n),
                          np.zeros(0, int), float(death_cost.sum()))

    allowed = np.isfinite(link_cost)
    finite_vals = np.concatenate([link_cost[allowed].ravel(), birth_cost, death_cost])
    max_finite = float(finite_vals.max()) if finite_vals.size else 1.0
    big = (n + m + 1) * (abs(max_finite) + 1.0) * 10.0

    size = n + m
    C = np.full((size, size), big, dtype=np.float64)
    C[:n, :m] = np.where(allowed, link_cost, big)
    # death block: track i may only take its OWN death entry
    C[:n, m:] = big
    C[np.arange(n), m + np.arange(n)] = death_cost
    # birth block: detection j may only take its OWN birth entry
    C[n:, :m] = big
    C[n + np.arange(m), np.arange(m)] = birth_cost
    # auxiliary block, mirroring the link block
    C[n:, m:] = np.where(allowed.T, link_cost.T, big)

    rows, cols = linear_sum_assignment(C)

    matches, dead, born, total = [], [], [], 0.0
    for r, c in zip(rows, cols):
        if r < n and c < m:
            if allowed[r, c]:
                matches.append((int(r), int(c)))
                total += float(link_cost[r, c])
        elif r < n and c >= m:
            if c - m == r:
                dead.append(int(r))
                total += float(death_cost[r])
        elif r >= n and c < m:
            if r - n == c:
                born.append(int(c))
                total += float(birth_cost[c])
    return Assignment(
        np.asarray(matches, int).reshape(-1, 2),
        np.asarray(sorted(dead), int),
        np.asarray(sorted(born), int),
        total,
    )
