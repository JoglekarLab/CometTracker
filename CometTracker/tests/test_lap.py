import numpy as np
import pytest

from comet_tracker_v7.lap import solve


def test_obvious_pairing():
    cost = np.array([[1.0, 50.0], [50.0, 1.0]])
    a = solve(cost, birth_cost=np.full(2, 100.0), death_cost=np.full(2, 100.0))
    assert sorted(map(tuple, a.matches)) == [(0, 0), (1, 1)]
    assert len(a.dead) == 0 and len(a.born) == 0


def test_expensive_link_loses_to_birth_and_death():
    """The behaviour a hard gate cannot express."""
    cost = np.array([[30.0]])
    a = solve(cost, birth_cost=np.array([5.0]), death_cost=np.array([5.0]))
    assert len(a.matches) == 0
    assert a.dead.tolist() == [0] and a.born.tolist() == [0]


def test_cheap_link_beats_birth_plus_death():
    cost = np.array([[3.0]])
    a = solve(cost, birth_cost=np.array([5.0]), death_cost=np.array([5.0]))
    assert a.matches.tolist() == [[0, 0]]


def test_weak_detection_links_but_does_not_birth():
    """THE design property. Detection 0 is weak (expensive to birth) and sits
    where track 0 predicts; detection 1 is weak and sits alone."""
    link = np.array([[0.5, np.inf]])          # only det 0 is reachable
    birth = np.array([90.0, 90.0])            # both weak -> costly to birth
    a = solve(link, birth_cost=birth, death_cost=np.array([90.0]))
    assert a.matches.tolist() == [[0, 0]], "weak but predicted -> should link"
    assert a.born.tolist() == [1], "weak and alone -> should birth (or be dropped later)"


def test_fully_gated_row_dies_rather_than_erroring():
    link = np.array([[np.inf, np.inf]])
    a = solve(link, birth_cost=np.full(2, 4.0), death_cost=np.array([4.0]))
    assert a.dead.tolist() == [0]
    assert a.born.tolist() == [0, 1]
    assert len(a.matches) == 0


def test_blocked_entries_are_never_chosen():
    """A gated pair must not appear even when it is the only column left."""
    link = np.array([[np.inf, 1.0], [np.inf, 2.0]])
    a = solve(link, birth_cost=np.full(2, 100.0), death_cost=np.full(2, 100.0))
    for _, d in a.matches:
        assert d == 1
    assert len(a.matches) == 1


def test_empty_inputs():
    a = solve(np.zeros((0, 3)), np.full(3, 1.0), np.zeros(0))
    assert a.born.tolist() == [0, 1, 2] and len(a.matches) == 0
    b = solve(np.zeros((2, 0)), np.zeros(0), np.full(2, 1.0))
    assert b.dead.tolist() == [0, 1] and len(b.matches) == 0


def test_every_track_and_detection_is_accounted_for():
    rng = np.random.default_rng(0)
    for _ in range(30):
        n, m = rng.integers(1, 7), rng.integers(1, 7)
        link = rng.uniform(0, 20, (n, m))
        link[rng.random((n, m)) < 0.3] = np.inf
        a = solve(link, rng.uniform(5, 15, m), rng.uniform(5, 15, n))
        tracks = set(a.matches[:, 0].tolist()) | set(a.dead.tolist())
        dets = set(a.matches[:, 1].tolist()) | set(a.born.tolist())
        assert tracks == set(range(n)), "a track was neither linked nor died"
        assert dets == set(range(m)), "a detection was neither linked nor born"
        assert len(a.matches[:, 0]) == len(set(a.matches[:, 0].tolist()))
        assert len(a.matches[:, 1]) == len(set(a.matches[:, 1].tolist()))


def test_beats_the_v1_sentinel_approach_on_a_two_candidate_case():
    """V1 wrote 1e9 into gated cells and filtered afterwards. Here track 0 can
    reach det 0 (cost 20, poor) and nothing else; det 1 is unreachable. With
    birth+death cheaper than the poor link, the right answer is to refuse it."""
    link = np.array([[20.0, np.inf]])
    a = solve(link, birth_cost=np.array([6.0, 6.0]), death_cost=np.array([6.0]))
    assert len(a.matches) == 0, "a poor link should lose to birth+death"
    assert set(a.born.tolist()) == {0, 1}
