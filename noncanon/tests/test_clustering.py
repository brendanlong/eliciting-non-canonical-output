"""Clustering statistics on synthetic rollouts."""

import numpy as np

from noncanon.clustering import gap_test, hazard, poisson_multi


def rollout(n_tokens, positions):
    return {"prompt_id": "p", "sample": 0, "n_tokens": n_tokens, "n_units": n_tokens, "nc_events": len(positions), "event_positions": sorted(positions)}


def test_poisson_expectation_matches_a_single_flagged_rollout():
    rows = [rollout(1000, [10, 20]), rollout(1000, [])]  # rate 0.001/token → λ = 1 for the flagged rollout
    n1, n2, expected = poisson_multi(rows)
    assert (n1, n2) == (1, 1) and abs(expected - (1 - np.exp(-1) / (1 - np.exp(-1)))) < 1e-9


def test_gap_test_detects_tight_clusters_and_not_uniform_spread():
    rng = np.random.default_rng(0)
    tight = [rollout(10000, [t, t + 3, t + 6]) for t in rng.integers(0, 9000, 40)]
    obs, shuf, p, n = gap_test(tight, np.random.default_rng(1), B=200)
    assert obs == 3 and shuf > 500 and p < 0.01 and n == 80
    spread = [rollout(10000, sorted(rng.choice(10000, 3, replace=False))) for _ in range(40)]
    obs, shuf, p, n = gap_test(spread, np.random.default_rng(1), B=200)
    assert p > 0.05


def test_hazard_compares_to_depth_matched_windows():
    rows = [rollout(1000, [100, 130]), rollout(1000, [100]), rollout(1000, [500]), rollout(1000, [])]
    obs, base, n = hazard(rows, 64, np.random.default_rng(0))
    # events at 100 (twice), 130, 500: only the first rollout's event at 100 has a follower within 64 tokens
    assert n == 4 and abs(obs - 0.25) < 1e-9
    assert 0 <= base < obs
