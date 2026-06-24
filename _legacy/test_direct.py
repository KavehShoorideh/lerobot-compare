"""
baseline test suite. Statistical tests use fixed seeds, enough replicates, and a margin
(alpha + a few binomial SEs).

  Baseline  validity: false-positive rate <= alpha at the stopping time, under H0,
      for paired, unpaired, and covariate-matched data
  Enhanced  power: a real gap is detected (and the direction is correct)
  V3  matching ladder: exact > covariate > unpaired routing works; leftovers go
      to unpaired (never dropped); matching never peeks at labels
  V4  structural: determinism, alpha split, report integrity
"""

import math
import numpy as np
import pytest

from lerobot_doctor import compare_success, match, Trial

ALPHA = 0.10


def _se(p, n):
    return math.sqrt(p * (1 - p) / n)


def _bern(p, n, rng):
    return [int(rng.random() < p) for _ in range(n)]


# ---- Baseline validity ----------------------------------------------------------

def test_V1_type_i_unpaired():
    T = 400
    dec = 0
    for t in range(T):
        rng = np.random.default_rng(t)
        A, B = _bern(0.6, 300, rng), _bern(0.6, 300, rng)
        r = compare_success(A, B, alpha=ALPHA, rng=np.random.default_rng(9000 + t))
        dec += int(r.decided)
    assert dec / T <= ALPHA + 3 * _se(ALPHA, T)


def test_V1_type_i_paired_exact():
    """Paired same-instance data under H0 also controls type-I."""
    T = 400
    dec = 0
    for t in range(T):
        rng = np.random.default_rng(100 + t)
        A, B = [], []
        for k in range(300):
            A.append(Trial(int(rng.random() < 0.6), key=k))
            B.append(Trial(int(rng.random() < 0.6), key=k))
        r = compare_success(A, B, alpha=ALPHA, rng=np.random.default_rng(9100 + t))
        dec += int(r.decided)
    assert dec / T <= ALPHA + 3 * _se(ALPHA, T)


def test_V1_type_i_one_sided():
    T = 400
    dec = 0
    for t in range(T):
        rng = np.random.default_rng(200 + t)
        A, B = _bern(0.6, 300, rng), _bern(0.6, 300, rng)
        r = compare_success(A, B, alpha=ALPHA, alternative="A>B",
                            rng=np.random.default_rng(9200 + t))
        dec += int(r.decided and r.direction == "A>B")
    assert dec / T <= ALPHA + 3 * _se(ALPHA, T)


# ---- Enhanced power -------------------------------------------------------------

def test_V2_detects_true_gap():
    T = 200
    dec = 0
    for t in range(T):
        rng = np.random.default_rng(300 + t)
        A, B = _bern(0.75, 600, rng), _bern(0.55, 600, rng)
        r = compare_success(A, B, alpha=ALPHA, rng=np.random.default_rng(9300 + t))
        dec += int(r.decided and r.direction == "A>B")
    assert dec / T >= 0.9


def test_V2_paired_decides_faster_than_unpaired():
    """Correlated same-instance pairs reduce variance -> earlier stop on average."""
    paired_n, unpaired_n = [], []
    for t in range(120):
        rng = np.random.default_rng(400 + t)
        A_t, B_t, A_u, B_u = [], [], [], []
        for k in range(800):
            base = rng.random()                      # shared difficulty
            ya = int(base < 0.75); yb = int(base < 0.55)   # positively correlated
            A_t.append(Trial(ya, key=k)); B_t.append(Trial(yb, key=k))
            A_u.append(ya); B_u.append(yb)
        rp = compare_success(A_t, B_t, alpha=ALPHA, rng=np.random.default_rng(1))
        ru = compare_success(A_u, B_u, alpha=ALPHA, rng=np.random.default_rng(1))
        if rp.decided:
            paired_n.append(rp.n_to_decision)
        if ru.decided:
            unpaired_n.append(ru.n_to_decision)
    assert np.median(paired_n) <= np.median(unpaired_n)


# ---- V3 matching ----------------------------------------------------------

def test_V3_exact_match_used():
    A = [Trial(1, key=i) for i in range(50)]
    B = [Trial(0, key=i) for i in range(50)]
    p = match(A, B)
    assert p.rung_counts["exact"] == 50
    assert p.rung_counts["unpaired"] == 0


def test_V3_covariate_match_used_and_improves_balance():
    rng = np.random.default_rng(7)
    A = [Trial(1, covariates=np.array([x, 0.0]))
         for x in rng.normal(0.5, 1.0, 60)]
    B = [Trial(0, covariates=np.array([x, 0.0]))
         for x in rng.normal(-0.5, 1.0, 60)]            # shifted -> imbalance
    p = match(A, B, caliper=5.0)
    assert p.rung_counts["covariate"] > 0
    assert p.covariate_balance_after <= p.covariate_balance_before + 1e-9


def test_V3_leftovers_routed_to_unpaired_not_dropped():
    # 30 A's share keys with only 10 B's; the other 20 A's must still be used
    A = [Trial(1, key=i) for i in range(10)] + [Trial(1) for _ in range(20)]
    B = [Trial(0, key=i) for i in range(10)] + [Trial(0) for _ in range(20)]
    p = match(A, B)
    assert p.rung_counts["exact"] == 10
    assert p.rung_counts["unpaired"] == 20
    assert p.n_pairs == 30                              # nothing dropped


def test_V3_matching_ignores_labels():
    """Matching must use keys/covariates only -- flipping success labels must not
    change which pairs are formed (only their values)."""
    A1 = [Trial(1, key=i) for i in range(20)]
    B1 = [Trial(0, key=i) for i in range(20)]
    A2 = [Trial(0, key=i) for i in range(20)]           # labels flipped
    B2 = [Trial(1, key=i) for i in range(20)]
    p1 = match(A1, B1, rng=np.random.default_rng(0))
    p2 = match(A2, B2, rng=np.random.default_rng(0))
    assert p1.rung_counts == p2.rung_counts


# ---- V4 structural --------------------------------------------------------

def test_V4_determinism():
    rng_data = np.random.default_rng(11)
    A, B = _bern(0.7, 400, rng_data), _bern(0.55, 400, rng_data)
    r1 = compare_success(A, B, alpha=ALPHA, rng=np.random.default_rng(5))
    r2 = compare_success(A, B, alpha=ALPHA, rng=np.random.default_rng(5))
    assert (r1.decided, r1.direction, r1.n_to_decision) == \
           (r2.decided, r2.direction, r2.n_to_decision)


def test_V4_report_integrity():
    rng = np.random.default_rng(12)
    A, B = _bern(0.8, 300, rng), _bern(0.5, 300, rng)
    r = compare_success(A, B, alpha=ALPHA)
    d = r.to_dict()
    assert d["n_pairs"] == sum(r.rung_counts.values())
    assert 0.0 <= r.rate_A <= 1.0 and 0.0 <= r.rate_B <= 1.0
    assert abs(r.p_value_equiv - 1.0 / r.e_value) < 1e-9
    assert len(r.wealth_AgtB) == r.n_pairs
