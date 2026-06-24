"""
matching.py -- build the best-available paired stream for the A/B test.

The comparison engine bets on per-pair differences g = y_A - y_B. How we pair A
and B rollouts changes the VARIANCE of those differences (how fast we decide),
not the VALIDITY of the test. So we always pair as well as the data allows and
report which rung each pair came from, so the strength of the claim is visible.

Rungs, best to worst:
  exact      same instance: identical match key (seed / scene id / init-condition).
             A clean paired test; matching is pure efficiency here.
  covariate  no shared key, but pre-outcome covariates exist: nearest-neighbour
             match via optimal assignment within a caliper. Best-EFFORT, not a
             randomized control -- the matched-difference estimand equals the true
             p_A - p_B only under an ignorability/overlap assumption.
  unpaired   no usable key/covariate: leftovers are randomly paired. Still valid
             (independent samples -> E[y_A - y_B] = p_A - p_B for ANY pairing),
             just higher variance.

Matching uses ONLY pre-outcome information (keys/covariates), never the success
labels -- required for the anytime-valid guarantee to hold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, List, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    _HAVE_SCIPY = True
except Exception:                                   # pragma: no cover
    _HAVE_SCIPY = False


@dataclass
class Trial:
    """One rollout outcome plus whatever lets us match it to a control."""
    success: int                                    # 0/1 task success
    key: Optional[Hashable] = None                  # exact-match key (seed/scene)
    covariates: Optional[np.ndarray] = None         # pre-outcome covariate vector
    meta: dict = field(default_factory=dict)


@dataclass
class Pairing:
    """Result of matching: an ordered list of (y_A, y_B, rung) plus diagnostics."""
    pairs: List[Tuple[int, int, str]]               # (idx_A, idx_B, rung)
    rung_counts: dict                               # rung -> n pairs
    n_unmatched_A: int
    n_unmatched_B: int
    covariate_balance_before: Optional[float] = None
    covariate_balance_after: Optional[float] = None

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)


def _std_mean_diff(XA: np.ndarray, XB: np.ndarray) -> float:
    """Mean absolute standardized mean difference across covariates (a love-plot
    summary): 0 = perfectly balanced."""
    if XA.size == 0 or XB.size == 0:
        return 0.0
    mu_a, mu_b = XA.mean(0), XB.mean(0)
    sd = np.sqrt(0.5 * (XA.var(0) + XB.var(0))) + 1e-12
    return float(np.mean(np.abs(mu_a - mu_b) / sd))


def match(trials_A: List[Trial], trials_B: List[Trial], *,
          caliper: float = 1.0, rng: Optional[np.random.Generator] = None) -> Pairing:
    """Pair A and B trials with the best available method, rung by rung."""
    rng = rng or np.random.default_rng(0)
    pairs: List[Tuple[int, int, str]] = []
    rung_counts = {"exact": 0, "covariate": 0, "unpaired": 0}

    used_A = [False] * len(trials_A)
    used_B = [False] * len(trials_B)

    # ---- rung 1: exact key match ------------------------------------------
    by_key_B: dict = {}
    for j, b in enumerate(trials_B):
        if b.key is not None:
            by_key_B.setdefault(b.key, []).append(j)
    for i, a in enumerate(trials_A):
        if a.key is None:
            continue
        bucket = by_key_B.get(a.key)
        if bucket:
            j = bucket.pop()
            used_A[i] = used_B[j] = True
            pairs.append((i, j, "exact"))
            rung_counts["exact"] += 1

    rem_A = [i for i in range(len(trials_A)) if not used_A[i]]
    rem_B = [j for j in range(len(trials_B)) if not used_B[j]]

    bal_before = bal_after = None

    # ---- rung 2: covariate (optimal assignment within caliper) ------------
    covA = [i for i in rem_A if trials_A[i].covariates is not None]
    covB = [j for j in rem_B if trials_B[j].covariates is not None]
    if covA and covB and _HAVE_SCIPY:
        XA = np.array([trials_A[i].covariates for i in covA], dtype=float)
        XB = np.array([trials_B[j].covariates for j in covB], dtype=float)
        sd = XA.std(0) + XB.std(0) + 1e-12
        XAn, XBn = XA / sd, XB / sd
        bal_before = _std_mean_diff(XAn, XBn)
        # cost = euclidean distance in standardized covariate space
        cost = np.linalg.norm(XAn[:, None, :] - XBn[None, :, :], axis=2)
        ri, cj = linear_sum_assignment(cost)
        mA, mB = [], []
        for r, c in zip(ri, cj):
            if cost[r, c] <= caliper:
                i, j = covA[r], covB[c]
                used_A[i] = used_B[j] = True
                pairs.append((i, j, "covariate"))
                rung_counts["covariate"] += 1
                mA.append(XAn[r]); mB.append(XBn[c])
        if mA:
            bal_after = _std_mean_diff(np.array(mA), np.array(mB))
        rem_A = [i for i in range(len(trials_A)) if not used_A[i]]
        rem_B = [j for j in range(len(trials_B)) if not used_B[j]]

    # ---- rung 3: unpaired (random pairing of leftovers) -------------------
    rng.shuffle(rem_A); rng.shuffle(rem_B)
    for i, j in zip(rem_A, rem_B):
        used_A[i] = used_B[j] = True
        pairs.append((i, j, "unpaired"))
        rung_counts["unpaired"] += 1

    n_unmatched_A = sum(1 for u in used_A if not u)
    n_unmatched_B = sum(1 for u in used_B if not u)

    return Pairing(pairs, rung_counts, n_unmatched_A, n_unmatched_B,
                   bal_before, bal_after)
