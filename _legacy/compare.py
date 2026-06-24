"""
compare.py -- baseline entry point: anytime-valid A/B success test, human-only.

This is the simple front door. You bring per-rollout success labels for policy A
and policy B (the human's yes/no). It pairs them as well as the data allows and
runs a betting e-process: you may check after every rollout and stop the moment
the evidence crosses the threshold, with a false-positive rate <= alpha at ANY
stopping time.

baseline uses the human label on every rollout, so there is no label-efficiency win
yet -- the value here is purely *trustworthy stop-early testing*. The prediction-
powered machinery (cheap VLM/automatic evaluators + cost-aware human auditing,
the ~3x human-label saving) lives in the same engine and switches on in enhanced.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np

from .engine import BettingEProcess
from .matching import Trial, match, Pairing
from .report import Report

TrialLike = Union[int, Trial]


def _as_trials(xs: Sequence[TrialLike]) -> List[Trial]:
    out = []
    for x in xs:
        out.append(x if isinstance(x, Trial) else Trial(success=int(x)))
    return out


def compare_success(
    successes_A: Sequence[TrialLike],
    successes_B: Sequence[TrialLike],
    *,
    alpha: float = 0.10,
    alternative: str = "two-sided",      # "two-sided" | "A>B" | "B>A"
    adaptive: bool = True,
    caliper: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> Report:
    """Anytime-valid comparison of two policies' success rates.

    Inputs are either 0/1 successes or `Trial` objects (carrying match keys /
    covariates). Returns a Report; inspect `.decided`, `.direction`, `.summary()`.
    """
    rng = rng or np.random.default_rng(0)
    A, B = _as_trials(successes_A), _as_trials(successes_B)
    pairing: Pairing = match(A, B, caliper=caliper, rng=rng)

    # two one-sided e-processes; split alpha for a two-sided guarantee (union bound)
    a_each = alpha / 2 if alternative == "two-sided" else alpha
    ep_AgtB = BettingEProcess(alpha=a_each, adaptive=adaptive)   # H0: pA <= pB
    ep_BgtA = BettingEProcess(alpha=a_each, adaptive=adaptive)   # H0: pB <= pA
    test_AgtB = alternative in ("two-sided", "A>B")
    test_BgtA = alternative in ("two-sided", "B>A")

    wealth_AgtB, wealth_BgtA = [], []
    pair_diffs, pair_rungs = [], []
    decided, direction, n_to_decision = False, None, None
    sa = sb = 0

    for t, (i, j, rung) in enumerate(pairing.pairs, start=1):
        ya, yb = A[i].success, B[j].success
        sa += ya; sb += yb
        pair_diffs.append(ya - yb); pair_rungs.append(rung)
        if test_AgtB:
            ep_AgtB.update(ya - yb)                 # g in {-1,0,1}, E[g]<=0 under H0
        if test_BgtA:
            ep_BgtA.update(yb - ya)
        wealth_AgtB.append(ep_AgtB.wealth)
        wealth_BgtA.append(ep_BgtA.wealth)
        if not decided:
            if test_AgtB and ep_AgtB.decided:
                decided, direction, n_to_decision = True, "A>B", t
            elif test_BgtA and ep_BgtA.decided:
                decided, direction, n_to_decision = True, "B>A", t

    n_pairs = pairing.n_pairs
    e_value = max(max(wealth_AgtB, default=1.0) if test_AgtB else 1.0,
                  max(wealth_BgtA, default=1.0) if test_BgtA else 1.0)

    return Report(
        alpha=alpha, alternative=alternative,
        decided=decided, direction=direction, n_to_decision=n_to_decision,
        n_pairs=n_pairs, n_A=len(A), n_B=len(B),
        successes_A=sa, successes_B=sb,
        rate_A=sa / max(n_pairs, 1), rate_B=sb / max(n_pairs, 1),
        effect=(sa - sb) / max(n_pairs, 1),
        e_value=float(e_value), p_value_equiv=float(1.0 / max(e_value, 1e-12)),
        wealth_AgtB=wealth_AgtB, wealth_BgtA=wealth_BgtA,
        pair_diffs=pair_diffs, pair_rungs=pair_rungs,
        rung_counts=pairing.rung_counts,
        n_unmatched_A=pairing.n_unmatched_A, n_unmatched_B=pairing.n_unmatched_B,
        covariate_balance_before=pairing.covariate_balance_before,
        covariate_balance_after=pairing.covariate_balance_after,
    )
