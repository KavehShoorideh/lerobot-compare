"""
compare_policies.py -- enhanced entry: cheap evaluator + cost-aware human auditing.

Every matched pair is scored by a CHEAP evaluator (free-ish). A cost-aware
acquisition rule decides which pairs are worth an EXPENSIVE human label. The
prediction-powered payoff makes the result unbiased for ANY evaluator quality
(a bad evaluator costs efficiency, never validity), and the betting e-process
keeps it anytime-valid. The payoff is the ~3x human-label saving over auditing
everything -- which only appears here, once a cheap evaluator carries most pairs.

    compare_policies(rollouts_A, rollouts_B, evaluator, human_gold, ...) -> Report
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .engine import BettingEProcess, Comparison, Obs, MaxEvidencePerCost, AcquisitionRule
from .evaluators import Rollout, Evaluator, HumanGold
from .matching import match, Pairing
from .report import Report


def compare_policies(
    rollouts_A: Sequence[Rollout],
    rollouts_B: Sequence[Rollout],
    evaluator: Evaluator,
    human_gold: HumanGold,
    *,
    alpha: float = 0.10,
    alternative: str = "two-sided",
    acquisition: Optional[AcquisitionRule] = None,
    control_variate: bool = True,
    caliper: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> Report:
    """Anytime-valid A/B comparison that leans on a cheap evaluator and audits
    only where it helps. Returns a enhanced Report (see `.summary()`)."""
    rng = rng or np.random.default_rng(0)
    A, B = list(rollouts_A), list(rollouts_B)
    pairing: Pairing = match(A, B, caliper=caliper, rng=rng)

    acq = acquisition or MaxEvidencePerCost()
    a_each = alpha / 2 if alternative == "two-sided" else alpha
    est = Comparison(pi_min=getattr(acq, "pi_min", 0.05),
                     control_variate=control_variate)
    ep_AgtB = BettingEProcess(alpha=a_each, adaptive=True)
    ep_BgtA = BettingEProcess(alpha=a_each, adaptive=True)
    test_AgtB = alternative in ("two-sided", "A>B")
    test_BgtA = alternative in ("two-sided", "B>A")

    wealth_AgtB, wealth_BgtA, pair_diffs, pair_rungs = [], [], [], []
    decided, direction, n_to_decision = False, None, None
    n_audited = 0
    n_audited_at_decision = None
    sa = sb = na = nb = 0      # audited-success counters (for descriptive rates)

    for t, (i, j, rung) in enumerate(pairing.pairs, start=1):
        ra, rb = A[i], B[j]
        fA, fB = evaluator.score(ra), evaluator.score(rb)         # cheap, every pair
        pi = acq.audit_prob(fA, fB, human_gold.cost)
        audited = rng.random() < pi
        yA = yB = None
        if audited:
            yA, yB = human_gold.label(ra, rb)
            n_audited += 1
            sa += yA; sb += yB; na += 1; nb += 1

        o = Obs(fA, fB, audited, pi, yA, yB)
        g = est.payoff(o)                       # predictable lam_PP used inside
        est.observe(o)                          # update lam_PP stats AFTER
        if test_AgtB:
            ep_AgtB.update(g)
        if test_BgtA:
            ep_BgtA.update(-g)
        wealth_AgtB.append(ep_AgtB.wealth); wealth_BgtA.append(ep_BgtA.wealth)
        pair_diffs.append((yA - yB) if audited else 0); pair_rungs.append(rung)

        if not decided:
            if test_AgtB and ep_AgtB.decided:
                decided, direction, n_to_decision = True, "A>B", t
                n_audited_at_decision = n_audited
            elif test_BgtA and ep_BgtA.decided:
                decided, direction, n_to_decision = True, "B>A", t
                n_audited_at_decision = n_audited

    n_pairs = pairing.n_pairs
    e_value = max(max(wealth_AgtB, default=1.0) if test_AgtB else 1.0,
                  max(wealth_BgtA, default=1.0) if test_BgtA else 1.0)
    rate_A = sa / max(na, 1); rate_B = sb / max(nb, 1)
    # report what happened UP TO the stopping point (anytime-valid use stops there)
    seen = n_to_decision if decided else n_pairs
    aud = n_audited_at_decision if decided else n_audited

    return Report(
        alpha=alpha, alternative=alternative,
        decided=decided, direction=direction, n_to_decision=n_to_decision,
        n_pairs=n_pairs, n_A=len(A), n_B=len(B),
        successes_A=sa, successes_B=sb,
        rate_A=rate_A, rate_B=rate_B, effect=rate_A - rate_B,
        e_value=float(e_value), p_value_equiv=float(1.0 / max(e_value, 1e-12)),
        wealth_AgtB=wealth_AgtB, wealth_BgtA=wealth_BgtA,
        pair_diffs=pair_diffs, pair_rungs=pair_rungs,
        rung_counts=pairing.rung_counts,
        n_unmatched_A=pairing.n_unmatched_A, n_unmatched_B=pairing.n_unmatched_B,
        covariate_balance_before=pairing.covariate_balance_before,
        covariate_balance_after=pairing.covariate_balance_after,
        mode="enhanced", n_audited=aud, n_seen=seen,
        audit_rate=aud / max(seen, 1), lam_pp=est._lam_pp(),
        gold_saved_frac=1.0 - aud / max(seen, 1),
    )
