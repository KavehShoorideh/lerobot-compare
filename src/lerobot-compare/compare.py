#!/usr/bin/env python

# Copyright 2026 Kaveh Shoorideh
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
compare.py -- direct A/B comparison entry point: pick a test METHOD, get a Report.

You bring per-rollout success labels for policy A and policy B (the human's yes/no).
`compare_success` pairs them as well as the data allows and then runs the test
METHOD you select. Two methods ship today, chosen with the `method=` argument whose
*values are full, descriptive names* so future tests slot in without renaming:

  "anytime-valid-betting-e-process"  (default)
      The betting e-process: false-positive rate <= alpha at ANY stopping time, so
      you may check after every rollout and stop the moment the evidence crosses the
      threshold. Reports an e-value and a (deliberately wider) anytime-valid CI.

  "fixed-horizon-proportion-z-test"
      The classic fixed-sample-size A/B test (the kind Statsig & co. run by default).
      You analyse ONCE, at the planned N -- no peeking. Reports a p-value, standard
      error (SE), achieved power, and minimum detectable effect (MDE).

Both methods feed the SAME paired data through the SAME matching step and return the
SAME `Report` shape, so the scorecard and JSON are uniform; only the headline
statistics differ. A shared "enrich" pass then adds the platform-style extras that
do not depend on the method: per-variant confidence intervals, relative lift, the
Sample Ratio Mismatch (SRM) health check, and the Bayesian chance-to-beat.

(The label-efficient, cheap-evaluator + human-auditing path -- the ~3x human-label
saving -- lives in `core/compare.py`; this front door uses the human label on every
rollout, so its value is purely trustworthy, method-flexible testing.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Union

import numpy as np

from .bayesian import chance_to_beat
from .checks import sample_ratio_mismatch
from .core.eprocess import BettingEProcess
from .fixed_horizon import paired_z_test
from .intervals import betting_confidence_interval, clopper_pearson
from .matching import Pairing, Trial, match
from .report import DEFAULT_METHOD, Report

TrialLike = Union[int, Trial]


def _as_trials(xs: Sequence[TrialLike]) -> List[Trial]:
    """Accept either raw 0/1 successes or full `Trial` objects (with match keys)."""
    return [x if isinstance(x, Trial) else Trial(success=int(x)) for x in xs]


# --------------------------------------------------------------------------- #
# Shared context: everything both engines need, computed once from the pairing. #
# --------------------------------------------------------------------------- #
@dataclass
class _Ctx:
    A: List[Trial]
    B: List[Trial]
    pairing: Pairing
    diffs: List[int]            # per-pair d = y_A - y_B, in {-1, 0, +1}
    rungs: List[str]            # match rung for each pair
    sa: int                     # successes among paired A rollouts
    sb: int                     # successes among paired B rollouts
    n_pairs: int
    rate_A: float
    rate_B: float
    effect: float               # rate_A - rate_B (absolute gap)
    alpha: float
    alternative: str
    adaptive: bool
    target_power: float
    rng: np.random.Generator


def _base_report(
    ctx: _Ctx, *, decided: bool, direction: Optional[str], n_to_decision: Optional[int],
    effect: float, e_value: float, p_value_equiv: float,
    effect_ci_lo: Optional[float], effect_ci_hi: Optional[float],
    e_threshold: Optional[float] = None,
    wealth_AgtB: Optional[List[float]] = None, wealth_BgtA: Optional[List[float]] = None,
    standard_error: Optional[float] = None, z_score: Optional[float] = None,
    p_value: Optional[float] = None, power: Optional[float] = None,
    mde: Optional[float] = None,
) -> Report:
    """Build a `Report` with all the fields common to every method filled in; the
    method-specific statistics arrive as keyword arguments."""
    p = ctx.pairing
    return Report(
        alpha=ctx.alpha, alternative=ctx.alternative,
        decided=decided, direction=direction, n_to_decision=n_to_decision,
        n_pairs=ctx.n_pairs, n_A=len(ctx.A), n_B=len(ctx.B),
        successes_A=ctx.sa, successes_B=ctx.sb,
        rate_A=ctx.rate_A, rate_B=ctx.rate_B, effect=effect,
        e_value=float(e_value), p_value_equiv=float(p_value_equiv),
        e_threshold=e_threshold,
        effect_ci_lo=effect_ci_lo, effect_ci_hi=effect_ci_hi,
        standard_error=standard_error, z_score=z_score, p_value=p_value,
        power=power, mde=mde, target_power=ctx.target_power,
        wealth_AgtB=wealth_AgtB or [], wealth_BgtA=wealth_BgtA or [],
        pair_diffs=ctx.diffs, pair_rungs=ctx.rungs,
        rung_counts=p.rung_counts,
        n_unmatched_A=p.n_unmatched_A, n_unmatched_B=p.n_unmatched_B,
        covariate_balance_before=p.covariate_balance_before,
        covariate_balance_after=p.covariate_balance_after,
    )


# --------------------------------------------------------------------------- #
# Engine 1: the anytime-valid betting e-process (the original logic, unchanged). #
# --------------------------------------------------------------------------- #
def _run_eprocess(ctx: _Ctx) -> Report:
    # Two one-sided e-processes; split alpha for a two-sided guarantee (union bound).
    a_each = ctx.alpha / 2 if ctx.alternative == "two-sided" else ctx.alpha
    ep_AgtB = BettingEProcess(alpha=a_each, adaptive=ctx.adaptive)   # H0: pA <= pB
    ep_BgtA = BettingEProcess(alpha=a_each, adaptive=ctx.adaptive)   # H0: pB <= pA
    test_AgtB = ctx.alternative in ("two-sided", "A>B")
    test_BgtA = ctx.alternative in ("two-sided", "B>A")

    wealth_AgtB: List[float] = []
    wealth_BgtA: List[float] = []
    decided, direction, n_to_decision = False, None, None

    for t, d in enumerate(ctx.diffs, start=1):
        if test_AgtB:
            ep_AgtB.update(d)               # g in {-1,0,1}, E[g] <= 0 under H0: pA <= pB
        if test_BgtA:
            ep_BgtA.update(-d)
        wealth_AgtB.append(ep_AgtB.wealth)
        wealth_BgtA.append(ep_BgtA.wealth)
        if not decided:
            if test_AgtB and ep_AgtB.decided:
                decided, direction, n_to_decision = True, "A>B", t
            elif test_BgtA and ep_BgtA.decided:
                decided, direction, n_to_decision = True, "B>A", t

    e_value = max(max(wealth_AgtB, default=1.0) if test_AgtB else 1.0,
                  max(wealth_BgtA, default=1.0) if test_BgtA else 1.0)
    # Honest decision bar and anytime-valid p-value. A two-sided test spends alpha/2
    # per side (union bound), so it must reach 2/alpha and its p-value is 2/e_value;
    # a one-sided test uses 1/alpha and 1/e_value. `sides` captures that factor.
    sides = 2 if ctx.alternative == "two-sided" else 1
    e_threshold = sides / ctx.alpha
    p_value_equiv = min(1.0, sides / max(e_value, 1e-12))
    # Anytime-valid effect CI by test inversion (wider than fixed-horizon at same alpha).
    lo, hi = betting_confidence_interval(ctx.diffs, ctx.alpha, adaptive=ctx.adaptive)
    return _base_report(
        ctx, decided=decided, direction=direction, n_to_decision=n_to_decision,
        effect=ctx.effect, e_value=float(e_value),
        p_value_equiv=p_value_equiv, e_threshold=e_threshold,
        effect_ci_lo=lo, effect_ci_hi=hi,
        wealth_AgtB=wealth_AgtB, wealth_BgtA=wealth_BgtA,
    )


# --------------------------------------------------------------------------- #
# Engine 2: the fixed-horizon paired proportion z-test.                         #
# --------------------------------------------------------------------------- #
def _run_fixed_z(ctx: _Ctx) -> Report:
    stats = paired_z_test(ctx.diffs, ctx.alpha, alternative=ctx.alternative,
                          target_power=ctx.target_power)
    decided = bool(ctx.n_pairs > 0 and stats.p_value < ctx.alpha)
    if not decided:
        direction = None
    elif ctx.alternative in ("A>B", "B>A"):
        direction = ctx.alternative                  # one-sided: the tested direction
    else:
        direction = "A>B" if stats.effect > 0 else "B>A"
    return _base_report(
        ctx, decided=decided, direction=direction, n_to_decision=None,
        effect=stats.effect, e_value=1.0, p_value_equiv=1.0,
        effect_ci_lo=stats.ci_lo, effect_ci_hi=stats.ci_hi,
        standard_error=stats.se, z_score=stats.z, p_value=stats.p_value,
        power=stats.power, mde=stats.mde,
    )


# --------------------------------------------------------------------------- #
# Method registry: full descriptive name -> engine + display labels. Adding a   #
# future test (e.g. "anytime-valid-mixture-sprt") is a single new entry here.    #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Method:
    run: Callable[[_Ctx], Report]
    horizon: str                # "anytime-valid" | "fixed-horizon"  (for the scorecard)
    test: str                   # human-readable specific-test label


_METHODS = {
    "anytime-valid-betting-e-process": _Method(_run_eprocess, "anytime-valid", "betting e-process"),
    "fixed-horizon-proportion-z-test": _Method(_run_fixed_z, "fixed-horizon", "proportion z-test"),
}


def available_methods() -> List[str]:
    """The full descriptive names accepted by `compare_success(method=...)`."""
    return list(_METHODS)


def _enrich(report: Report, ctx: _Ctx) -> None:
    """Add the method-independent, platform-style extras to a finished report."""
    # Per-variant exact binomial (Clopper-Pearson) CIs on each policy's success rate.
    report.rate_A_ci = clopper_pearson(ctx.sa, ctx.n_pairs, ctx.alpha)
    report.rate_B_ci = clopper_pearson(ctx.sb, ctx.n_pairs, ctx.alpha)

    # Relative lift = (rate_A - rate_B) / rate_B (Statsig's headline "Delta %"). The
    # relative CI is the absolute CI rescaled by the control rate -- a first-order
    # approximation, but the convention experimentation platforms display.
    if ctx.rate_B > 0:
        report.effect_relative = report.effect / ctx.rate_B
        lo, hi = report.effect_ci_lo, report.effect_ci_hi
        report.effect_relative_ci_lo = None if lo is None else lo / ctx.rate_B
        report.effect_relative_ci_hi = None if hi is None else hi / ctx.rate_B

    # Sample Ratio Mismatch: did each policy actually get the rollout count we expect?
    srm = sample_ratio_mismatch(len(ctx.A), len(ctx.B))
    report.srm_p_value = float(srm.data.get("p_value", 1.0))
    report.srm_flag = srm.level == "warn"

    # Bayesian companion: posterior probability that A's true rate beats B's.
    report.chance_to_beat = chance_to_beat(
        ctx.sa, ctx.n_pairs, ctx.sb, ctx.n_pairs, rng=ctx.rng)


def compare_success(
    successes_A: Sequence[TrialLike],
    successes_B: Sequence[TrialLike],
    *,
    method: str = DEFAULT_METHOD,        # see _METHODS / available_methods()
    alpha: float = 0.10,
    alternative: str = "two-sided",      # "two-sided" | "A>B" | "B>A"
    adaptive: bool = True,               # betting e-process only
    caliper: float = 1.0,
    target_power: float = 0.80,          # power level the fixed-horizon MDE targets
    rng: Optional[np.random.Generator] = None,
) -> Report:
    """Direct A/B comparison of two policies' success rates.

    Inputs are either 0/1 successes or `Trial` objects (carrying match keys /
    covariates). `method` selects the test (default: the anytime-valid betting
    e-process). Returns a `Report`; inspect `.decided`, `.direction`, `.summary()`.
    """
    if method not in _METHODS:
        raise ValueError(
            f"unknown method {method!r}; choose one of {available_methods()}")
    spec = _METHODS[method]

    rng = rng or np.random.default_rng(0)
    A, B = _as_trials(successes_A), _as_trials(successes_B)
    pairing: Pairing = match(A, B, caliper=caliper, rng=rng)

    # Per-pair labels + differences, shared by every method.
    diffs: List[int] = []
    rungs: List[str] = []
    sa = sb = 0
    for (i, j, rung) in pairing.pairs:
        ya, yb = A[i].success, B[j].success
        sa += ya
        sb += yb
        diffs.append(ya - yb)
        rungs.append(rung)
    n_pairs = pairing.n_pairs
    denom = max(n_pairs, 1)

    ctx = _Ctx(
        A=A, B=B, pairing=pairing, diffs=diffs, rungs=rungs, sa=sa, sb=sb,
        n_pairs=n_pairs, rate_A=sa / denom, rate_B=sb / denom,
        effect=(sa - sb) / denom, alpha=alpha, alternative=alternative,
        adaptive=adaptive, target_power=target_power, rng=rng,
    )

    report = spec.run(ctx)
    report.method = method
    report.horizon = spec.horizon
    report.test = spec.test
    report.confidence_level = 1.0 - alpha
    _enrich(report, ctx)
    return report
