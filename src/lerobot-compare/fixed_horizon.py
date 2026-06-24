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
fixed_horizon.py -- the classic FIXED-HORIZON A/B test on paired success labels.

"Fixed-horizon" means the sample size is chosen in advance: you collect N rollouts,
then run the test ONCE. This is the textbook A/B (a controlled experiment comparing
variant A against variant B) test used by most experimentation platforms (Statsig,
Optimizely, ...). It is the counterpart to the anytime-valid betting e-process in
`core/eprocess.py`: simpler and slightly more powerful at the planned N, but you may
NOT peek at the data and stop early without inflating the false-positive rate (the
"peeking problem"). If you need to monitor continuously, use the e-process instead.

The unit of analysis is the matched pair. For pair t we observe

    d_t = y_A_t - y_B_t   in {-1, 0, +1}

(policy A's success minus policy B's success on the same/comparable task). The mean
of d over the N pairs estimates the true success-rate gap

    effect = p_A - p_B

and we test H0: effect = 0 (or a one-sided variant) with a one-sample z-test on the
d_t -- i.e. the test statistic is referred to the standard-normal (Gaussian)
distribution.

Acronyms introduced here (kept consistent with the README glossary):
  CI  = confidence interval        -- a plausible range for the effect
  SE  = standard error             -- the standard deviation of an estimate
  MDE = minimum detectable effect  -- the smallest true effect a design can reliably
                                      catch at a target power
  z-test = a hypothesis test using the standard-normal approximation

Relationship to two textbook tests (so the code's scope is explicit):
  * For binary (0/1) outcomes, this paired z-test on the d_t is asymptotically
    equivalent to McNEMAR'S TEST -- the classic test for paired binary data, which
    looks only at the "discordant" pairs where A and B disagree (d_t = +/-1).
  * For randomly-paired, independent, equal-size samples it further reduces to the
    standard TWO-PROPORTION z-test (`two_proportion_z_test` below). So a single
    paired implementation covers both the matched and the unmatched case; the
    explicit two-proportion helper is provided for callers who hold only aggregate
    success counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.stats import norm


@dataclass
class FixedStats:
    """Everything the fixed-horizon test computes about the effect (the A-B gap).

    `ci_lo`/`ci_hi` use `None` to denote an unbounded side (-inf / +inf), which only
    happens for a one-sided alternative; this keeps the value JSON-serialisable
    (unlike a literal float infinity).
    """
    effect: float                  # absolute effect = mean(y_A - y_B) = p_A - p_B
    se: float                      # standard error (SE) of `effect`
    z: float                       # z-score = effect / se (the test statistic)
    p_value: float                 # p-value for the chosen alternative
    ci_lo: Optional[float]         # (1 - alpha) CI lower bound (None means -inf)
    ci_hi: Optional[float]         # (1 - alpha) CI upper bound (None means +inf)
    power: Optional[float] = None  # achieved power at the observed effect & SE
    mde: Optional[float] = None    # minimum detectable effect at the target power


def _z(prob: float) -> float:
    """Standard-normal quantile (inverse CDF) at `prob`; e.g. `_z(0.975) ~ 1.96`."""
    return float(norm.ppf(prob))


def _p_value(z: float, alternative: str) -> float:
    """Convert a z-score into a p-value for the requested alternative hypothesis."""
    if alternative == "two-sided":            # H1: effect != 0
        return float(2.0 * norm.sf(abs(z)))
    if alternative == "A>B":                  # H1: effect > 0  (A better than B)
        return float(norm.sf(z))
    if alternative == "B>A":                  # H1: effect < 0  (B better than A)
        return float(norm.cdf(z))
    raise ValueError(
        f"alternative must be 'two-sided' | 'A>B' | 'B>A', got {alternative!r}")


def _ci(effect: float, se: float, alpha: float, alternative: str):
    """Confidence interval for `effect` at confidence level (1 - alpha).

    Two-sided -> a symmetric `effect +/- z_{1-alpha/2}*SE`. One-sided -> a one-sided
    confidence BOUND with the other side left open (`None`), so that "CI excludes 0"
    is exactly equivalent to "significant" for every alternative.
    """
    if se == 0.0:                             # degenerate: no variance to speak of
        return (effect, effect)
    if alternative == "two-sided":
        h = _z(1.0 - alpha / 2.0) * se
        return (effect - h, effect + h)
    h = _z(1.0 - alpha) * se
    if alternative == "A>B":
        return (effect - h, None)             # lower bound only; upper side is +inf
    return (None, effect + h)                 # upper bound only; lower side is -inf


def achieved_power(
    effect: float, se: float, alpha: float = 0.10, alternative: str = "two-sided",
) -> float:
    """Probability this fixed-N design would reject H0 GIVEN the observed effect & SE.

    This is the "observed" (post-hoc) power, reported the way experimentation
    platforms report an experiment's power. Treat it as a design diagnostic ("was N
    large enough to see an effect this size?"), not as extra evidence about this
    particular run.
    """
    if se == 0.0:
        return 1.0 if effect != 0.0 else float(alpha)
    ncp = effect / se                         # noncentrality: the z-shift under truth
    if alternative == "two-sided":
        zc = _z(1.0 - alpha / 2.0)
        return float(norm.sf(zc - ncp) + norm.cdf(-zc - ncp))
    zc = _z(1.0 - alpha)
    if alternative == "A>B":
        return float(norm.sf(zc - ncp))
    return float(norm.cdf(-zc - ncp))         # "B>A"


def min_detectable_effect(
    se: float, alpha: float = 0.10, power: float = 0.80, alternative: str = "two-sided",
) -> float:
    """Minimum detectable effect (MDE) at the realised SE:

        MDE = (z_alpha + z_power) * SE

    i.e. the smallest |effect| this fixed-N design would detect with probability
    `power`. Smaller is better (you can resolve subtler differences).
    """
    if se == 0.0:
        return 0.0
    z_alpha = _z(1.0 - alpha / 2.0) if alternative == "two-sided" else _z(1.0 - alpha)
    z_power = _z(power)
    return float((z_alpha + z_power) * se)


def paired_z_test(
    pair_diffs: Sequence[int], alpha: float = 0.10, *,
    alternative: str = "two-sided", target_power: float = 0.80,
) -> FixedStats:
    """Fixed-horizon paired z-test on per-pair differences `d_t = y_A - y_B`.

    This is the engine behind `method="fixed-horizon-proportion-z-test"`. It is the
    matched-pairs test (McNemar-equivalent for binary data) and degrades gracefully
    to the two-proportion z-test when the pairs are random and balanced.
    """
    d = np.asarray(list(pair_diffs), dtype=float)
    n = d.size
    if n == 0:
        return FixedStats(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, None, None)

    effect = float(d.mean())
    # Sample variance with Bessel's correction (ddof=1); SE of the mean = sqrt(var/n).
    var = float(d.var(ddof=1)) if n > 1 else 0.0
    se = float(np.sqrt(var / n)) if var > 0.0 else 0.0

    if se == 0.0:                             # all pairs identical: no sampling noise
        z = 0.0 if effect == 0.0 else float(np.copysign(np.inf, effect))
        p = 1.0 if effect == 0.0 else 0.0
    else:
        z = effect / se
        p = _p_value(z, alternative)

    lo, hi = _ci(effect, se, alpha, alternative)
    return FixedStats(
        effect=effect, se=se, z=float(z), p_value=float(p), ci_lo=lo, ci_hi=hi,
        power=achieved_power(effect, se, alpha, alternative),
        mde=min_detectable_effect(se, alpha, target_power, alternative),
    )


def two_proportion_z_test(
    succ_A: int, n_A: int, succ_B: int, n_B: int, alpha: float = 0.10, *,
    alternative: str = "two-sided", target_power: float = 0.80,
) -> FixedStats:
    """Classic unpaired two-proportion z-test (the textbook A/B test when you only
    hold aggregate success COUNTS and have no pairing).

    The test statistic uses the POOLED SE (the right null-variance under H0:
    p_A = p_B); the confidence interval and reported SE use the UNPOOLED SE (the
    standard convention, since under H1 the two rates differ).
    """
    if n_A == 0 or n_B == 0:
        return FixedStats(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, None, None)

    pA, pB = succ_A / n_A, succ_B / n_B
    effect = float(pA - pB)
    p_pool = (succ_A + succ_B) / (n_A + n_B)
    se_pool = float(np.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n_A + 1.0 / n_B)))
    se = float(np.sqrt(pA * (1.0 - pA) / n_A + pB * (1.0 - pB) / n_B))  # unpooled (CI/SE)

    if se_pool == 0.0:
        z = 0.0 if effect == 0.0 else float(np.copysign(np.inf, effect))
        p = 1.0 if effect == 0.0 else 0.0
    else:
        z = effect / se_pool
        p = _p_value(z, alternative)

    lo, hi = _ci(effect, se, alpha, alternative)
    return FixedStats(
        effect=effect, se=se, z=float(z), p_value=float(p), ci_lo=lo, ci_hi=hi,
        power=achieved_power(effect, se, alpha, alternative),
        mde=min_detectable_effect(se, alpha, target_power, alternative),
    )
