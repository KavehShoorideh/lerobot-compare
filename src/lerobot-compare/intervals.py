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
intervals.py -- anytime-valid confidence sequence + Clopper-Pearson.

betting_confidence_sequence: an anytime-valid interval for the mean effect
mu = p_A - p_B that NARROWS as pairs accumulate. Built by test inversion: a value
m is kept in the interval at time t as long as a betting capital process (the same
construction as the comparison engine) has not rejected it. Because it reuses the
engine, the sequence inherits the engine's validity -- the true mu is in every
interval simultaneously with probability >= 1 - alpha.

clopper_pearson: exact binomial interval, used for the "how many rollouts" power
curve (the TRI / LBM-style plot of CI width vs N).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.stats import beta as _beta

from .core.eprocess import BettingEProcess


def clopper_pearson(k: int, n: int, alpha: float = 0.10) -> Tuple[float, float]:
    """Exact (conservative) binomial CI for a success probability."""
    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else _beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else _beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lo), float(hi))


def cp_halfwidth_curve(p: float, ns, alpha: float = 0.10):
    """Half-width of the Clopper-Pearson CI at rate p over a range of N."""
    out = []
    for n in ns:
        k = int(round(p * n))
        lo, hi = clopper_pearson(k, n, alpha)
        out.append((hi - lo) / 2)
    return out


def betting_confidence_sequence(
    diffs: List[int], alpha: float = 0.10, *,
    grid: Optional[np.ndarray] = None, adaptive: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Anytime-valid CI for the mean of `diffs` (each in {-1,0,1}) over time.

    Returns (t, lo, hi) with lo/hi in effect units [-1, 1]. Two-sided level alpha
    via a hedged capital (alpha/2 per side) at each grid point.
    """
    if grid is None:
        grid = np.linspace(0.0, 1.0, 161)          # candidate means in x-space [0,1]
    x = (np.asarray(diffs, dtype=float) + 1.0) / 2.0   # map {-1,0,1} -> {0,.5,1}
    thr = 1.0 / (alpha / 2.0)

    # one growing capital per side per grid point
    eps_gt = [BettingEProcess(alpha=alpha / 2, adaptive=adaptive) for _ in grid]
    eps_lt = [BettingEProcess(alpha=alpha / 2, adaptive=adaptive) for _ in grid]

    T = len(x)
    lo = np.empty(T); hi = np.empty(T)
    for t in range(T):
        included = []
        for gi, m in enumerate(grid):
            eps_gt[gi].update(x[t] - m)             # rejects -> mean > m
            eps_lt[gi].update(m - x[t])             # rejects -> mean < m
            if eps_gt[gi].wealth < thr and eps_lt[gi].wealth < thr:
                included.append(m)
        if included:
            lo[t] = 2 * min(included) - 1           # map x-space back to effect space
            hi[t] = 2 * max(included) - 1
        else:                                        # everything rejected: degenerate
            lo[t] = hi[t] = 2 * float(x[:t + 1].mean()) - 1
    return np.arange(1, T + 1), lo, hi


def betting_confidence_interval(
    diffs: List[int], alpha: float = 0.10, *, adaptive: bool = True,
) -> Tuple[float, float]:
    """Final (after-all-pairs) anytime-valid CI for the mean effect p_A - p_B.

    A thin convenience wrapper over `betting_confidence_sequence` that returns just
    the last `(lo, hi)` of the narrowing sequence, in effect units [-1, 1]. This is
    the interval the report shows for the anytime-valid method; it is WIDER than a
    fixed-horizon CI at the same alpha, which is the price of being allowed to peek.
    """
    if not diffs:
        return (-1.0, 1.0)
    _, lo, hi = betting_confidence_sequence(diffs, alpha, adaptive=adaptive)
    return (float(lo[-1]), float(hi[-1]))
