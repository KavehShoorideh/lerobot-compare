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
eprocess -- betting e-process and observation dataclass.

Validity (type-I <= alpha at every stopping time, via Ville's inequality)
rests on ONE thing: the per-step payoff has non-positive mean under H0. The
evaluator cancels in the mean and only affects variance, so a useless
evaluator costs efficiency, never validity. All data-driven quantities
(pi_t, lambda_PP, lambda_t) are PREDICTABLE -- computed from the past only,
never from the gold label about to be bought -- which is what keeps the
supermartingale property intact under adaptive auditing.

numpy only; no torch / lerobot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Obs:
    """One paired rollout observation."""
    pred_A: float
    pred_B: float
    audited: bool
    pi: float
    gold_A: Optional[int] = None
    gold_B: Optional[int] = None


@dataclass
class BettingEProcess:
    """One-sided test (super)martingale; reject H0 when wealth >= 1/alpha.

        W_t = prod (1 + lam_t * g_t),   g_t in [-1, 1],  lam_t in [0, lam_max].

    If E[g_t | past] <= 0 under H0 then (W_t) is a nonneg supermartingale with
    W_0 = 1, so Ville gives  P(sup_t W_t >= 1/alpha) <= alpha.

    Fixed betting fraction (adaptive=False): uses a constant lam_fixed.
    Adaptive betting fraction (adaptive=True): predictable Kelly-style
        lam_t = clip(g_bar / g2_bar), the Waudby-Smith--Ramdas betting
        fraction. Uses ONLY past payoffs, so it stays predictable and
        validity is preserved.
    """
    alpha: float = 0.10
    adaptive: bool = True
    lam_fixed: float = 0.5
    lam_max: float = 0.9            # < 1 keeps 1 + lam*g > 0 for g >= -1
    warmup: int = 5                 # bet small until a little data accrues

    log_wealth: float = 0.0
    _n: int = 0
    _sum_g: float = 0.0
    _sum_g2: float = 0.0

    @property
    def wealth(self) -> float:
        return float(np.exp(self.log_wealth))

    @property
    def decided(self) -> bool:
        return self.wealth >= 1.0 / self.alpha

    def _lambda(self) -> float:
        """Predictable betting fraction (depends on PAST payoffs only)."""
        if not self.adaptive:
            return self.lam_fixed
        if self._n < self.warmup or self._sum_g2 <= 0.0:
            return 0.0
        g_bar = self._sum_g / self._n
        g2_bar = self._sum_g2 / self._n
        return float(np.clip(g_bar / g2_bar, 0.0, self.lam_max))  # one-sided Kelly

    def update(self, payoff: float) -> None:
        g = float(np.clip(payoff, -1.0, 1.0))
        lam = self._lambda()
        self.log_wealth += float(np.log1p(lam * g))
        self._n += 1
        self._sum_g += g
        self._sum_g2 += g * g
