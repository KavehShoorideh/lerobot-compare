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
payoff -- anytime-valid payoff / estimand for paired policy comparison.

Prediction-powered score with importance weighting so it is UNBIASED for
p_P under adaptive auditing (pi predictable):

    importance-weighted (IPW):       m = f + 1{aud}/pi * (y - f)
    control-variate (PPI++):         m = lam_PP*f + 1{aud}/pi * (y - lam_PP*f)

lam_PP in [0,1] is the predictable tuning weight (a regression slope of y
on f over audited items). Good evaluator -> lam_PP -> 1 (full control
variate); useless evaluator -> lam_PP -> 0 (degrade to gold-only IPW,
never worse). For ANY lam_PP, E[m] = p_P, so validity is untouched.

Package note: the PPI++ tuning weight below is the canonical construction
from `ppi-py` (PPI++). The library is not installed in this sandbox, so it
is implemented directly here and remains a drop-in swap locally.

numpy only; no torch / lerobot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .eprocess import Obs


@dataclass
class Comparison:
    """Anytime-valid test of  H0: p_A <= p_B  on paired rollouts.

    Prediction-powered score with importance weighting so it is UNBIASED for
    p_P under adaptive auditing (pi predictable).

    control_variate=True  uses PPI++ tuned weight (lam_PP adapts to evaluator
    quality -- full control variate for a good evaluator, degrades gracefully
    to gold-only IPW for a useless evaluator).

    control_variate=False uses a fixed importance-weighted (IPW) payoff.
    """
    pi_min: float = 0.05
    control_variate: bool = True     # PPI++ control-variate if True, IPW if False
    warmup: int = 10

    # running stats over AUDITED items, for the lam_PP regression slope
    _n: int = 0
    _sf: float = 0.0
    _sy: float = 0.0
    _sff: float = 0.0
    _sfy: float = 0.0

    @property
    def _bound(self) -> float:
        # |m| <= 1 + 1/pi_min ; |m_A - m_B| <= 2*(1 + 1/pi_min)
        return 2.0 * (1.0 + 1.0 / self.pi_min)

    def _lam_pp(self) -> float:
        """Predictable PPI++ weight from PAST audited (f, y) pairs only."""
        if not self.control_variate:
            return 1.0
        if self._n < self.warmup:
            return 0.0
        var_f = self._sff / self._n - (self._sf / self._n) ** 2
        if var_f <= 1e-9:
            return 0.0
        cov_fy = self._sfy / self._n - (self._sf / self._n) * (self._sy / self._n)
        return float(np.clip(cov_fy / var_f, 0.0, 1.0))

    def _score(self, f: float, audited: bool, pi: float, y: Optional[int],
               lam_pp: float) -> float:
        m = lam_pp * f
        if audited:
            m += (y - lam_pp * f) / pi
        return m

    def payoff(self, o: Obs) -> float:
        lam_pp = self._lam_pp()                      # predictable (past only)
        mA = self._score(o.pred_A, o.audited, o.pi, o.gold_A, lam_pp)
        mB = self._score(o.pred_B, o.audited, o.pi, o.gold_B, lam_pp)
        return (mA - mB) / self._bound               # E = (p_A - p_B)/B, <=0 under H0

    def observe(self, o: Obs) -> None:
        """Update lam_PP stats AFTER the payoff is formed (keeps predictability)."""
        if o.audited and self.control_variate:
            for f, y in ((o.pred_A, o.gold_A), (o.pred_B, o.gold_B)):
                self._n += 1
                self._sf += f
                self._sy += y
                self._sff += f * f
                self._sfy += f * y
