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
auditing -- predictable audit-probability rules (acquisition strategies).

All rules are PREDICTABLE: audit_prob depends only on cheap predictions
(f_A, f_B) and gold cost, never on the gold label about to be bought.
Predictability is what preserves the supermartingale property under
adaptive auditing.

numpy only; no torch / lerobot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Auditor(Protocol):
    """Protocol for predictable audit-probability rules."""
    def audit_prob(self, pred_A: float, pred_B: float, gold_cost: float) -> float: ...


@dataclass
class MaxEvidenceAuditor:
    """Principled, wealth-growth-oriented acquisition (the project's recommended rule).

    Tilt audits toward DECISIVE items (large |f_A - f_B|): those carry the
    directional A-vs-B evidence AND let the control variate's lam_PP learn that
    f predicts y, so gold is spent where it grows WEALTH fastest -- not where it
    minimizes estimator variance. Critically, keep a HIGH audit floor so the
    importance weights (1/pi) stay bounded; low-pi audits inject huge-variance
    payoffs that sabotage the bet. (That last point is exactly why classical
    Neyman allocation -- audit where UNCERTAIN, with a low floor -- backfires
    here; see NeymanAuditor.) Predictable: depends only on f_A, f_B.

    Validated on synthetic data to reach decisions on ~3x fewer gold labels than
    uniform auditing at matched budget. Real-data tilt/floor tuning is future
    work; the invariant is: directional tilt + bounded weights.
    """
    target_rate: float = 0.4
    pi_min: float = 0.2          # HIGH floor -> bounds 1/pi importance variance
    pi_max: float = 0.8
    ref: float = 0.75

    def audit_prob(self, pred_A: float, pred_B: float, gold_cost: float) -> float:
        tilt = 0.5 + 0.5 * abs(pred_A - pred_B)        # [0.5, 1.0], decisive higher
        pi = self.target_rate * tilt / self.ref
        return float(np.clip(pi, self.pi_min, self.pi_max))


@dataclass
class UniformAuditor:
    """Baseline: audit every item with a fixed probability (no adaptivity)."""
    pi: float = 0.3
    pi_min: float = 0.05

    def audit_prob(self, pred_A: float, pred_B: float, gold_cost: float) -> float:
        return float(np.clip(self.pi, self.pi_min, 1.0))


@dataclass
class NeymanAuditor:
    """DOCUMENTED NEGATIVE RESULT (kept as a baseline, not the recommended rule).

    Neyman allocation -- audit pi_i ∝ residual std sigma_i, i.e. where the
    evaluator is UNCERTAIN -- is variance-optimal for ESTIMATING a mean, but it
    BACKFIRES for sequential DETECTION via a prediction-powered e-process: it
    audits ambiguous items (f ~ 0.5) where y is near-coin-flip, which (a)
    starves the directional A>B signal and (b) makes the control variate's
    lam_PP collapse toward 0 (f does not predict y on those items), while its
    low floor lets the few audits of decisive items carry huge importance
    weight. Empirically this detects far less often than uniform at matched
    budget. The lesson -- spend gold for wealth growth, not estimator variance,
    with bounded weights -- is what MaxEvidenceAuditor implements.
    Predictable: sigma_i depends only on the cheap predictions.
    """
    target_rate: float = 0.3
    pi_min: float = 0.02
    pi_max: float = 1.0
    sigma_ref: float = 0.5

    def audit_prob(self, pred_A: float, pred_B: float, gold_cost: float) -> float:
        var = pred_A * (1.0 - pred_A) + pred_B * (1.0 - pred_B)
        sigma = float(np.sqrt(max(var, 1e-12)))
        pi = self.target_rate * sigma / self.sigma_ref
        return float(np.clip(pi, self.pi_min, self.pi_max))
