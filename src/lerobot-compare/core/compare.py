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
compare -- high-level comparison engine and synthetic world helpers.

Domain-agnostic. Knows only: estimands, evaluators (score + cost), an
auditing rule, and a betting e-process. numpy only; no torch / lerobot.

Package note: the WSR adaptive bet is the canonical construction from
`confseq` (betting confidence sequences). The library is not installed in
this sandbox (confseq needs a Boost/C++ build), so it is implemented
directly in eprocess.py and remains a drop-in swap locally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, Callable, Optional, Any, Tuple, runtime_checkable

import numpy as np

from .eprocess import BettingEProcess, Obs
from .payoff import Comparison
from .auditing import MaxEvidenceAuditor, Auditor


# ===========================================================================
# Evaluators and gold
# ===========================================================================

@runtime_checkable
class Evaluator(Protocol):
    name: str
    cost: float
    def score(self, item: Any) -> float: ...


@dataclass
class GoldSource:
    """Scarce, trustworthy label. label_fn(item) -> (gold_A, gold_B) in {0,1}."""
    cost: float
    label_fn: Callable[[Any], Tuple[int, int]]
    name: str = "gold"

    def label(self, item: Any) -> Tuple[int, int]:
        return self.label_fn(item)


# ===========================================================================
# Result and compare function
# ===========================================================================

@dataclass
class Result:
    decided: bool
    direction: Optional[str]
    wealth: float
    n_seen: int
    n_audited: int
    cost_spent: float


def compare(
    items: Sequence[Any],
    evaluator: Evaluator,
    gold: GoldSource,
    predict: Callable[[Any, Evaluator], Tuple[float, float]],
    rule: Optional[Auditor] = None,
    alpha: float = 0.10,
    adaptive: bool = True,            # adaptive betting fraction if True, fixed if False
    control_variate: bool = True,     # PPI++ control-variate payoff if True, IPW if False
    rng: Optional[np.random.Generator] = None,
) -> Result:
    rule = rule or MaxEvidenceAuditor()
    rng = rng or np.random.default_rng(0)
    est = Comparison(pi_min=getattr(rule, "pi_min", 0.05),
                     control_variate=control_variate)
    ep = BettingEProcess(alpha=alpha, adaptive=adaptive)

    n = n_aud = 0
    cost = 0.0
    for item in items:
        n += 1
        fA, fB = predict(item, evaluator)
        cost += 2 * evaluator.cost
        pi = rule.audit_prob(fA, fB, gold.cost)
        audited = bool(rng.random() < pi)
        gA = gB = None
        if audited:
            gA, gB = gold.label(item)
            cost += 2 * gold.cost
            n_aud += 1
        o = Obs(fA, fB, audited, pi, gA, gB)
        ep.update(est.payoff(o))      # payoff uses predictable lam_PP / lam_t
        est.observe(o)                # then update stats for next step
        if ep.decided:
            return Result(True, "A>B", ep.wealth, n, n_aud, cost)
    return Result(False, None, ep.wealth, n, n_aud, cost)


# ===========================================================================
# Synthetic world (shared by demo and tests)
# ===========================================================================

def make_world(pA: float, pB: float, rng: np.random.Generator,
               rho_acc: float = 0.75, N: int = 1500):
    """Paired rollouts + a GRADED cheap predictor. rho_acc in [0.5, 1]:
    0.5 = useless (confidence independent of truth), 1.0 = strongly informative.
    f is a per-item probability in (0,1) -- like a real VLM's graded score, not
    a hard label -- so an uncertainty-based audit rule has something to exploit.
    """
    yA = (rng.random(N) < pA).astype(int)
    yB = (rng.random(N) < pB).astype(int)
    k = 3.0 * (2.0 * rho_acc - 1.0)              # 0 at rho=0.5, 3.0 at rho=1.0

    def graded(y):
        logit = k * (2 * y - 1) + rng.normal(0.0, 1.0, size=len(y))
        return 1.0 / (1.0 + np.exp(-logit))      # graded confidence in (0,1)

    fA, fB = graded(yA), graded(yB)
    items = list(range(N))

    def predict(i, _ev):
        return float(fA[i]), float(fB[i])

    gold = GoldSource(cost=1.0, label_fn=lambda i: (int(yA[i]), int(yB[i])))
    return items, predict, gold


class CheapVLM:
    """Cheap surrogate evaluator (e.g., a small VLM). Supplies graded predictions."""
    name = "vlm"
    cost = 0.01

    def score(self, item):           # unused in synthetic demo; predict() supplies scores
        return 0.0
