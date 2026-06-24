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

"""Updaters that turn accumulated human labels into a better evaluator.

Two jobs, one interface. Both consume the same record stream
(cheap score f, human label y, [frames, question]) and return an improved
Evaluator:

  Recalibrate (here)   learn how much to TRUST the evaluator: an isotonic map
                       f -> calibrated P(success). Cheap, local, no GPU, no
                       download. Runs in-sandbox.
  retrain (training.py) UPDATE the evaluator's weights (LoRA fine-tune). Expensive,
                       GPU, handed off to LLaMA-Factory / SkyPilot.

So `EvaluatorUpdater.update(records) -> Evaluator` is the swap point: recalibrate
vs retrain is just which backend you pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Protocol

import numpy as np

from .base import Evaluator, Rollout


@dataclass
class LabelRecord:
    """One audited datum. `f`/`y` drive recalibration; `frames`/`question` drive
    retraining (the VLM fine-tune target is `y` given `frames` + `question`)."""
    f: float                              # cheap evaluator score in [0,1]
    y: int                                # human gold label in {0,1}
    frames: Any = None
    question: str = ""
    meta: dict = field(default_factory=dict)


class EvaluatorUpdater(Protocol):
    def update(self, records: List[LabelRecord]) -> Any: ...


# ---- isotonic regression via Pool-Adjacent-Violators (numpy only) ----------

def _pava(x: np.ndarray, y: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """Monotone non-decreasing fit of y on sorted x. Returns (knots_x, knots_y)."""
    order = np.argsort(x, kind="mergesort")
    xs, ys = x[order], y[order].astype(float)
    w = np.ones_like(ys)
    # pool adjacent violators
    vals = list(ys); wts = list(w); xsr = list(xs)
    i = 0
    while i < len(vals) - 1:
        if vals[i] > vals[i + 1] + 1e-12:
            new_w = wts[i] + wts[i + 1]
            new_v = (vals[i] * wts[i] + vals[i + 1] * wts[i + 1]) / new_w
            vals[i] = new_v; wts[i] = new_w; xsr[i] = xsr[i + 1]
            del vals[i + 1]; del wts[i + 1]; del xsr[i + 1]
            if i > 0:
                i -= 1
        else:
            i += 1
    return np.array(xsr), np.clip(np.array(vals), 0.0, 1.0)


@dataclass
class _Isotonic:
    kx: np.ndarray
    ky: np.ndarray
    def __call__(self, f: float) -> float:
        return float(np.interp(f, self.kx, self.ky))


@dataclass
class CalibratedEvaluator:
    """Wraps a base evaluator so its score is mapped through a calibration curve."""
    base: Evaluator
    cal: Callable[[float], float]
    cost: float = 0.0
    def __post_init__(self):
        self.cost = self.base.cost
    def score(self, rollout: Rollout) -> float:
        return float(np.clip(self.cal(self.base.score(rollout)), 0.0, 1.0))


@dataclass
class Recalibrate:
    """Fit an isotonic f -> P(success) map from audited (f, y) pairs and return a
    CalibratedEvaluator. Cheap, local, GPU-free. `min_points` guards against
    fitting on too little data (returns the base evaluator unchanged)."""
    base: Evaluator
    min_points: int = 20

    def update(self, records: List[LabelRecord]) -> Evaluator:
        if len(records) < self.min_points:
            return self.base
        f = np.array([r.f for r in records], dtype=float)
        y = np.array([r.y for r in records], dtype=float)
        kx, ky = _pava(f, y)
        if kx.size < 2:                              # degenerate -> identity
            return self.base
        return CalibratedEvaluator(self.base, _Isotonic(kx, ky))
