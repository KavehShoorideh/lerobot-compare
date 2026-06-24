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

"""Torch-free stub evaluator for tests and sandbox use."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base import Rollout


# ---- torch-free stub for tests / sandbox ----------------------------------

@dataclass
class StubVLM:
    """Synthetic graded evaluator (no torch). Reads a hidden truth from
    rollout.meta['p'] (or 'y') and emits a score whose fidelity is set by `rho`
    in [0.5, 1.0]: 0.5 = useless coin flip, 1.0 = near-perfect. For tests only."""
    rho: float = 0.8
    cost: float = 1.0
    _rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def score(self, rollout: Rollout) -> float:
        p = rollout.meta.get("p")
        if p is None:
            p = float(rollout.meta.get("y", 0.5))
        # blend truth with noise; sharper as rho -> 1
        k = 1.0 + 12.0 * (self.rho - 0.5)
        base = p if self._rng.random() < self.rho else self._rng.random()
        z = (base - 0.5) * k
        return float(1.0 / (1.0 + np.exp(-z)))
