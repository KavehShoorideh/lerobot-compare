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

"""Cheap automatic and internal-state evaluators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .base import Rollout


# ---- cheap automatic / internal sources -----------------------------------

@dataclass
class AutomaticChecker:
    """A programmatic/sensor check, e.g. an AprilTag pose test -> [0,1]."""
    fn: Callable[[Rollout], float]
    cost: float = 0.001
    def score(self, rollout: Rollout) -> float:
        return float(np.clip(self.fn(rollout), 0.0, 1.0))


@dataclass
class InternalState:
    """A near-free signal from policy internals (action-chunk entropy, OOD score,
    proprioceptive heuristic). fn(rollout) -> [0,1]."""
    fn: Callable[[Rollout], float]
    cost: float = 0.0005
    def score(self, rollout: Rollout) -> float:
        return float(np.clip(self.fn(rollout), 0.0, 1.0))
