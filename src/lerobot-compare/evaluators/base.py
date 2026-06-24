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

"""Base rollout type, Evaluator protocol, and the trusted human gold source."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Tuple, runtime_checkable

import numpy as np


@dataclass
class Rollout:
    """One whole-task attempt. `frames` is whatever the evaluator consumes
    (paths, arrays, a handle); `key`/`covariates` drive matching."""
    task: str = ""
    frames: Any = None
    key: Optional[Any] = None
    covariates: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)


@runtime_checkable
class Evaluator(Protocol):
    cost: float
    def score(self, rollout: Rollout) -> float: ...


# ---- the trusted, expensive source ----------------------------------------

@dataclass
class HumanGold:
    """Scarce ground truth. label_fn(rollout_A, rollout_B) -> (y_A, y_B) in {0,1}."""
    label_fn: Callable[[Rollout, Rollout], Tuple[int, int]]
    cost: float = 60.0
    name: str = "human"

    def label(self, ra: Rollout, rb: Rollout) -> Tuple[int, int]:
        return self.label_fn(ra, rb)
