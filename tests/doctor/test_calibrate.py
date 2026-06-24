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

import numpy as np
from lerobot_doctor.evaluators import StubVLM, Recalibrate, CalibratedEvaluator, LabelRecord, Rollout


def test_isotonic_is_monotone_and_clipped():
    """Recalibrate produces a monotone, clipped calibration curve."""
    rng = np.random.default_rng(1)
    recs = []
    for _ in range(200):
        f = rng.random()
        y = int(rng.random() < f)  # f correlates with y
        recs.append(LabelRecord(f=f, y=y))
    cal = Recalibrate(StubVLM(), min_points=20).update(recs)
    assert isinstance(cal, CalibratedEvaluator)
    xs = np.linspace(0, 1, 50)
    ys = [cal.cal(x) for x in xs]
    assert all(0.0 <= v <= 1.0 for v in ys)
    assert all(ys[i] <= ys[i + 1] + 1e-9 for i in range(len(ys) - 1))


def test_recalibrate_returns_base_below_min_points():
    """Recalibrate returns the base evaluator unchanged when data is insufficient."""
    base = StubVLM()
    out = Recalibrate(base, min_points=20).update([LabelRecord(f=0.5, y=1)] * 5)
    assert out is base
