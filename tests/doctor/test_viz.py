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
Tests for intervals + viz.

  I1  Clopper-Pearson: known values, monotone narrowing in N
  I2  betting confidence sequence: covers the truth, narrows, valid range
  C1  dashboard renders a non-empty PNG
"""

import os
import numpy as np
import pytest

from lerobot_doctor import compare_success
from lerobot_doctor.intervals import clopper_pearson, betting_confidence_sequence


def test_I1_clopper_pearson_basic():
    """Clopper-Pearson CI contains the expected interval bounds."""
    lo, hi = clopper_pearson(5, 10, 0.10)
    assert 0 < lo < 0.5 < hi < 1
    assert clopper_pearson(0, 10, 0.10)[0] == 0.0
    assert clopper_pearson(10, 10, 0.10)[1] == 1.0


def test_I1_narrows_with_n():
    """Clopper-Pearson CI narrows as sample size grows."""
    w_small = np.subtract(*clopper_pearson(5, 10, 0.10)[::-1])
    w_big = np.subtract(*clopper_pearson(50, 100, 0.10)[::-1])
    assert w_big < w_small


def test_I2_confidence_sequence_covers_and_narrows():
    """Anytime-valid CI covers the true effect and narrows over time."""
    rng = np.random.default_rng(0)
    # true effect = 0.75 - 0.55 = 0.20
    diffs = [int(rng.random() < 0.75) - int(rng.random() < 0.55) for _ in range(800)]
    t, lo, hi = betting_confidence_sequence(diffs, alpha=0.10)
    assert np.all(lo <= hi)
    assert np.all((lo >= -1.001) & (hi <= 1.001))
    # final interval should contain the truth and be much tighter than [-1,1]
    assert lo[-1] <= 0.20 <= hi[-1]
    assert (hi[-1] - lo[-1]) < (hi[len(hi)//8] - lo[len(lo)//8])   # narrows


def test_I2_cs_anytime_coverage():
    """Across many H0 runs, the all-times CI should miss 0 rarely (<= alpha-ish)."""
    misses = 0; T = 200
    for s in range(T):
        rng = np.random.default_rng(100 + s)
        diffs = [int(rng.random() < 0.6) - int(rng.random() < 0.6) for _ in range(300)]
        _, lo, hi = betting_confidence_sequence(diffs, alpha=0.10)
        if np.any((lo > 0) | (hi < 0)):        # 0 ever excluded = miss under H0
            misses += 1
    assert misses / T <= 0.10 + 0.05


def test_C1_dashboard_renders(tmp_path):
    """Dashboard renders a non-empty PNG file."""
    pytest.importorskip("matplotlib")
    rng = np.random.default_rng(1)
    A = [int(rng.random() < 0.75) for _ in range(400)]
    B = [int(rng.random() < 0.55) for _ in range(400)]
    rep = compare_success(A, B, alpha=0.10)
    from lerobot_doctor.viz import dashboard
    out = dashboard(rep, str(tmp_path / "r.png"))
    assert os.path.getsize(out) > 5000
