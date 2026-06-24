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

"""Tests for the Bayesian chance-to-beat readout (Beta-Binomial posterior)."""

import numpy as np
import pytest

from lerobot_doctor.bayesian import chance_to_beat


def test_symmetric_data_is_a_coin_flip():
    """Identical data for A and B -> ~0.5 (no posterior preference)."""
    p = chance_to_beat(40, 100, 40, 100, rng=np.random.default_rng(0))
    assert p == pytest.approx(0.5, abs=0.05)


def test_no_data_is_a_coin_flip():
    """With the uniform prior and no observations, A beating B is ~0.5 by symmetry."""
    assert chance_to_beat(0, 0, 0, 0, rng=np.random.default_rng(0)) == pytest.approx(0.5, abs=0.05)


def test_strong_A_dominates():
    assert chance_to_beat(90, 100, 10, 100, rng=np.random.default_rng(0)) > 0.99


def test_strong_B_dominates():
    assert chance_to_beat(10, 100, 90, 100, rng=np.random.default_rng(0)) < 0.01


def test_bounded_and_deterministic_under_fixed_rng():
    a = chance_to_beat(55, 100, 45, 100, rng=np.random.default_rng(7))
    b = chance_to_beat(55, 100, 45, 100, rng=np.random.default_rng(7))
    assert 0.0 <= a <= 1.0
    assert a == b                                  # same seed -> identical estimate
