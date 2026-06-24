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

"""Tests for the Sample Ratio Mismatch (SRM) health check in checks.py."""

from lerobot_doctor.checks import sample_ratio_mismatch


def test_balanced_split_not_flagged():
    d = sample_ratio_mismatch(500, 500)
    assert d.level == "info" and d.code == "srm_ok"
    assert d.data["p_value"] > 0.5                 # perfectly balanced -> p == 1


def test_mild_imbalance_not_flagged():
    """Ordinary sampling noise must NOT trip SRM (that is the whole point)."""
    d = sample_ratio_mismatch(510, 490)
    assert d.level == "info"


def test_severe_imbalance_flagged():
    d = sample_ratio_mismatch(50, 500)
    assert d.level == "warn" and d.code == "srm"
    assert d.data["p_value"] < 1e-3                # far beyond chance


def test_expected_ratio_respected():
    """A 30/70 split is fine when that is the INTENDED split, flagged when it is not."""
    ok = sample_ratio_mismatch(300, 700, expected=0.3)
    assert ok.level == "info"
    bad = sample_ratio_mismatch(700, 300, expected=0.3)
    assert bad.level == "warn"


def test_zero_trials_is_skipped():
    d = sample_ratio_mismatch(0, 0)
    assert d.level == "info" and d.data["p_value"] == 1.0
