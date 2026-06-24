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

"""Tests for report.py -- the Statsig-style scorecard and serialization."""

import json

from lerobot_doctor.report import Report


def _mk(**kw):
    """An anytime-valid report (the default method) with a clear A>B decision."""
    base = dict(alpha=0.1, alternative="A>B", decided=True, direction="A>B",
                n_to_decision=10, n_pairs=10, n_A=10, n_B=10, successes_A=8,
                successes_B=4, rate_A=0.8, rate_B=0.4, effect=0.4, e_value=12.0,
                p_value_equiv=1 / 12.0, confidence_level=0.9,
                effect_ci_lo=0.05, effect_ci_hi=0.75, effect_relative=1.0,
                effect_relative_ci_lo=0.1, effect_relative_ci_hi=1.9,
                rate_A_ci=(0.5, 0.95), rate_B_ci=(0.15, 0.7),
                srm_p_value=1.0, chance_to_beat=0.99)
    base.update(kw)
    return Report(**base)


def _fixed(**kw):
    """A fixed-horizon report with the fixed-only statistics populated."""
    defaults = dict(horizon="fixed-horizon", test="proportion z-test",
                    method="fixed-horizon-proportion-z-test",
                    standard_error=0.02, z_score=4.0, p_value=0.0001,
                    power=0.9, mde=0.05)
    defaults.update(kw)
    return _mk(**defaults)


def test_anytime_valid_summary_has_decision_and_scorecard():
    r = _mk()
    s = r.summary()
    assert "anytime-valid" in s                 # header names the horizon
    assert "Decision: A>B" in s                  # verdict line
    assert "e-value" in s                        # anytime-valid idiom
    assert "Relative lift:" in s                 # Statsig headline number
    assert "Bayesian P(A>B):" in s
    assert "anytime-valid" in s.split("CI")[0] or "(anytime-valid)" in s


def test_anytime_valid_no_call_renders():
    r = _mk(decided=False, direction=None, n_to_decision=None, e_value=2.0,
            p_value_equiv=0.5)
    assert "NO CALL" in r.summary()


def test_fixed_horizon_summary_shows_pvalue_se_power():
    s = _fixed().summary()
    assert "fixed-horizon" in s
    assert "SIGNIFICANT" in s
    assert "p = 0.0001" in s
    assert "Standard error:" in s
    assert "Power:" in s and "MDE" in s          # fixed-horizon-only block


def test_fixed_horizon_not_significant_renders():
    s = _fixed(decided=False, direction=None, p_value=0.42).summary()
    assert "NOT SIGNIFICANT" in s


def test_to_dict_roundtrips_and_json_serialises():
    r = _fixed()
    d = r.to_dict()
    assert d["e_value"] == 12.0
    assert d["method"] == "fixed-horizon-proportion-z-test"
    assert d["standard_error"] == 0.02
    # The whole thing must be JSON-serialisable (no inf / numpy scalars leaking in).
    json.dumps(d)


def test_one_sided_open_ci_renders_infinity():
    """A one-sided CI (open upper side -> None) renders as +inf, not a crash."""
    s = _fixed(effect_ci_lo=0.05, effect_ci_hi=None,
               effect_relative_ci_lo=0.1, effect_relative_ci_hi=None).summary()
    assert "+inf" in s
