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
Tests for the fixed-horizon method: the paired/two-proportion z-tests, power/MDE,
and the `compare_success(method="fixed-horizon-proportion-z-test")` integration.

Mirrors the core F1 validity philosophy: the headline statistical guarantee is that
type-I error stays at or below alpha (here at the fixed sample size). Monte-Carlo
checks use fixed seeds, enough replicates, and a binomial-SE tolerance margin.
"""

import math

import numpy as np
import pytest
from scipy.stats import norm

from lerobot_doctor import (
    available_methods, compare_success, min_detectable_effect,
    paired_z_test, two_proportion_z_test, Trial,
)

FIXED = "fixed-horizon-proportion-z-test"
EPROC = "anytime-valid-betting-e-process"
ALPHA = 0.10


# ---------------------------------------------------------------- unit: stats
def test_paired_z_hand_calc():
    """effect/SE/z match a by-hand computation on a tiny fixture."""
    diffs = [1, 1, 1, 1, 0, 0, 0, 0, -1, -1]           # mean = 0.2
    s = paired_z_test(diffs, alpha=0.10, alternative="two-sided")
    se = math.sqrt((5.6 / 9) / 10)                     # var(ddof=1)=5.6/9; SE=sqrt(var/n)
    assert s.effect == pytest.approx(0.2)
    assert s.se == pytest.approx(se, abs=1e-9)
    assert s.z == pytest.approx(0.2 / se, abs=1e-9)
    assert s.p_value == pytest.approx(2 * norm.sf(abs(s.z)), abs=1e-9)


def test_two_proportion_matches_textbook():
    """Classic two-proportion z-test against a known value (pooled-SE statistic)."""
    s = two_proportion_z_test(30, 100, 20, 100, alpha=0.10, alternative="two-sided")
    assert s.effect == pytest.approx(0.10)
    assert s.z == pytest.approx(1.6330, abs=1e-3)
    assert s.p_value == pytest.approx(0.10247, abs=1e-4)


def test_one_sided_ci_has_open_side():
    """A one-sided alternative leaves the non-tested CI side open (None)."""
    up = paired_z_test([1, 1, 0, 1, 0, 1], alternative="A>B")
    assert up.ci_hi is None and up.ci_lo is not None
    down = paired_z_test([1, 1, 0, 1, 0, 1], alternative="B>A")
    assert down.ci_lo is None and down.ci_hi is not None


def test_mde_shrinks_with_more_data_and_power_grows_with_effect():
    se_small_n, se_large_n = 0.05, 0.01
    assert min_detectable_effect(se_large_n, ALPHA) < min_detectable_effect(se_small_n, ALPHA)
    from lerobot_doctor.fixed_horizon import achieved_power
    assert achieved_power(0.10, 0.02, ALPHA) > achieved_power(0.02, 0.02, ALPHA)


def test_degenerate_no_variance_does_not_crash():
    """All-identical diffs -> SE 0; must not raise or emit nan."""
    s = paired_z_test([1, 1, 1, 1], alternative="two-sided")
    assert s.se == 0.0 and math.isfinite(s.p_value)


# ------------------------------------------------------- integration: dispatch
def _paired_world(pA, pB, n, seed):
    """Paired trials sharing a scene key (clean exact match), with given rates."""
    rng = np.random.default_rng(seed)
    A = [Trial(success=int(rng.random() < pA), key=k) for k in range(n)]
    B = [Trial(success=int(rng.random() < pB), key=k) for k in range(n)]
    return A, B


def test_methods_registry_lists_both():
    assert set(available_methods()) == {FIXED, EPROC}


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown method"):
        compare_success([1, 0], [0, 1], method="nope")


def test_fixed_recovers_strong_effect():
    A, B = _paired_world(0.75, 0.45, 400, seed=3)
    r = compare_success(A, B, method=FIXED, alpha=0.05, rng=np.random.default_rng(1))
    assert r.decided and r.direction == "A>B"
    assert r.p_value is not None and r.p_value < 0.05
    assert r.horizon == "fixed-horizon"
    # the absolute-effect CI excludes zero when significant (two-sided)
    assert r.effect_ci_lo is not None and r.effect_ci_lo > 0
    # scorecard extras are populated for the fixed method
    assert r.standard_error is not None and r.power is not None and r.mde is not None
    assert r.chance_to_beat is not None and r.rate_A_ci is not None


def test_eprocess_threshold_and_pvalue_account_for_two_sidedness():
    """Two-sided e-process spends alpha/2 per side: honest bar is 2/alpha and the
    anytime-valid p-value is min(1, 2/e_value); one-sided uses 1/alpha and 1/e_value."""
    A, B = _paired_world(0.7, 0.5, 300, seed=8)
    two = compare_success(A, B, method=EPROC, alpha=0.05, rng=np.random.default_rng(1))
    one = compare_success(A, B, method=EPROC, alpha=0.05, alternative="A>B",
                          rng=np.random.default_rng(1))
    assert two.e_threshold == pytest.approx(2 / 0.05)        # = 40
    assert one.e_threshold == pytest.approx(1 / 0.05)        # = 20
    assert two.p_value_equiv == pytest.approx(min(1.0, 2 / two.e_value))
    assert one.p_value_equiv == pytest.approx(min(1.0, 1 / one.e_value))


def test_both_methods_share_report_shape():
    A, B = _paired_world(0.7, 0.5, 300, seed=5)
    for m in (FIXED, EPROC):
        r = compare_success(A, B, method=m, alpha=0.05, rng=np.random.default_rng(2))
        assert r.method == m
        assert r.effect_relative is not None        # relative lift for both
        assert r.rate_A_ci and r.rate_B_ci          # per-variant CIs for both
        assert r.srm_p_value is not None            # SRM for both
        assert r.chance_to_beat is not None         # Bayesian for both


# ----------------------------------------------------------- validity (slow)
@pytest.mark.slow
def test_fixed_type_i_at_or_below_alpha():
    """A/A (pA = pB): empirical significant-rate <= alpha within MC tolerance.

    For a two-sided test, "CI excludes 0" == "significant", so this simultaneously
    checks confidence-interval coverage.
    """
    trials, n, p = 300, 500, 0.5
    sig = 0
    for t in range(trials):
        A, B = _paired_world(p, p, n, seed=10_000 + t)
        r = compare_success(A, B, method=FIXED, alpha=ALPHA,
                            rng=np.random.default_rng(20_000 + t))
        sig += int(r.decided)
    rate = sig / trials
    se = math.sqrt(ALPHA * (1 - ALPHA) / trials)        # binomial SE of the estimate
    assert rate <= ALPHA + 3 * se
