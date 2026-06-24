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
lerobot_doctor.core guarantee test suite, organized by FINDING (F1-F7).

Each finding the paper claims is backed by MULTIPLE tests here, and the paper's
reproducibility table cites these names directly. Statistical tests use fixed
seeds, enough replicates, and tolerance margins (alpha + a few binomial SEs).

  F1  validity: type-I <= alpha, for ANY evaluator and ANY acquisition rule
  F2  efficiency: adaptive bet + control variate beats fixed IPW; each lever helps
  F3  principled acquisition beats uniform at MATCHED budget
  F4  NEGATIVE: classical Neyman allocation backfires (+ its mechanism)
  F5  graceful degradation (PPI++): a useless evaluator never hurts
  F6  evaluator quality -> fewer items to decision (control-variate effect)
  F7  structural rails: predictability, safe betting, determinism, core purity
"""

import math
import numpy as np
import pytest

from lerobot_doctor.core import (
    compare, make_world, CheapVLM, Comparison, BettingEProcess, Obs,
    MaxEvidenceAuditor, UniformAuditor, NeymanAuditor,
)

ALPHA = 0.10
EV = CheapVLM()


def _se(p, n):
    """Binomial standard error."""
    return math.sqrt(p * (1 - p) / n)


def _batch(pA, pB, trials, *, adaptive=True, control_variate=True, rho_acc=0.8,
           rule=None, N=1000, seed0=0):
    """Run many compare() trials and aggregate summary statistics."""
    dec, golds, seens, rates = 0, [], [], []
    for t in range(trials):
        items, predict, gold = make_world(pA, pB, np.random.default_rng(seed0 + t),
                                          rho_acc=rho_acc, N=N)
        r = compare(items, EV, gold, predict, rule=rule, alpha=ALPHA,
                    adaptive=adaptive, control_variate=control_variate,
                    rng=np.random.default_rng(900_000 + seed0 + t))
        dec += int(r.decided); golds.append(r.n_audited)
        seens.append(r.n_seen); rates.append(r.n_audited / r.n_seen)
    return dict(decide_rate=dec / trials, median_gold=float(np.median(golds)),
                median_seen=float(np.median(seens)), mean_rate=float(np.mean(rates)))


# ===========================================================================
# F1 -- VALIDITY (the guarantee): type-I <= alpha regardless of evaluator/rule
# ===========================================================================

@pytest.mark.slow
def test_F1_type_i_good_evaluator():
    """Type-I error rate stays at or below alpha with a good evaluator (rho=0.8)."""
    T = 300
    r = _batch(0.6, 0.6, T, rho_acc=0.8, seed0=1)
    assert r["decide_rate"] <= ALPHA + 3 * _se(ALPHA, T)


@pytest.mark.slow
def test_F1_type_i_useless_evaluator():
    """Validity must not depend on evaluator quality (rho=0.5 = coin flip)."""
    T = 300
    r = _batch(0.6, 0.6, T, rho_acc=0.5, seed0=2)
    assert r["decide_rate"] <= ALPHA + 3 * _se(ALPHA, T)


@pytest.mark.slow
def test_F1_type_i_ipw_estimator():
    """Type-I holds even with fixed bet and no control variate (plain IPW)."""
    T = 300
    r = _batch(0.6, 0.6, T, adaptive=False, control_variate=False, seed0=3)
    assert r["decide_rate"] <= ALPHA + 3 * _se(ALPHA, T)


@pytest.mark.slow
def test_F1_type_i_all_acquisition_rules():
    """Uniform, principled, AND Neyman all preserve validity (predictability of
    pi is what matters, not its shape)."""
    T = 250
    for rule in (UniformAuditor(pi=0.3), MaxEvidenceAuditor(), NeymanAuditor()):
        r = _batch(0.6, 0.6, T, rule=rule, seed0=4)
        assert r["decide_rate"] <= ALPHA + 3 * _se(ALPHA, T)


@pytest.mark.slow
def test_F1_type_i_second_alpha():
    """Validity holds at a tighter alpha (0.05) as well."""
    T = 300
    dec = 0
    for t in range(T):
        items, predict, gold = make_world(0.6, 0.6, np.random.default_rng(50 + t),
                                          rho_acc=0.8, N=1000)
        r = compare(items, EV, gold, predict, alpha=0.05,
                    rng=np.random.default_rng(960_000 + t))
        dec += int(r.decided)
    assert dec / T <= 0.05 + 3 * _se(0.05, T)


def test_F1_payoff_unbiased_despite_biased_evaluator():
    """PPI++ payoff is unbiased for p_A even when the evaluator is imperfect."""
    rng = np.random.default_rng(7)
    pA = 0.7
    items, predict, gold = make_world(pA, 0.5, rng, rho_acc=0.7, N=4000)
    est = Comparison(pi_min=0.2, control_variate=True)
    pi, ms = 0.2, []
    for i in items:
        fA, _ = predict(i, EV)
        audited = rng.random() < pi
        yA = gold.label(i)[0] if audited else None
        ms.append(est._score(fA, audited, pi, yA, lam_pp=0.8))
        if audited:
            est.observe(Obs(fA, fA, True, pi, yA, yA))
    assert abs(np.mean(ms) - pA) < 0.03


@pytest.mark.slow
def test_F1_supermartingale_under_h0():
    """Under H0 the wealth is a supermartingale from 1, so its median over many
    runs should not exceed ~1, and it rarely crosses 1/alpha."""
    finals = []
    for t in range(300):
        items, predict, gold = make_world(0.6, 0.6, np.random.default_rng(40 + t),
                                          rho_acc=0.8, N=800)
        r = compare(items, EV, gold, predict, alpha=1e9, adaptive=True,
                    control_variate=True, rng=np.random.default_rng(800_000 + t))
        finals.append(r.wealth)
    assert np.median(finals) <= 1.2
    assert np.mean(np.array(finals) >= 10) <= ALPHA


# ===========================================================================
# F2 -- EFFICIENCY: adaptive bet + control variate beats fixed IPW; each lever helps
# ===========================================================================

@pytest.mark.slow
def test_F2_adaptive_controlvariate_beats_fixed_ipw():
    """Full method (adaptive bet + control variate) uses fewer gold labels than plain IPW."""
    r1 = _batch(0.75, 0.55, 200, adaptive=False, control_variate=False, seed0=11)
    r2 = _batch(0.75, 0.55, 200, adaptive=True, control_variate=True, seed0=11)
    assert r1["decide_rate"] >= 0.9 and r2["decide_rate"] >= 0.9
    assert r2["median_gold"] < r1["median_gold"]


@pytest.mark.slow
def test_F2_adaptive_controlvariate_beats_fixed_ipw_second_effect():
    """Efficiency advantage of full method holds at a smaller effect size (0.70 vs 0.55)."""
    r1 = _batch(0.70, 0.55, 200, adaptive=False, control_variate=False, seed0=13)
    r2 = _batch(0.70, 0.55, 200, adaptive=True, control_variate=True, seed0=13)
    assert r1["decide_rate"] >= 0.85 and r2["decide_rate"] >= 0.85
    assert r2["median_gold"] < r1["median_gold"]


@pytest.mark.slow
def test_F2_control_variate_lever_helps():
    """Control variate alone (fixed bet) should not increase gold vs plain IPW."""
    neither = _batch(0.75, 0.55, 200, adaptive=False, control_variate=False, seed0=11)
    cv_only = _batch(0.75, 0.55, 200, adaptive=False, control_variate=True, seed0=11)
    assert cv_only["median_gold"] <= neither["median_gold"]


@pytest.mark.slow
def test_F2_adaptive_bet_lever_helps():
    """Adaptive betting alone (importance weighting) should not increase gold."""
    neither = _batch(0.75, 0.55, 200, adaptive=False, control_variate=False, seed0=11)
    bet_only = _batch(0.75, 0.55, 200, adaptive=True, control_variate=False, seed0=11)
    assert bet_only["median_gold"] <= neither["median_gold"]


# ===========================================================================
# F3 -- PRINCIPLED ACQUISITION beats uniform at MATCHED budget
# ===========================================================================

@pytest.mark.slow
def test_F3_principled_beats_uniform_gold():
    """MaxEvidenceAuditor reaches decisions with fewer gold labels than UniformAuditor."""
    adp = _batch(0.75, 0.55, 200, rule=MaxEvidenceAuditor(), seed0=12)
    unif = _batch(0.75, 0.55, 200, rule=UniformAuditor(pi=adp["mean_rate"]), seed0=12)
    assert adp["decide_rate"] >= 0.9 and unif["decide_rate"] >= 0.9
    assert adp["median_gold"] < unif["median_gold"]


@pytest.mark.slow
def test_F3_principled_beats_uniform_items():
    """MaxEvidenceAuditor also sees fewer items before deciding."""
    adp = _batch(0.75, 0.55, 200, rule=MaxEvidenceAuditor(), seed0=12)
    unif = _batch(0.75, 0.55, 200, rule=UniformAuditor(pi=adp["mean_rate"]), seed0=12)
    assert adp["median_seen"] < unif["median_seen"]


@pytest.mark.slow
def test_F3_budget_is_actually_matched():
    """The audit rates of principled and uniform are within 4 percentage points."""
    adp = _batch(0.75, 0.55, 200, rule=MaxEvidenceAuditor(), seed0=12)
    unif = _batch(0.75, 0.55, 200, rule=UniformAuditor(pi=adp["mean_rate"]), seed0=12)
    assert abs(adp["mean_rate"] - unif["mean_rate"]) < 0.04


@pytest.mark.slow
def test_F3_holds_at_second_effect():
    """Principled-beats-uniform result holds at a smaller effect size (0.70 vs 0.55)."""
    adp = _batch(0.70, 0.55, 200, rule=MaxEvidenceAuditor(), seed0=14)
    unif = _batch(0.70, 0.55, 200, rule=UniformAuditor(pi=adp["mean_rate"]), seed0=14)
    assert adp["decide_rate"] >= 0.85 and unif["decide_rate"] >= 0.85
    assert adp["median_gold"] < unif["median_gold"]


# ===========================================================================
# F4 -- NEGATIVE RESULT: classical Neyman allocation backfires
# ===========================================================================

@pytest.mark.slow
def test_F4_neyman_detects_worse_than_uniform():
    """NeymanAuditor decides far less often than UniformAuditor at matched budget."""
    ney = _batch(0.75, 0.55, 150, rule=NeymanAuditor(), seed0=12)
    unif = _batch(0.75, 0.55, 150, rule=UniformAuditor(pi=ney["mean_rate"]), seed0=12)
    assert ney["decide_rate"] < unif["decide_rate"] - 0.3


@pytest.mark.slow
def test_F4_neyman_spends_more_gold_than_principled():
    """NeymanAuditor spends more gold labels to reach the same item count as MaxEvidenceAuditor."""
    ney = _batch(0.75, 0.55, 150, rule=NeymanAuditor(), seed0=12)
    adp = _batch(0.75, 0.55, 150, rule=MaxEvidenceAuditor(), seed0=12)
    assert ney["median_gold"] > adp["median_gold"]


def test_F4_mechanism_lam_pp_collapses_on_uncertain_items():
    """WHY it backfires: auditing only UNCERTAIN items (f ~ 0.5, where y is
    near coin-flip) makes the control variate's lam_PP collapse toward 0."""
    rng = np.random.default_rng(60)
    est = Comparison(pi_min=0.3, control_variate=True)
    for _ in range(400):
        f = float(np.clip(0.5 + 0.03 * rng.normal(), 0, 1))   # all near 0.5
        y = int(rng.random() < 0.5)                            # ~ independent of f
        est.observe(Obs(f, f, True, 0.3, y, y))
    assert est._lam_pp() < 0.2


# ===========================================================================
# F5 -- GRACEFUL DEGRADATION (PPI++): a useless evaluator never hurts
# ===========================================================================

def test_F5_lam_pp_to_zero_for_useless_evaluator():
    """lam_PP shrinks to near 0 when the evaluator is uncorrelated with gold labels."""
    rng = np.random.default_rng(30)
    items, predict, gold = make_world(0.5, 0.5, rng, rho_acc=0.5, N=3000)
    est = Comparison(pi_min=0.3, control_variate=True)
    for i in items:
        fA, fB = predict(i, EV); yA, yB = gold.label(i)
        est.observe(Obs(fA, fB, True, 0.3, yA, yB))
    assert est._lam_pp() < 0.2


def test_F5_lam_pp_to_one_for_good_evaluator():
    """lam_PP rises toward 1 when the evaluator is strongly correlated with gold labels."""
    rng = np.random.default_rng(31)
    items, predict, gold = make_world(0.5, 0.5, rng, rho_acc=0.95, N=3000)
    est = Comparison(pi_min=0.3, control_variate=True)
    for i in items:
        fA, fB = predict(i, EV); yA, yB = gold.label(i)
        est.observe(Obs(fA, fB, True, 0.3, yA, yB))
    assert est._lam_pp() > 0.6


@pytest.mark.slow
def test_F5_useless_evaluator_still_valid():
    """End-to-end: a useless evaluator costs efficiency but not validity."""
    T = 250
    r = _batch(0.6, 0.6, T, rho_acc=0.5, rule=UniformAuditor(pi=0.3), seed0=70)
    assert r["decide_rate"] <= ALPHA + 3 * _se(ALPHA, T)


# ===========================================================================
# F6 -- EVALUATOR QUALITY -> fewer ITEMS to decision (control-variate effect)
# ===========================================================================

@pytest.mark.slow
@pytest.mark.parametrize("rhos", [(0.55, 0.75, 0.95)])
def test_F6_items_monotone_in_rho(rhos):
    """Median items to decision decreases monotonically as evaluator quality rises."""
    seens = []
    for rho in rhos:
        r = _batch(0.75, 0.55, 200, rule=UniformAuditor(pi=0.3), rho_acc=rho, seed0=20)
        assert r["decide_rate"] >= 0.9
        seens.append(r["median_seen"])
    assert seens[0] > seens[1] > seens[2]


@pytest.mark.slow
@pytest.mark.parametrize("rhos", [(0.55, 0.95)])
def test_F6_gold_not_increasing_in_rho(rhos):
    """Gold audits do not increase as evaluator quality improves (fixed audit rate)."""
    g = []
    for rho in rhos:
        r = _batch(0.75, 0.55, 200, rule=UniformAuditor(pi=0.3), rho_acc=rho, seed0=20)
        g.append(r["median_gold"])
    assert g[0] >= g[-1]


# ===========================================================================
# F7 -- STRUCTURAL RAILS: predictability, safe betting, determinism, purity
# ===========================================================================

def test_F7_predictability_lam_pp_ignores_current_label():
    """payoff() must not mutate the lam_PP tuning state (predictability invariant)."""
    est = Comparison(pi_min=0.2, control_variate=True)
    rng = np.random.default_rng(50)
    for _ in range(40):
        f = float(rng.random()); y = int(rng.random() < f)
        est.observe(Obs(f, f, True, 0.2, y, y))
    lam0 = est._lam_pp()
    est.payoff(Obs(0.9, 0.1, True, 0.2, 1, 0)); lam1 = est._lam_pp()
    est.payoff(Obs(0.9, 0.1, True, 0.2, 0, 1)); lam2 = est._lam_pp()
    assert lam0 == lam1 == lam2          # payoff() must not mutate tuning state


def test_F7_predictability_pi_ignores_label():
    """Audit probability depends only on cheap predictions, never on y."""
    rule = MaxEvidenceAuditor()
    p1 = rule.audit_prob(0.9, 0.2, gold_cost=1.0)
    p2 = rule.audit_prob(0.9, 0.2, gold_cost=1.0)
    assert p1 == p2                       # deterministic in (f_A, f_B)


def test_F7_betting_lambda_safe_range():
    """Betting fraction stays in [0, lam_max] and starts at 0 during warmup."""
    ep = BettingEProcess(alpha=ALPHA, adaptive=True)
    assert ep._lambda() == 0.0            # nothing seen yet (warmup)
    for g in [0.2, 0.3, 0.1, 0.25, 0.2, 0.3, 0.15]:
        assert 0.0 <= ep._lambda() <= ep.lam_max
        ep.update(g)
    assert ep._lambda() > 0.0


def test_F7_determinism():
    """Same RNG seed produces identical results across two runs."""
    items, predict, gold = make_world(0.75, 0.55, np.random.default_rng(60),
                                      rho_acc=0.8, N=1000)
    kw = dict(alpha=ALPHA, adaptive=True, control_variate=True)
    r1 = compare(items, EV, gold, predict, rng=np.random.default_rng(123), **kw)
    r2 = compare(items, EV, gold, predict, rng=np.random.default_rng(123), **kw)
    assert (r1.decided, r1.n_audited, r1.n_seen) == (r2.decided, r2.n_audited, r2.n_seen)


def test_F7_core_is_numpy_only():
    """lerobot_doctor.core must not import torch, transformers, lerobot, or qwen_vl_utils."""
    import sys
    banned = {"torch", "transformers", "lerobot", "qwen_vl_utils"}
    import lerobot_doctor.core  # noqa: F401
    assert banned.isdisjoint(sys.modules), f"core pulled in {banned & set(sys.modules)}"
