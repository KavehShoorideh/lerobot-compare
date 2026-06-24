"""
Tests for the robotics adapter that run WITHOUT a GPU, weights, or LeRobot data.

They prove the integration contract (evaluators -> paired predict -> core engine)
using a torch-free StubVLM, and that the lazy-import boundary holds so the
domain-agnostic core stays pure.
"""

import sys
import numpy as np
import pytest

import lerobot_doctor as core
import robotics_adapter as pr
from robotics_adapter import Rollout, Pair, StubVLM, InternalState


def _synth_pairs(n, pA, pB, rng):
    """Paired rollouts. meta['p'] is the hidden success prob the stub VLM keys
    off; meta['y'] is the realized gold label (read by the gold source). Stored
    per-rollout so the gold lookup never relies on value-equality of pairs."""
    pairs = []
    for _ in range(n):
        a = Rollout(task="put cube in bin", meta={"p": pA, "y": int(rng.random() < pA)})
        b = Rollout(task="put cube in bin", meta={"p": pB, "y": int(rng.random() < pB)})
        pairs.append(Pair(task="put cube in bin", A=a, B=b))
    return pairs


def _gold_from_meta(pair):
    return (pair.A.meta["y"], pair.B.meta["y"])


# --------------------------------------------------------------------------
# Boundary: importing the adapter must not pull heavy deps
# --------------------------------------------------------------------------

def test_import_does_not_pull_heavy_deps():
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules
    assert "lerobot" not in sys.modules


def test_rubricvlm_constructs_without_loading_weights():
    """Instantiating the real VLM evaluator must NOT load the model (lazy)."""
    ev = pr.RubricVLM(rubric=("Is the cube in the bin?",))
    assert ev._model is None        # nothing loaded until .score() is called
    assert ev.cost > 0
    assert "torch" not in sys.modules


# --------------------------------------------------------------------------
# End-to-end wiring through the real core engine, with a stub evaluator
# --------------------------------------------------------------------------

def test_stub_vlm_end_to_end_detects_better_policy():
    rng = np.random.default_rng(0)
    pairs = _synth_pairs(1500, pA=0.75, pB=0.55, rng=rng)
    gold = pr.human_gold(_gold_from_meta, cost=1.0)
    ev = StubVLM(rho=0.8, cost=1.0)
    r = pr.compare_policies(pairs, ev, gold, alpha=0.10,
                            rng=np.random.default_rng(1))
    assert r.decided and r.direction == "A>B"
    assert r.n_audited < r.n_seen           # cheap evaluator saved gold labels


def test_validity_under_h0_with_stub():
    """No real difference -> false positives controlled (validity is the core's,
    but this checks the adapter doesn't break it)."""
    T, alpha, fp = 120, 0.10, 0
    for t in range(T):
        rng = np.random.default_rng(100 + t)
        pairs = _synth_pairs(1000, pA=0.6, pB=0.6, rng=rng)
        gold = pr.human_gold(_gold_from_meta, cost=1.0)
        r = pr.compare_policies(pairs, StubVLM(rho=0.8, seed=t), gold,
                                alpha=alpha, rng=np.random.default_rng(500 + t))
        fp += int(r.decided)
    assert fp / T <= alpha + 3 * (alpha * (1 - alpha) / T) ** 0.5


# --------------------------------------------------------------------------
# Evaluator unit behavior
# --------------------------------------------------------------------------

def test_internal_state_route():
    ev = InternalState(fn=lambda ro: ro.meta["p"], cost=0.001)
    assert ev.score(Rollout(task="t", meta={"p": 0.9})) == 0.9
    assert ev.cost < 0.01


def test_rubric_combination_modes():
    """mean / product / depth combine per-subgoal probabilities correctly."""
    class FakeRubric(pr.RubricVLM):
        probs = (0.9, 0.9, 0.2)
        def _ensure_loaded(self):  # skip real model
            pass
        def _p_yes(self, task, frames, question):
            return self.probs[self.rubric.index(question)]

    rub = ("q1", "q2", "q3")
    mean = FakeRubric(rubric=rub, combine="mean").score(Rollout("t"))
    prod = FakeRubric(rubric=rub, combine="product").score(Rollout("t"))
    depth = FakeRubric(rubric=rub, combine="depth").score(Rollout("t"))
    assert abs(mean - (0.9 + 0.9 + 0.2) / 3) < 1e-9
    assert abs(prod - 0.9 * 0.9 * 0.2) < 1e-9
    assert abs(depth - 2 / 3) < 1e-9          # two leading subgoals pass, third fails


def test_stub_vlm_quality_tracks_rho():
    """A higher-rho stub correlates better with the hidden truth."""
    rng = np.random.default_rng(3)
    good = StubVLM(rho=0.95, seed=1)
    bad = StubVLM(rho=0.5, seed=2)
    ps = [0.0] * 200 + [1.0] * 200
    g = [good.score(Rollout("t", meta={"p": p})) for p in ps]
    b = [bad.score(Rollout("t", meta={"p": p})) for p in ps]
    # correlation of score with truth: good >> bad
    cg = np.corrcoef(g, ps)[0, 1]
    cb = np.corrcoef(b, ps)[0, 1]
    assert cg > cb
