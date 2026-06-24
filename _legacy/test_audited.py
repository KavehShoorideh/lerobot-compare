"""
enhanced test suite (cheap evaluator + calibration + handoff). Stub evaluator path only
(no torch / no GPU), so it runs fully in-sandbox.

  W1  validity: prediction-powered e-process keeps type-I <= alpha, even with a
      USELESS evaluator (rho=0.5)
  W2  savings: under a true gap, decides while auditing far fewer than all pairs
  W3  graceful degradation: lam_PP -> ~0 useless evaluator, -> high for a good one
  W4  recalibration: isotonic map improves calibration error of a biased evaluator
  W5  handoff: LLaMA-Factory / SkyPilot backends write a valid dataset + config
"""

import json
import math
import os
import numpy as np
import pytest

from lerobot_doctor import (
    compare_policies, Rollout, HumanGold, StubVLM,
    Recalibrate, LabelRecord, TrainingConfig, LlamaFactoryBackend, SkyPilotBackend,
)

ALPHA = 0.10


def _se(p, n):
    return math.sqrt(p * (1 - p) / n)


def _world(pA, pB, n, rng):
    """Paired rollouts on shared instances; meta['p'] is the hidden truth the
    StubVLM reads, meta['y'] is the realized success the human would confirm."""
    A, B = [], []
    for k in range(n):
        base = rng.random()
        A.append(Rollout(task="t", key=k,
                         meta={"p": pA, "y": int(base < pA)}))
        B.append(Rollout(task="t", key=k,
                         meta={"p": pB, "y": int(base < pB)}))
    return A, B


def _gold():
    return HumanGold(lambda ra, rb: (ra.meta["y"], rb.meta["y"]), cost=60.0)


# ---- W1 validity ----------------------------------------------------------

def test_W1_type_i_good_evaluator():
    T, dec = 250, 0
    for t in range(T):
        A, B = _world(0.6, 0.6, 400, np.random.default_rng(t))
        r = compare_policies(A, B, StubVLM(rho=0.85), _gold(), alpha=ALPHA,
                             rng=np.random.default_rng(5000 + t))
        dec += int(r.decided)
    assert dec / T <= ALPHA + 3 * _se(ALPHA, T)


def test_W1_type_i_useless_evaluator():
    T, dec = 250, 0
    for t in range(T):
        A, B = _world(0.6, 0.6, 400, np.random.default_rng(100 + t))
        r = compare_policies(A, B, StubVLM(rho=0.5), _gold(), alpha=ALPHA,
                             rng=np.random.default_rng(5100 + t))
        dec += int(r.decided)
    assert dec / T <= ALPHA + 3 * _se(ALPHA, T)


# ---- W2 power + savings ---------------------------------------------------

def test_W2_detects_and_saves_labels():
    T, dec, saved = 150, 0, []
    for t in range(T):
        A, B = _world(0.75, 0.55, 1500, np.random.default_rng(200 + t))
        r = compare_policies(A, B, StubVLM(rho=0.85), _gold(), alpha=ALPHA,
                             alternative="A>B", rng=np.random.default_rng(5200 + t))
        dec += int(r.decided and r.direction == "A>B")
        if r.decided:
            saved.append(r.gold_saved_frac)
    assert dec / T >= 0.9
    assert np.median(saved) > 0.3            # audited well under "everything"


# ---- W3 graceful degradation ----------------------------------------------

def test_W3_lam_pp_low_for_useless():
    A, B = _world(0.6, 0.6, 2000, np.random.default_rng(7))
    r = compare_policies(A, B, StubVLM(rho=0.5), _gold(), alpha=ALPHA,
                         rng=np.random.default_rng(9))
    assert r.lam_pp < 0.3


def test_W3_lam_pp_higher_for_good():
    A, B = _world(0.6, 0.6, 2000, np.random.default_rng(8))
    r_good = compare_policies(A, B, StubVLM(rho=0.95), _gold(), alpha=ALPHA,
                              rng=np.random.default_rng(9))
    r_bad = compare_policies(A, B, StubVLM(rho=0.5), _gold(), alpha=ALPHA,
                             rng=np.random.default_rng(9))
    assert r_good.lam_pp > r_bad.lam_pp


# ---- W4 recalibration -----------------------------------------------------

def test_W4_isotonic_improves_calibration():
    rng = np.random.default_rng(11)
    # biased evaluator: reported score is squashed vs true success prob
    recs, raw_err, n = [], 0.0, 600
    for _ in range(n):
        p = rng.random()
        y = int(rng.random() < p)
        f = float(np.clip(p ** 2, 0, 1))            # systematically miscalibrated
        recs.append(LabelRecord(f=f, y=y))
        raw_err += abs(f - p)
    base = StubVLM(rho=0.8)
    cal = Recalibrate(base).update(recs)
    # calibration error of the isotonic map vs the raw score, on held-out grid
    from lerobot_doctor.calibrate import _pava, _Isotonic
    kx, ky = _pava(np.array([r.f for r in recs]), np.array([r.y for r in recs]))
    iso = _Isotonic(kx, ky)
    grid_p = np.linspace(0.05, 0.95, 50)
    raw = np.mean([abs(p**2 - p) for p in grid_p])
    cal_e = np.mean([abs(iso(p**2) - p) for p in grid_p])
    assert cal_e < raw


# ---- W5 handoff artifacts -------------------------------------------------

def _records(n=30):
    return [LabelRecord(f=0.6, y=i % 2, frames=[f"/tmp/ep{i}.png"],
                        question="Is the cube in the bin?") for i in range(n)]


def test_W5_llamafactory_writes_valid_artifacts(tmp_path):
    cfg = TrainingConfig(workdir=str(tmp_path), output_dir=str(tmp_path / "out"))
    res = LlamaFactoryBackend(cfg, dry_run=True).update(_records())
    assert not res.ran
    assert os.path.exists(res.dataset_path) and os.path.exists(res.config_path)
    # dataset is valid JSONL with the expected shape
    lines = open(res.dataset_path).read().strip().splitlines()
    assert len(lines) == 30
    ex = json.loads(lines[0])
    assert ex["messages"][1]["content"] in ("yes", "no")
    assert res.command[:2] == ["llamafactory-cli", "train"]
    assert os.path.exists(os.path.join(str(tmp_path), "dataset_info.json"))


def test_W5_skypilot_wraps_job(tmp_path):
    cfg = TrainingConfig(workdir=str(tmp_path), output_dir=str(tmp_path / "out"))
    res = SkyPilotBackend(cfg, accelerator="A100:1", dry_run=True).update(_records())
    assert res.command[0] == "sky" and res.command[1] == "launch"
    task = open(res.config_path).read()
    assert "accelerators: A100:1" in task and "llamafactory-cli train" in task
