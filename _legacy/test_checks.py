"""
Tests for checks.py -- the built-in success-flag policy.

  F1  a flag column triggers an ignore+warn diagnostic by default
  F2  use_as='evaluator' / 'human_gold' switch to info diagnostics
  F3  no flag-like column -> no diagnostics
  F4  preflight_rollouts warns when no frames are present
  F5  emit() raises a real Python warning for warn-level diagnostics
"""

import warnings
import pytest

from lerobot_doctor.checks import (
    scan_success_flags, preflight_rollouts, emit, Diagnostic,
)
from lerobot_doctor import Rollout

COLS = ["observation.images.top", "action", "timestamp", "success", "reward"]


def test_F1_flag_ignored_by_default():
    d = scan_success_flags(COLS)
    assert len(d) == 1 and d[0].level == "warn"
    assert d[0].code == "builtin_flag_ignored"
    assert "IGNORED" in d[0].message


def test_F2_optin_modes():
    assert scan_success_flags(COLS, use_as="evaluator")[0].code == "builtin_flag_as_evaluator"
    assert scan_success_flags(COLS, use_as="human_gold")[0].code == "builtin_flag_as_gold"
    assert scan_success_flags(COLS, use_as="evaluator")[0].level == "info"


def test_F3_no_flag_no_diag():
    assert scan_success_flags(["observation.images.top", "action", "timestamp"]) == []


def test_F4_no_frames_warns():
    rolls = [Rollout(task="t", key=i, meta={"success": 1}) for i in range(5)]
    diags = preflight_rollouts(rolls)
    codes = {d.code for d in diags}
    assert "builtin_flag_ignored" in codes      # meta carried a flag
    assert "no_frames" in codes                 # and no frames to judge


def test_F4_frames_present_no_frame_warning():
    rolls = [Rollout(task="t", key=i, frames=["a.png"], meta={}) for i in range(5)]
    diags = preflight_rollouts(rolls)
    assert "no_frames" not in {d.code for d in diags}


def test_F5_emit_raises_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        emit([Diagnostic("warn", "x", "hello")])
        assert len(w) == 1 and "hello" in str(w[0].message)
