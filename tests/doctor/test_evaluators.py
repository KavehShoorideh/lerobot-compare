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
from lerobot_doctor.evaluators import StubVLM, AutomaticChecker, InternalState, HumanGold, Rollout


def test_stubvlm_useless_when_rho_half():
    """StubVLM with rho=0.5 produces scores in [0, 1]."""
    ev = StubVLM(rho=0.5)
    r = Rollout(meta={"p": 0.9})
    scores = [ev.score(r) for _ in range(200)]
    assert 0.0 <= min(scores) and max(scores) <= 1.0


def test_internal_and_automatic_clip():
    """AutomaticChecker and InternalState clip out-of-range fn outputs."""
    assert AutomaticChecker(fn=lambda r: 2.0).score(Rollout()) == 1.0
    assert InternalState(fn=lambda r: -1.0).score(Rollout()) == 0.0


def test_human_gold_label():
    """HumanGold.label delegates to label_fn and returns the pair."""
    g = HumanGold(lambda a, b: (a.meta["y"], b.meta["y"]))
    assert g.label(Rollout(meta={"y": 1}), Rollout(meta={"y": 0})) == (1, 0)
