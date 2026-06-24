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
bayesian.py -- the Bayesian "chance to beat control" readout.

Most of lerobot-compare is FREQUENTIST (it controls a false-positive rate). This
module adds the one BAYESIAN number that experimentation platforms (Statsig's
"Bayesian" mode, and others) like to show alongside: the posterior probability that
policy A's true success rate beats policy B's,

    chance_to_beat = P(p_A > p_B | data).

It is reported as an intuitive companion to the frequentist verdict ("A wins with
99.1% probability"), NOT as the thing the statistical guarantee rests on.

Model. Each policy's success rate gets an independent BETA-BINOMIAL posterior: with
a Beta(a, b) prior and `s` successes in `n` trials, the posterior is
Beta(a + s, b + n - s). (Beta is the conjugate prior for the Binomial, so the
posterior is Beta again -- hence "Beta-Binomial".) The default prior Beta(1, 1) is
the uniform prior on [0, 1] (every rate equally likely a priori). We then estimate
P(p_A > p_B) by Monte-Carlo: draw many samples from each posterior and count how
often A's draw exceeds B's. (A closed form exists via a sum of Beta functions, but
Monte-Carlo is simpler, dependency-free, and plenty accurate here.)

Caveat (documented on purpose). Treating p_A and p_B as INDEPENDENT ignores the
pairing/correlation that the matched comparison exploits, so this number is a
slightly conservative, auxiliary readout -- consistent with how platforms present
their Bayesian probability. The rigorous, pairing-aware evidence is the e-value /
p-value from the chosen `method`.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def chance_to_beat(
    succ_A: int, n_A: int, succ_B: int, n_B: int, *,
    prior: Tuple[float, float] = (1.0, 1.0), draws: int = 20_000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Posterior probability P(p_A > p_B) under independent Beta-Binomial posteriors.

    Returns a probability in [0, 1]: ~0.5 means "no idea / a coin flip", values near
    1.0 mean A almost surely beats B, values near 0.0 mean B almost surely beats A.
    With no data (n_A = n_B = 0) and the default uniform prior this is ~0.5 by
    symmetry.
    """
    rng = rng or np.random.default_rng(0)
    a0, b0 = prior
    # Posterior shape parameters: prior + (successes, failures) for each policy.
    pA = rng.beta(a0 + succ_A, b0 + (n_A - succ_A), size=draws)
    pB = rng.beta(a0 + succ_B, b0 + (n_B - succ_B), size=draws)
    # Ties (pA == pB) have probability zero for continuous Beta draws, so `>` is fine.
    return float(np.mean(pA > pB))
