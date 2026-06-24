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
lerobot_doctor.core -- anytime-valid, cost-aware, prediction-powered evaluation.

Domain-agnostic. Knows only: estimands, evaluators (score + cost), an
auditing rule, and a betting e-process. numpy only; no torch / lerobot.
"""

from .eprocess import BettingEProcess, Obs
from .payoff import Comparison
from .auditing import Auditor, MaxEvidenceAuditor, UniformAuditor, NeymanAuditor
from .compare import (
    Evaluator, GoldSource, Result,
    compare, make_world, CheapVLM,
)

__all__ = [
    # eprocess
    "BettingEProcess",
    "Obs",
    # payoff
    "Comparison",
    # auditing
    "Auditor",
    "MaxEvidenceAuditor",
    "UniformAuditor",
    "NeymanAuditor",
    # compare
    "Evaluator",
    "GoldSource",
    "Result",
    "compare",
    "make_world",
    "CheapVLM",
]
