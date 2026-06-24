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

"""Public re-exports for lerobot_doctor.evaluators.

Heavy deps (torch/transformers) are NOT imported here; use
`lerobot_doctor.evaluators.vlm` explicitly if you need VLMJudge.
"""

from .base import Evaluator, HumanGold, Rollout
from .calibrate import (
    CalibratedEvaluator,
    EvaluatorUpdater,
    LabelRecord,
    Recalibrate,
)
from .internal import AutomaticChecker, InternalState
from .stub import StubVLM

__all__ = [
    "Rollout",
    "Evaluator",
    "HumanGold",
    "StubVLM",
    "AutomaticChecker",
    "InternalState",
    "Recalibrate",
    "CalibratedEvaluator",
    "LabelRecord",
    "EvaluatorUpdater",
]
