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

"""lerobot-compare: label-efficient, anytime-valid policy comparison for lerobot."""
__version__ = "0.3.0"

from .compare import available_methods, compare_success
from .matching import Trial, match
from .report import DEFAULT_METHOD, Report
from .evaluators import Rollout

# Advanced / standalone statistics, exported for power users who want the pieces
# without the full `compare_success` pipeline.
from .bayesian import chance_to_beat
from .checks import sample_ratio_mismatch
from .fixed_horizon import min_detectable_effect, paired_z_test, two_proportion_z_test

__all__ = [
    # primary entry point + result
    "compare_success",
    "Report",
    "Trial",
    "match",
    "Rollout",
    # method selection
    "DEFAULT_METHOD",
    "available_methods",
    # standalone statistics
    "paired_z_test",
    "two_proportion_z_test",
    "min_detectable_effect",
    "chance_to_beat",
    "sample_ratio_mismatch",
]
