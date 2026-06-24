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
checks.py -- preflight diagnostics, including the built-in success-flag policy.

Core stance: lerobot-compare exists to DETERMINE success and the trustworthiness of
a success signal. So a dataset's built-in success flag is the thing under
examination -- it cannot double as the answer key. By default it is detected,
WARNED about, and IGNORED. You cannot audit a signal by trusting it.

You may override, but only by declaring what the flag actually is:
    use_as="ignore"      (default) -- warn and drop it
    use_as="evaluator"   -- treat it as a CHEAP, fallible evaluator to be
                            calibrated against human gold (never trusted as gold)
    use_as="human_gold"  -- trust it as gold ONLY if you assert a human produced it

The source cannot be inferred from the column values, so it must be declared;
silence means "ignore".
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

# column names commonly used for a built-in verdict / reward / termination
_FLAG_NAMES = {
    "success", "is_success", "task_success", "episode_success", "succeeded",
    "reward", "rewards", "done", "terminal", "terminated", "next.success",
    "next.reward", "next.done", "info.success",
}


@dataclass
class Diagnostic:
    level: str          # "warn" | "info" | "error"
    code: str
    message: str
    data: dict = field(default_factory=dict)   # optional structured payload (e.g. p-values)

    def __str__(self) -> str:
        return f"[{self.level.upper()}] {self.message}"


def _looks_like_flag(col: str) -> bool:
    c = col.lower().strip()
    return c in _FLAG_NAMES or "success" in c or c.endswith(".reward") or c == "reward"


def scan_success_flags(
    columns: Iterable[str], *, use_as: str = "ignore",
) -> List[Diagnostic]:
    """Inspect column names for built-in success flags and return diagnostics
    according to the declared `use_as` policy (default: ignore + warn)."""
    found = [c for c in columns if _looks_like_flag(c)]
    diags: List[Diagnostic] = []
    if not found:
        return diags
    cols = ", ".join(repr(c) for c in found)
    if use_as == "ignore":
        diags.append(Diagnostic(
            "warn", "builtin_flag_ignored",
            f"Built-in success flag(s) {cols} found and IGNORED. lerobot-compare "
            f"determines success and its trustworthiness from human gold plus "
            f"calibrated evaluators; an unaudited flag is the signal under "
            f"examination, not ground truth. To examine it, pass use_as='evaluator' "
            f"(calibrated, never trusted as gold); to trust it as gold you must "
            f"assert a human produced it with use_as='human_gold'."))
    elif use_as == "evaluator":
        diags.append(Diagnostic(
            "info", "builtin_flag_as_evaluator",
            f"Using built-in flag(s) {cols} as a CHEAP evaluator to be calibrated "
            f"against human gold -- not trusted as ground truth."))
    elif use_as == "human_gold":
        diags.append(Diagnostic(
            "info", "builtin_flag_as_gold",
            f"Trusting built-in flag(s) {cols} as gold on your assertion that a "
            f"human produced them. If the source is a script/sim/model, this is "
            f"unsound -- prefer use_as='evaluator'."))
    else:
        diags.append(Diagnostic(
            "error", "bad_use_as",
            f"use_as={use_as!r} invalid; choose ignore|evaluator|human_gold."))
    return diags


def preflight_rollouts(rollouts: Sequence, *, use_as: str = "ignore") -> List[Diagnostic]:
    """Scan Rollout.meta keys for flags (the same policy applied to in-memory
    rollouts rather than a dataset's column schema)."""
    keys = set()
    for r in rollouts:
        keys.update(getattr(r, "meta", {}).keys())
    diags = scan_success_flags(keys, use_as=use_as)
    if not any(getattr(r, "frames", None) is not None for r in rollouts):
        diags.append(Diagnostic(
            "warn", "no_frames",
            "No rollout carries observation frames; a VLM judge has nothing to "
            "read. Record camera frames, or use an automatic/internal evaluator."))
    return diags


def sample_ratio_mismatch(
    n_A: int, n_B: int, *, expected: float = 0.5, threshold: float = 1e-3,
) -> Diagnostic:
    """Sample Ratio Mismatch (SRM) check.

    A chi-square (pronounced "kai-square") goodness-of-fit test that the observed
    split between policy A and policy B matches the one you intended. SRM is a
    standard experimentation guardrail: if you meant to give A and B the same number
    of rollouts (a 50/50 split) but the counts come out lopsided, something upstream
    is biased -- a crash that drops one policy's hard episodes, a logging gap, a
    broken assignment -- and that bias can masquerade as a real A-vs-B effect.

    Ordinary sampling noise should NOT trip this, so (as is standard practice) the
    p-value `threshold` is deliberately tiny (default 1e-3): only a split far beyond
    chance raises the alarm. `expected` is policy A's intended share of the trials
    (0.5 = an even split).

    Returns a `Diagnostic` whose `.data["p_value"]` carries the chi-square p-value
    (read back by `compare.py` for the report). Level is "warn"/code "srm" when
    tripped, else "info"/code "srm_ok".
    """
    from scipy.stats import chi2 as _chi2          # lazy: keep checks.py import-light

    n = n_A + n_B
    if n == 0:
        return Diagnostic("info", "srm_ok",
                          "sample-ratio check skipped (no trials)",
                          {"p_value": 1.0, "n_A": 0, "n_B": 0})
    exp_A, exp_B = expected * n, (1.0 - expected) * n
    # Pearson chi-square statistic with 1 degree of freedom: sum (obs - exp)^2 / exp.
    stat = (n_A - exp_A) ** 2 / exp_A + (n_B - exp_B) ** 2 / exp_B
    p = float(_chi2.sf(stat, df=1))
    payload = {"p_value": p, "chi2": float(stat), "n_A": n_A, "n_B": n_B}
    if p < threshold:
        return Diagnostic(
            "warn", "srm",
            f"Sample Ratio Mismatch: observed split n_A={n_A}, n_B={n_B} differs from "
            f"the expected {expected:.0%}/{1 - expected:.0%} far beyond chance "
            f"(chi-square p={p:.2e} < {threshold:g}). The result may reflect biased "
            f"data collection rather than a real A-vs-B difference.",
            payload)
    return Diagnostic(
        "info", "srm_ok",
        f"sample-ratio OK (n_A={n_A}, n_B={n_B}; chi-square p={p:.3f})",
        payload)


def emit(diags: Iterable[Diagnostic]) -> List[Diagnostic]:
    """Surface diagnostics: warnings via the warnings module, others printed."""
    out = list(diags)
    for d in out:
        if d.level == "warn":
            warnings.warn(str(d), stacklevel=2)
        else:
            print(d)
    return out
