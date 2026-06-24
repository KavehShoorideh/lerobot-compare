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
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

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


def emit(diags: Iterable[Diagnostic]) -> List[Diagnostic]:
    """Surface diagnostics: warnings via the warnings module, others printed."""
    out = list(diags)
    for d in out:
        if d.level == "warn":
            warnings.warn(str(d), stacklevel=2)
        else:
            print(d)
    return out
