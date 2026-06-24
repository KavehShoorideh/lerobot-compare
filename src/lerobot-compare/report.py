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
report.py -- the structured result of an A/B comparison, plus a Statsig-style readout.

Everything the tool knows, as plain data: the verdict, the per-policy numbers, the
effect and its confidence interval (CI), the evidence (e-value / p-value), the
health checks, and the match quality. `to_dict()` gives JSON for the CLI and the
chart layer; `summary()` renders a human-readable scorecard modelled on how
experimentation platforms (e.g. Statsig) present results. No plotting dependencies
live here.

The same `Report` shape is produced by BOTH supported methods (selected via the
`method` argument of `compare_success`):
  * "anytime-valid-betting-e-process" -- the default; reports an e-value and a
    widened, anytime-valid CI (you may peek and stop early).
  * "fixed-horizon-proportion-z-test" -- the classic fixed-N A/B test; reports a
    p-value, standard error (SE), achieved power, and minimum detectable effect
    (MDE).
Fields that only one method populates are left at `None` and simply omitted from the
scorecard.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

# Canonical default method name. Defined here (the low-level module) so it can be the
# `Report.method` default without report.py having to import the registry in
# compare.py (which would be a circular import). compare.py imports this back.
DEFAULT_METHOD = "anytime-valid-betting-e-process"


# --------------------------------------------------------------------------- #
# Small formatting helpers for the scorecard (kept module-level so summary()    #
# reads as a layout, not a pile of f-strings). Effects/rates are proportions in #
# [0, 1]; we render them as percents, and absolute gaps as "percentage points". #
# --------------------------------------------------------------------------- #
def _row(label: str, value: str, extra: str = "") -> str:
    """One aligned scorecard row: '  <label padded><value><extra>'."""
    return f"  {label:<22}{value}{extra}"


def _ci_pts(lo: Optional[float], hi: Optional[float]) -> str:
    """Confidence interval rendered in percentage points; None side -> +/-infinity."""
    lo_s = "-inf" if lo is None else f"{lo * 100:+.1f}"
    hi_s = "+inf" if hi is None else f"{hi * 100:+.1f}"
    return f"[{lo_s}, {hi_s}]"


def _ci_pct(lo: Optional[float], hi: Optional[float]) -> str:
    """Confidence interval rendered as percentages; None side -> +/-infinity."""
    lo_s = "-inf" if lo is None else f"{lo * 100:+.1f}%"
    hi_s = "+inf" if hi is None else f"{hi * 100:+.1f}%"
    return f"[{lo_s}, {hi_s}]"


def _ci_rate(ci: Optional[Tuple[float, float]]) -> str:
    """A per-variant success-rate CI like '[59.8%, 66.1%]' (empty string if absent)."""
    if not ci:
        return ""
    lo, hi = ci
    return f"[{lo * 100:.1f}%, {hi * 100:.1f}%]"


@dataclass
class Report:
    # --- required: test configuration + core outcome ----------------------- #
    alpha: float
    alternative: str                      # "two-sided" | "A>B" | "B>A"

    decided: bool                         # significant? (e-value crossed, or p < alpha)
    direction: Optional[str]              # "A>B" | "B>A" | None
    n_to_decision: Optional[int]          # pair index at which it decided (anytime-valid)

    n_pairs: int
    n_A: int
    n_B: int
    successes_A: int
    successes_B: int
    rate_A: float
    rate_B: float
    effect: float                         # rate_A - rate_B (absolute, descriptive)

    e_value: float                        # max wealth across the tested direction(s)
    p_value_equiv: float                  # anytime-valid p-value bound (see e_threshold note)

    # The wealth an e-process must reach to decide. For a TWO-SIDED test we run two
    # one-sided processes each at level alpha/2 (union bound), so the honest bar is
    # 1/(alpha/2) = 2/alpha, NOT 1/alpha -- and the matching anytime-valid p-value is
    # min(1, 2/e_value). Stored explicitly so the scorecard never misstates the bar.
    e_threshold: Optional[float] = None   # decision threshold for e_value (anytime-valid)

    # --- method / horizon selection (drives the scorecard layout) ---------- #
    method: str = DEFAULT_METHOD          # full descriptive name; see compare._METHODS
    horizon: str = "anytime-valid"        # "anytime-valid" | "fixed-horizon" (display)
    test: str = "betting e-process"       # specific test within the horizon (display)
    confidence_level: float = 0.90        # 1 - alpha, shown in the header

    # --- effect confidence interval on the absolute gap p_A - p_B ---------- #
    effect_ci_lo: Optional[float] = None  # None = open (-inf) side, for a one-sided test
    effect_ci_hi: Optional[float] = None  # None = open (+inf) side

    # --- relative lift = (rate_A - rate_B) / rate_B (Statsig's "Delta %") --- #
    effect_relative: Optional[float] = None
    effect_relative_ci_lo: Optional[float] = None
    effect_relative_ci_hi: Optional[float] = None

    # --- per-variant success-rate CIs (Clopper-Pearson exact binomial) ----- #
    rate_A_ci: Optional[Tuple[float, float]] = None
    rate_B_ci: Optional[Tuple[float, float]] = None

    # --- fixed-horizon-only frequentist statistics ------------------------- #
    standard_error: Optional[float] = None   # SE of the effect
    z_score: Optional[float] = None          # test statistic (effect / SE)
    p_value: Optional[float] = None          # exact p-value (cf. p_value_equiv)
    power: Optional[float] = None            # achieved power at the observed effect
    mde: Optional[float] = None              # minimum detectable effect at target power
    target_power: float = 0.80               # the power MDE is reported against

    # --- sample ratio mismatch (SRM) health check -------------------------- #
    srm_p_value: Optional[float] = None
    srm_flag: bool = False                   # True => split is suspiciously lopsided

    # --- Bayesian companion readout ---------------------------------------- #
    chance_to_beat: Optional[float] = None   # posterior P(p_A > p_B)

    # --- trajectory / diagnostics data ------------------------------------- #
    wealth_AgtB: List[float] = field(default_factory=list)   # trajectory
    wealth_BgtA: List[float] = field(default_factory=list)
    pair_diffs: List[int] = field(default_factory=list)      # ordered y_A - y_B
    pair_rungs: List[str] = field(default_factory=list)      # match rung per pair

    rung_counts: dict = field(default_factory=dict)
    n_unmatched_A: int = 0
    n_unmatched_B: int = 0
    covariate_balance_before: Optional[float] = None
    covariate_balance_after: Optional[float] = None

    # --- audited-mode (cheap-evaluator + auditing) fields ------------------ #
    mode: str = "direct"                  # "direct" (human-only) | "audited" (cheap judge + audit)
    n_audited: int = 0                    # rollouts that cost a human label
    n_seen: int = 0                       # rollouts the cheap evaluator scored
    audit_rate: float = 1.0
    lam_pp: float = 1.0                   # learned trust in the cheap evaluator
    gold_saved_frac: float = 0.0          # 1 - n_audited/n_seen

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def match_strength(self) -> str:
        c = self.rung_counts or {}
        if c.get("unpaired", 0) == 0 and c.get("covariate", 0) == 0:
            return "exact (clean paired test)"
        if c.get("exact", 0) == 0 and c.get("covariate", 0) == 0:
            return "unpaired (two-sample; higher variance)"
        if c.get("covariate", 0):
            return "covariate-matched (best-effort; assumes ignorability)"
        return "mixed"

    # ------------------------------------------------------------------ #
    # Rendering. summary() assembles the scorecard from method-aware     #
    # blocks; each block is small and independently testable.            #
    # ------------------------------------------------------------------ #
    def summary(self) -> str:
        """Statsig-style scorecard: a verdict line, a per-variant table, the effect
        with confidence intervals and relative lift, then health/power/match notes.
        Plain text (no colour codes) so it pipes and tests cleanly."""
        cl = f"{self.confidence_level:.0%}"
        header = (f"lerobot-compare | A/B result -- {self.horizon} / {self.test}, "
                  f"{cl} confidence")
        lines = [header, "-" * len(header)]
        lines += self._headline_lines()
        lines.append("")
        lines += self._variant_lines()
        lines.append("")
        lines += self._effect_lines()
        lines += self._footer_lines()
        return "\n".join(lines)

    @staticmethod
    def _fmt_e(e: float) -> str:
        return f"{e:.1f}" if e < 1e4 else f"{e:.1e}"

    def _headline_lines(self) -> List[str]:
        """The verdict, phrased in the idiom of the chosen method."""
        if self.horizon == "fixed-horizon":
            if self.decided:
                p = self.p_value if self.p_value is not None else 0.0
                return [f"  Decision: {self.direction} is SIGNIFICANT  (p = {p:.4g})"]
            ptxt = f"  (p = {self.p_value:.4g})" if self.p_value is not None else ""
            return [f"  Decision: NOT SIGNIFICANT -- no reliable A/B difference{ptxt}"]
        # anytime-valid (betting e-process)
        ev = self._fmt_e(self.e_value)
        thr = self.e_threshold if self.e_threshold is not None else 1.0 / self.alpha
        if self.decided:
            return [f"  Decision: {self.direction}  (e-value {ev} >= {thr:.0f}; "
                    f"anytime-valid p < {max(self.p_value_equiv, 1e-4):.4f}); "
                    f"stopped after {self.n_to_decision} pairs"]
        return [f"  Decision: NO CALL yet  (e-value {ev} < {thr:.0f}; "
                f"need more rollouts)"]

    def _ci_label(self) -> str:
        cl = f"{self.confidence_level:.0%}"
        return f"{cl} CI (anytime-valid)" if self.horizon == "anytime-valid" else f"{cl} CI"

    def _variant_lines(self) -> List[str]:
        """A two-row table: test (A) vs control (B), rate + per-variant CI."""
        if self.mode == "audited":
            a_cell = f"~{self.rate_A:.1%}  (est. {self.n_audited} checks)"
            b_cell = f"~{self.rate_B:.1%}"
        else:
            a_cell = f"{self.rate_A:.1%}  ({self.successes_A}/{self.n_pairs})"
            b_cell = f"{self.rate_B:.1%}  ({self.successes_B}/{self.n_pairs})"
        return [
            f"  {'Variant':<13}{'Success rate':<26}{self._ci_label()}",
            f"  {'A (test)':<13}{a_cell:<26}{_ci_rate(self.rate_A_ci)}",
            f"  {'B (control)':<13}{b_cell:<26}{_ci_rate(self.rate_B_ci)}",
        ]

    def _effect_lines(self) -> List[str]:
        """Absolute lift + relative lift (with CIs), SE, and the Bayesian readout."""
        cl = f"{self.confidence_level:.0%}"
        widened = " (anytime-valid; widened)" if self.horizon == "anytime-valid" else ""
        out = [_row("Absolute lift (A-B):", f"{self.effect * 100:+.1f} pts",
                    f"   {cl} CI {_ci_pts(self.effect_ci_lo, self.effect_ci_hi)}{widened}")]
        if self.effect_relative is not None:
            out.append(_row(
                "Relative lift:", f"{self.effect_relative * 100:+.1f}%",
                f"   {cl} CI {_ci_pct(self.effect_relative_ci_lo, self.effect_relative_ci_hi)}"))
        if self.standard_error is not None:
            out.append(_row("Standard error:", f"{self.standard_error:.3f}"))
        if self.chance_to_beat is not None:
            out.append(_row("Bayesian P(A>B):", f"{self.chance_to_beat:.1%}"))
        return out

    def _footer_lines(self) -> List[str]:
        """Health (SRM), power/MDE (fixed-horizon), and match-quality diagnostics."""
        out: List[str] = []
        if self.srm_p_value is not None:
            status = "MISMATCH" if self.srm_flag else "OK"
            out.append(_row("Health:", f"sample-ratio {status} "
                            f"(n_A={self.n_A}, n_B={self.n_B}, SRM p={self.srm_p_value:.3g})"))
        if self.power is not None and self.mde is not None:
            out.append(_row("Power:", f"{self.power:.2f} achieved; "
                            f"MDE at {self.target_power:.0%} power = {self.mde * 100:.1f} pts"))
        out.append(_row("Match:", self.match_strength))
        out.append(_row("Rungs:", str(self.rung_counts)))
        if self.covariate_balance_before is not None:
            out.append(_row("Covariate imbalance:",
                            f"{self.covariate_balance_before:.3f} -> "
                            f"{self.covariate_balance_after:.3f} after matching"))
        if self.mode == "audited":
            out.append(_row("Cheap evaluator:",
                            f"scored {self.n_seen}, human-audited {self.n_audited} "
                            f"({self.gold_saved_frac:.0%} of human labels saved); "
                            f"trust lam_PP={self.lam_pp:.2f}"))
        return out
