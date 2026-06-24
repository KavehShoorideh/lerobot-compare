"""
report.py -- the structured result of an A/B comparison.

Everything the tool knows, as plain data: the verdict, the per-policy numbers,
the effect, the evidence (e-value / wealth), where it stopped, and the match
quality. `to_dict()` gives JSON for the CLI and the chart layer; `summary()`
gives a human-readable readout. No plotting dependencies live here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class Report:
    alpha: float
    alternative: str                      # "two-sided" | "A>B" | "B>A"

    decided: bool
    direction: Optional[str]              # "A>B" | "B>A" | None
    n_to_decision: Optional[int]          # pair index at which it decided

    n_pairs: int
    n_A: int
    n_B: int
    successes_A: int
    successes_B: int
    rate_A: float
    rate_B: float
    effect: float                         # rate_A - rate_B (descriptive)

    e_value: float                        # max wealth across the tested direction(s)
    p_value_equiv: float                  # 1 / e_value (anytime-valid p-value bound)
    wealth_AgtB: List[float] = field(default_factory=list)   # trajectory
    wealth_BgtA: List[float] = field(default_factory=list)
    pair_diffs: List[int] = field(default_factory=list)      # ordered y_A - y_B
    pair_rungs: List[str] = field(default_factory=list)      # match rung per pair

    rung_counts: dict = field(default_factory=dict)
    n_unmatched_A: int = 0
    n_unmatched_B: int = 0
    covariate_balance_before: Optional[float] = None
    covariate_balance_after: Optional[float] = None

    # --- enhanced (cheap-evaluator + auditing) fields ---
    mode: str = "baseline"                      # "baseline" (human-only) | "enhanced" (cheap+audit)
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

    def summary(self) -> str:
        ev = f"{self.e_value:.1f}" if self.e_value < 1e4 else f"{self.e_value:.1e}"
        if self.decided:
            verdict = (f"DECISION: {self.direction}  "
                       f"(e-value {ev} >= {1/self.alpha:.0f}, "
                       f"anytime-valid p < {max(self.p_value_equiv, 1e-4):.4f}); "
                       f"stopped after {self.n_to_decision} pairs")
        else:
            verdict = (f"NO CALL yet  (e-value {ev} < "
                       f"{1/self.alpha:.0f}; need more rollouts)")
        lines = [verdict]
        if self.mode == "enhanced":
            lines += [
                f"  policy A: ~{self.rate_A:.1%}   policy B: ~{self.rate_B:.1%}"
                f"   (estimated from {self.n_audited} human checks)",
                f"  estimated gap (A-B): {self.effect:+.1%}",
                f"  pairs seen: {self.n_seen}   match: {self.match_strength}",
            ]
        else:
            lines += [
                f"  policy A: {self.successes_A}/{self.n_A} = {self.rate_A:.1%}",
                f"  policy B: {self.successes_B}/{self.n_B} = {self.rate_B:.1%}",
                f"  observed gap (A-B): {self.effect:+.1%}",
                f"  pairs: {self.n_pairs}   match: {self.match_strength}",
            ]
        lines.append(f"  rungs: {self.rung_counts}")
        if self.covariate_balance_before is not None:
            lines.append(f"  covariate imbalance: {self.covariate_balance_before:.3f}"
                         f" -> {self.covariate_balance_after:.3f} after matching")
        if self.mode == "enhanced":
            lines.append(f"  cheap evaluator: scored {self.n_seen}, "
                         f"human-audited {self.n_audited} "
                         f"({self.gold_saved_frac:.0%} of human labels saved); "
                         f"trust lam_PP={self.lam_pp:.2f}")
        return "\n".join(lines)
