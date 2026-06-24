"""
viz.py -- charts for a Report. Requires the [viz] extra (matplotlib).

Plot set (grounded in what robot-eval practitioners actually use, plus the
anytime-valid signatures this tool uniquely has):
  evidence        wealth vs the 1/alpha threshold  -- "how evidence accrued"
  confidence_seq  the effect estimate with its narrowing anytime-valid band
  beta_violins    per-policy success-rate posteriors (TRI's recommended view;
                  avoids the overlapping-CI misreading of bar+error-bar plots)
  power_curve     Clopper-Pearson CI width vs N -- "how many rollouts you need"
  matching        match-rung breakdown + covariate balance before/after
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .report import Report
from .intervals import betting_confidence_sequence, cp_halfwidth_curve, clopper_pearson

_RUNG_COLOR = {"exact": "#2e7d32", "covariate": "#c2a14a", "unpaired": "#7aa6c2"}


def _lazy_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:                            # pragma: no cover
        raise ImportError("charts need the [viz] extra:  pip install matplotlib") from e


def plot_evidence(report: Report, ax=None):
    plt = _lazy_plt()
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    thr = 1.0 / report.alpha
    if report.wealth_AgtB:
        ax.plot(range(1, len(report.wealth_AgtB) + 1), report.wealth_AgtB,
                color="#2e7d32", label="evidence A>B")
    if report.wealth_BgtA and report.alternative != "A>B":
        ax.plot(range(1, len(report.wealth_BgtA) + 1), report.wealth_BgtA,
                color="#b23b3b", label="evidence B>A")
    ax.axhline(thr, ls="--", color="k", lw=1, label=f"decide (1/alpha={thr:.0f})")
    if report.n_to_decision:
        ax.axvline(report.n_to_decision, ls=":", color="#555")
        ax.annotate(f"stop @ {report.n_to_decision}",
                    (report.n_to_decision, thr), fontsize=9,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_yscale("log"); ax.set_xlabel("rollouts (pairs)")
    ax.set_ylabel("evidence (wealth, log)")
    ax.set_title("Evidence accumulated over time"); ax.legend(fontsize=8)
    return ax


def plot_confidence_seq(report: Report, ax=None):
    plt = _lazy_plt()
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    if report.pair_diffs:
        t, lo, hi = betting_confidence_sequence(report.pair_diffs, report.alpha)
        run = np.cumsum(report.pair_diffs) / t
        ax.fill_between(t, lo, hi, color="#7aa6c2", alpha=0.35,
                        label="anytime-valid CI")
        ax.plot(t, run, color="#1f3b5c", lw=1.5, label="effect estimate")
    ax.axhline(0.0, ls="--", color="k", lw=1, label="no difference")
    ax.set_xlabel("rollouts (pairs)"); ax.set_ylabel("effect  p_A - p_B")
    ax.set_title("Effect with narrowing anytime-valid interval")
    ax.legend(fontsize=8); ax.set_ylim(-1, 1)
    return ax


def _beta_violin(ax, k, n, center, color, label):
    from scipy.stats import beta as B
    ys = np.linspace(0, 1, 300)
    dens = B.pdf(ys, k + 1, n - k + 1)
    dens = dens / (dens.max() + 1e-12) * 0.4
    ax.fill_betweenx(ys, center - dens, center + dens, color=color, alpha=0.7)
    lo, hi = clopper_pearson(k, n, 0.10)
    ax.plot([center, center], [lo, hi], color="k", lw=1)
    ax.plot(center, k / max(n, 1), "o", color="k", ms=4)
    ax.text(center, 1.02, label, ha="center", fontsize=9)


def plot_beta_violins(report: Report, ax=None):
    plt = _lazy_plt()
    ax = ax or plt.subplots(figsize=(5, 4))[1]
    _beta_violin(ax, report.successes_A, report.n_pairs, 0.7, "#2e7d32",
                 f"A\n{report.successes_A}/{report.n_pairs}")
    _beta_violin(ax, report.successes_B, report.n_pairs, 1.5, "#b23b3b",
                 f"B\n{report.successes_B}/{report.n_pairs}")
    ax.set_xlim(0.2, 2.0); ax.set_ylim(0, 1.12); ax.set_xticks([])
    ax.set_ylabel("success rate (posterior)")
    ax.set_title("Per-policy uncertainty (beta posteriors)")
    return ax


def plot_power_curve(report: Report, ax=None):
    plt = _lazy_plt()
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    p = max(report.rate_A, report.rate_B, 0.01)
    ns = np.unique(np.round(np.linspace(10, max(report.n_pairs * 1.5, 60), 40))).astype(int)
    hw = cp_halfwidth_curve(p, ns, report.alpha)
    ax.plot(ns, hw, color="#444")
    ax.axvline(report.n_pairs, ls=":", color="#2e7d32",
               label=f"you are here (N={report.n_pairs})")
    ax.set_xlabel("number of rollouts N"); ax.set_ylabel("CI half-width")
    ax.set_title(f"How many rollouts (at rate {p:.0%})"); ax.legend(fontsize=8)
    return ax


def plot_matching(report: Report, ax=None):
    plt = _lazy_plt()
    ax = ax or plt.subplots(figsize=(5, 4))[1]
    rc = report.rung_counts or {}
    rungs = ["exact", "covariate", "unpaired"]
    vals = [rc.get(r, 0) for r in rungs]
    ax.bar(rungs, vals, color=[_RUNG_COLOR[r] for r in rungs])
    for i, v in enumerate(vals):
        if v:
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    sub = report.match_strength
    if report.covariate_balance_before is not None:
        sub += (f"\nimbalance {report.covariate_balance_before:.2f}"
                f" -> {report.covariate_balance_after:.2f}")
    ax.set_ylabel("pairs"); ax.set_title("Match quality\n" + sub, fontsize=10)
    return ax


def dashboard(report: Report, path: str = "report.png"):
    """Render all panels to a single PNG and return the path."""
    plt = _lazy_plt()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    plot_evidence(report, axes[0, 0])
    plot_confidence_seq(report, axes[0, 1])
    plot_beta_violins(report, axes[0, 2])
    plot_power_curve(report, axes[1, 0])
    plot_matching(report, axes[1, 1])
    axes[1, 2].axis("off")
    axes[1, 2].text(0.0, 0.95, report.summary(), va="top", family="monospace",
                    fontsize=10, transform=axes[1, 2].transAxes)
    fig.suptitle("lerobot-compare report", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=120); plt.close(fig)
    return path
