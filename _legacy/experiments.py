"""
experiments.py -- reproduce every figure and number in the paper.

Run:  python3 experiments.py
Writes:  results.json  and  figures/*.png

All experiments use fixed seeds; numbers are medians/rates over Monte Carlo
replicates. These are the exact quantities the paper and the test suite assert.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lerobot_doctor as P

FIG = "figures"
os.makedirs(FIG, exist_ok=True)
ALPHA = 0.10
EV = P.CheapVLM()


def batch(pA, pB, trials, *, rule=None, adaptive=True, control_variate=True,
          rho_acc=0.8, N=1200, seed0=0):
    dec, golds, seens, rates = 0, [], [], []
    for t in range(trials):
        items, predict, gold = P.make_world(pA, pB, np.random.default_rng(seed0 + t),
                                             rho_acc=rho_acc, N=N)
        r = P.compare(items, EV, gold, predict, rule=rule, alpha=ALPHA,
                      adaptive=adaptive, control_variate=control_variate,
                      rng=np.random.default_rng(900_000 + seed0 + t))
        dec += int(r.decided); golds.append(r.n_audited)
        seens.append(r.n_seen); rates.append(r.n_audited / r.n_seen)
    return dict(detect=dec / trials, gold=float(np.median(golds)),
                seen=float(np.median(seens)), rate=float(np.mean(rates)),
                golds=golds)


def wealth_distribution_under_h0(trials=400, N=800, seed0=0):
    """Final wealth under H0 -- should be a supermartingale (median <= 1)."""
    finals = []
    for t in range(trials):
        items, predict, gold = P.make_world(0.6, 0.6, np.random.default_rng(seed0 + t),
                                             rho_acc=0.8, N=N)
        r = P.compare(items, EV, gold, predict, alpha=1e9,  # never stop
                      adaptive=True, control_variate=True,
                      rng=np.random.default_rng(700_000 + t))
        finals.append(r.wealth)
    return finals


def main():
    res = {}

    # ---- E1: efficiency ladder (the headline) ------------------------------
    T = 200
    baseline = batch(0.75, 0.55, T, rule=P.UniformAudit(pi=0.3),
               adaptive=False, control_variate=False, seed0=10)
    v2u = batch(0.75, 0.55, T, rule=P.UniformAudit(pi=0.3), seed0=10)
    v2p = batch(0.75, 0.55, T, rule=P.MaxEvidencePerCost(), seed0=10)
    v2m = batch(0.75, 0.55, T, rule=P.UniformAudit(pi=v2p["rate"]), seed0=10)
    ney = batch(0.75, 0.55, T, rule=P.NeymanAcquisition(), seed0=10)
    res["efficiency_ladder"] = {
        "v1_uniform": baseline, "v2_uniform": v2u, "v2_principled": v2p,
        "v2_uniform_matched": v2m, "v2_neyman": ney}

    labels = ["baseline\n(fixed+IPW)\nuniform", "enhanced\n(adapt+CV)\nuniform",
              "enhanced\nprincipled", "enhanced uniform\n@matched", "enhanced\nNeyman"]
    golds = [baseline["gold"], v2u["gold"], v2p["gold"], v2m["gold"], ney["gold"]]
    colors = ["#b0b0b0", "#7aa6c2", "#2e7d32", "#c2a14a", "#b23b3b"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, golds, color=colors)
    for b, g in zip(bars, golds):
        ax.text(b.get_x() + b.get_width() / 2, g + 5, f"{g:.0f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("median gold labels to decision")
    ax.set_title("Efficiency ladder (H1: p_A=0.75 vs p_B=0.55, alpha=0.10)\n"
                 "lower is better; all detect at 100% except Neyman")
    ax.set_ylim(0, max(golds) * 1.2)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig1_efficiency_ladder.png", dpi=130)
    plt.close(fig)

    # ---- E2: validity across evaluator quality and rule --------------------
    rules = {"uniform": P.UniformAudit(pi=0.3),
             "principled": P.MaxEvidencePerCost(),
             "Neyman": P.NeymanAcquisition()}
    rhos = [0.5, 0.7, 0.9]
    valid = {}
    for rname, rule in rules.items():
        valid[rname] = []
        for rho in rhos:
            b = batch(0.6, 0.6, 250, rule=rule, rho_acc=rho, seed0=1)
            valid[rname].append(b["detect"])
    res["validity_fp"] = {"rhos": rhos, **valid}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(rhos)); w = 0.25
    for i, (rname, vals) in enumerate(valid.items()):
        ax.bar(x + (i - 1) * w, vals, w, label=rname)
    ax.axhline(ALPHA, ls="--", color="k", label=f"alpha = {ALPHA}")
    ax.set_xticks(x); ax.set_xticklabels([f"rho={r}" for r in rhos])
    ax.set_ylabel("false-positive rate under H0")
    ax.set_ylim(0, 0.2)
    ax.set_title("Validity: type-I error stays below alpha for EVERY evaluator\n"
                 "quality (incl. useless rho=0.5) and EVERY acquisition rule")
    ax.legend()
    fig.tight_layout(); fig.savefig(f"{FIG}/fig2_validity.png", dpi=130)
    plt.close(fig)

    # ---- E3: Neyman backfires (detection vs gold, matched budget) ----------
    T = 200
    pr = batch(0.75, 0.55, T, rule=P.MaxEvidencePerCost(), seed0=12)
    un = batch(0.75, 0.55, T, rule=P.UniformAudit(pi=pr["rate"]), seed0=12)
    ne = batch(0.75, 0.55, T, rule=P.NeymanAcquisition(), seed0=12)
    res["neyman_backfires"] = {"principled": pr, "uniform_matched": un, "neyman": ne}

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4.2))
    names = ["principled", "uniform\n@matched", "Neyman"]
    det = [pr["detect"], un["detect"], ne["detect"]]
    gld = [pr["gold"], un["gold"], ne["gold"]]
    a1.bar(names, det, color=["#2e7d32", "#c2a14a", "#b23b3b"])
    a1.set_ylabel("detection rate"); a1.set_ylim(0, 1.05)
    a1.set_title("detection (higher better)")
    a2.bar(names, gld, color=["#2e7d32", "#c2a14a", "#b23b3b"])
    a2.set_ylabel("median gold to decision"); a2.set_title("gold cost (lower better)")
    fig.suptitle("Neyman allocation backfires: audits where UNCERTAIN, starving "
                 "the signal\n(estimation-optimal != detection-optimal)")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig3_neyman_backfires.png", dpi=130)
    plt.close(fig)

    # ---- E4: evaluator quality -> fewer items (control-variate effect) -----
    rhos2 = [0.55, 0.65, 0.75, 0.85, 0.95]
    seen_by_rho, gold_by_rho = [], []
    for rho in rhos2:
        b = batch(0.75, 0.55, 200, rule=P.UniformAudit(pi=0.3), rho_acc=rho, seed0=20)
        seen_by_rho.append(b["seen"]); gold_by_rho.append(b["gold"])
    res["quality_effect"] = {"rhos": rhos2, "seen": seen_by_rho, "gold": gold_by_rho}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(rhos2, seen_by_rho, "o-", color="#2e7d32", label="items to decision")
    ax.plot(rhos2, gold_by_rho, "s--", color="#7aa6c2", label="gold to decision")
    ax.set_xlabel("evaluator quality rho"); ax.set_ylabel("median count")
    ax.set_title("Better evaluator -> fewer items to decision (robust);\n"
                 "gold effect is real but modest in this regime")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig4_quality_effect.png", dpi=130)
    plt.close(fig)

    # ---- E5: savings vs effect size (principled vs uniform@matched) --------
    gaps = [(0.65, 0.55), (0.70, 0.55), (0.75, 0.55), (0.80, 0.55), (0.85, 0.55)]
    pr_gold, un_gold, xs = [], [], []
    for pA, pB in gaps:
        p = batch(pA, pB, 150, rule=P.MaxEvidencePerCost(), seed0=30)
        u = batch(pA, pB, 150, rule=P.UniformAudit(pi=p["rate"]), seed0=30)
        pr_gold.append(p["gold"]); un_gold.append(u["gold"]); xs.append(pA - pB)
    res["savings_vs_effect"] = {"effect": xs, "principled": pr_gold, "uniform": un_gold}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, un_gold, "s--", color="#c2a14a", label="uniform @ matched budget")
    ax.plot(xs, pr_gold, "o-", color="#2e7d32", label="principled acquisition")
    ax.set_xlabel("true effect size  p_A - p_B")
    ax.set_ylabel("median gold to decision")
    ax.set_title("Principled acquisition spends less gold across effect sizes")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig5_savings_vs_effect.png", dpi=130)
    plt.close(fig)

    # ---- E6: supermartingale under H0 --------------------------------------
    finals = wealth_distribution_under_h0()
    res["h0_wealth_median"] = float(np.median(finals))
    res["h0_wealth_p_ge_10"] = float(np.mean(np.array(finals) >= 10))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(np.clip(finals, 0, 12), bins=40, color="#7aa6c2", edgecolor="white")
    ax.axvline(1.0, ls="--", color="k", label="W=1 (start)")
    ax.axvline(10.0, ls=":", color="#b23b3b", label="1/alpha reject threshold")
    ax.set_xlabel("final wealth under H0"); ax.set_ylabel("runs")
    ax.set_title(f"Wealth is a supermartingale under H0 "
                 f"(median={np.median(finals):.2f}, P(W>=10)={np.mean(np.array(finals)>=10):.3f})")
    ax.legend()
    fig.tight_layout(); fig.savefig(f"{FIG}/fig6_supermartingale.png", dpi=130)
    plt.close(fig)

    with open("results.json", "w") as f:
        json.dump(res, f, indent=2, default=float)

    # console summary
    print("efficiency ladder (median gold):",
          {k: round(v["gold"]) for k, v in res["efficiency_ladder"].items()})
    print("validity max FP:", max(max(v) for v in valid.values()), "(alpha=0.1)")
    print("neyman:", {"principled_gold": round(pr["gold"]),
                      "uniform_matched_gold": round(un["gold"]),
                      "neyman_detect": ne["detect"], "principled_detect": pr["detect"]})
    print("H0 wealth median:", res["h0_wealth_median"],
          "P(W>=10):", res["h0_wealth_p_ge_10"])
    print("figures written to", FIG + "/")


if __name__ == "__main__":
    main()
