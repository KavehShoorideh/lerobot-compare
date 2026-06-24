# enhanced Efficiency: why baseline is conservative, and how to fix it

## Context

The baseline core (`lerobot-compare.py`) is **valid but conservative**. On the synthetic
demo it controls type-I error far below budget (empirical 0.000 against an
`alpha = 0.10` target) and reaches the decision spending ~593 gold labels out
of 2000. Validity is not the issue; efficiency is. The conservativeness has
**two compounding causes, with one fix each**.

Neither fix touches the validity guarantee, and — importantly — neither is the
project's novelty. They make the engine *underneath* the acquisition rule
competitive, so the cost-aware multi-source acquisition (the actual
contribution) has a sharp substrate to run on.

---

## Cause 1 — the betting fraction is fixed at `lambda = 0.5`

The e-process accumulates wealth as a product:

```
W_t = prod_{s<=t} (1 + lambda * g_s)
```

The per-step growth rate is `E[log(1 + lambda * g)]`, and maximizing it over
`lambda` is exactly the **Kelly criterion**. The growth-optimal fraction is
approximately `mean(g) / second_moment(g)`, which **depends on the true effect
size**:

- For a *small* true gap, the optimal bet is small. Fixing `lambda = 0.5`
  wildly over-bets: wealth swings hard on adverse draws and grows slowly, even
  backward.
- For a *large* gap, `0.5` under-bets and leaves growth on the table.

Either way the wealth trajectory is off-optimal, so it crosses `1/alpha` later
than it should — which means more labels than necessary.

### Fix: Waudby-Smith--Ramdas (WSR) adaptive betting

Instead of fixing `lambda`, set `lambda_t` from the data seen *so far* — a
**predictable** (previsible) sequence that may depend on the past but not on the
current payoff `g_t`. The canonical choice approximates Kelly online:

```
lambda_t ~= running_mean(g) / running_second_moment(g)   (truncated to a safe range)
```

As evidence accumulates, `lambda_t` homes in on the growth-optimal fraction for
the *actual* effect size, so wealth grows at nearly the best possible rate.
Because `lambda_t` is predictable, the supermartingale property — and therefore
the entire anytime-valid guarantee — is preserved at **zero validity cost**.
You learn the optimal bet for free.

**Method-of-mixtures is the same idea by another door.** Put a prior over
`lambda` and integrate it out; the mixture is automatically a martingale that
auto-tunes without ever estimating `lambda`. This is the mixture sequential
probability ratio test (mixture SPRT). WSR and the mixture are two roads out of
the fixed-`lambda` penalty; either is fine.

---

## Cause 2 — the importance-weighted payoff inflates variance

The baseline audited score is

```
m = f + (1/pi) * (y - f)
```

That `1/pi` is the problem. When auditing is rare (`pi` near the 0.05 floor),
each audited payoff is multiplied by up to ~20, so it is enormous-variance — and
to keep the payoff in `[-1, 1]` we divide by a loose bound. High variance plus a
loose bound means each step delivers almost no evidence. baseline uses a
**variance-inflating** estimator.

### Fix: a control variate instead of an importance weight (this is CUPED)

The original prediction-powered inference (PPI) estimator is **not**
importance-weighted. It is:

```
theta_hat = mean(f over ALL items) + mean(y - f over AUDITED items)
```

The cheap predictions carry the estimate on every item; the gold label only
corrects a small bias via the residual `y - f`. When `f` is a decent predictor,
that residual has *small* variance — so the correction is low-variance and there
is no `1/pi` blow-up.

This is exactly **CUPED** (controlled experiment using pre-experiment data):
`f` is the covariate prediction of the outcome, you subtract it and add back a
low-variance correction, and the variance-reduction factor is `(1 - rho^2)`,
where `rho` is the correlation between the evaluator's score and true success.
It is CUPED for the e-process. **Gold savings scale directly with how good the
evaluator is**: `rho = 0.8` cuts the needed labels by roughly the factor baseline
currently wastes.

---

## How the two fixes compound, and what bounds the win

They stack. The control-variate payoff **shrinks the variance and tightens the
bound** on `g`; a tighter, lower-variance `g` then lets the WSR bet **grow
harder per step**. So the ~593 gold labels should drop substantially once both
are in.

The honest ceiling: the speedup is governed almost entirely by **predictor
quality**, scaling with `rho^2`. A sharp evaluator gives a large speedup; a
coin-flip evaluator gives none. PPI/CUPED are a free lunch *proportional to
`rho^2`* — never more. This is the number to measure empirically in enhanced: plot
gold-labels-to-decision against evaluator `rho`.

---

## Robustness bonus (PPI++), for free

Add the **PPI++** tuning weight — optimize how much to trust `f` in the
correction — and you gain a guarantee baseline lacks: if the evaluator turns out
useless, the weight goes to zero and the method **degrades gracefully to
gold-only inference, never worse**. baseline's raw importance weighting can actually
be *hurt* by a bad predictor; enhanced cannot. This directly addresses the documented
caveat that poor uncertainty scores break active inference — here, a bad
evaluator costs efficiency but never validity and never net harm.

---

## Scope note (what is and isn't the contribution)

WSR adaptive betting, the method of mixtures / mixture SPRT, PPI, PPI++,
prediction-powered e-values, and CUPED are all **established efficiency
machinery**. None of it is this project's novelty. The contribution remains the
**cost-aware, multi-source acquisition rule** layered on top — deciding which
evaluator to query and where to spend the scarce gold label to grow evidence
fastest per unit cost. enhanced exists only to make the engine underneath competitive
so that the acquisition rule has something sharp to optimize over.

---

## Acquisition: the third (and biggest) lever, with a surprise

Beyond the betting fraction and the payoff, *where* the scarce gold is spent is
the largest efficiency lever -- and getting it right was not what theory first
suggested. Measured on the synthetic comparison (H1: 0.75 vs 0.55, alpha 0.10):

| configuration                              | median gold to decision |
|--------------------------------------------|-------------------------|
| baseline (fixed bet + importance weighting), uniform audit | ~290 |
| enhanced (adaptive bet + control variate), uniform audit   | ~173 |
| enhanced + **principled acquisition**                       | **~71** |
| enhanced + uniform audit at the SAME budget                 | ~207 |
| enhanced + Neyman allocation (audit where uncertain)        | ~432, detection only ~0.86 |

Two findings, one positive, one a cautionary negative:

**Classical Neyman allocation backfires.** Auditing in proportion to residual
standard deviation (audit where the evaluator is *uncertain*) is variance-
optimal for *estimating* a mean, so it was the obvious "principled" choice. It
is the *worst* choice for sequential *detection* via a prediction-powered
e-process: it audits ambiguous items (f ~ 0.5) where the label is near a coin
flip, which starves the directional A-vs-B signal AND collapses the control
variate's tuning weight toward 0 (f does not predict y on those items). Its low
audit floor compounds the damage: the rare audits of decisive items carry huge
importance weight (1/pi), injecting high-variance payoffs that sabotage the bet.
Net effect: it detects far less often than plain uniform at matched budget.

**The rule that works: directional tilt + a high floor.** Audit *toward*
decisive items (large |f_A - f_B|), which carry the directional evidence and let
the control variate learn, while keeping a HIGH audit floor so importance
weights stay bounded. This reaches the decision on ~3x fewer gold labels than
uniform at matched budget. The invariant to preserve in any future rule:
**spend gold for wealth growth, not estimator variance, and keep the weights
bounded.**

**Honest note on the (1 - rho^2) claim.** The control-variate gold savings do
scale with evaluator quality, but in this synthetic regime the effect is
*modest* (~6% across rho in [0.55, 0.95] under uniform auditing), and once the
principled acquisition rule is in play the rho-dependence is absorbed almost
entirely into *items*-to-decision rather than *gold*-to-decision. So the
robustly true statement is "a better evaluator reaches the decision in fewer
items"; the gold-cost scaling is real but smaller than the (1 - rho^2) framing
implied. (All of the above is locked into the test suite, including the Neyman
negative result, so a future "improvement" cannot silently regress to it.)

## Build on (don't reimplement)

- **ppi-py** — the official PPI package (point estimates, PPI++ tuning).
- **Prediction-powered e-values** (Csillag et al., 2025) — the direct ancestor
  of the e-process payoff; extend it to the multi-source case.
- **WSR betting confidence sequences** — the adaptive-`lambda` construction.
- **CUPED** (Deng et al., 2013) — the control-variate framing of the rectifier.
