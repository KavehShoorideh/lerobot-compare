# lerobot-compare

**A label-efficient, statistically rigorous A/B test for robot and agent policies.**
Compare two policies on a success metric, get a clear verdict with confidence intervals and a
report in the style of modern experimentation platforms — and, when ground-truth labels are
expensive, reach that verdict using far fewer of them.

*A/B test = a controlled experiment comparing variant **A** against variant **B**. Here A and B
are two policies (e.g. two robot controllers or two agents) and the metric is task **success
rate**.*

---

## Contents

- [Why](#why) · [What you get](#what-you-get)
- [Installation](#installation) · [Quickstart](#quickstart)
- [Choosing a method: fixed-horizon vs. anytime-valid](#choosing-a-method-fixed-horizon-vs-anytime-valid)
- [Reading the result](#reading-the-result)
- [Label-efficient mode](#label-efficient-mode-a-cheap-judge--scarce-human-checks)
- [The statistical guarantee](#the-statistical-guarantee)
- [Command-line interface](#command-line-interface) · [Example notebooks](#example-notebooks)
- [Repository layout](#repository-layout) · [Development](#development)
- [Glossary](#glossary) · [References](#references)

---

## Why

You have two versions of a policy, A and B, and you want to know which is better at a task. For
some tasks a sensor tells you instantly whether an attempt worked. For many others — "folded the
towel *neatly*", "set the table *nicely*" — the only trustworthy judge is a human, and human time
is expensive. So you can afford only a handful of trustworthy labels, and must lean on
cheap-but-unreliable signals (a vision-language model, the robot's own sensors) for the rest.

lerobot-compare is built for exactly this regime. It is **a smarter A/B test, not a success
detector**: you supply a way to label a rollout; it supplies the label-efficient, guarantee-bearing
statistics on top, and a results report you can act on.

**When to use it**

| Situation | Recommendation |
|---|---|
| A cheap, reliable sensor already tells you success | You don't need this — just count sensor hits. |
| Success is verifiable but **expensive** to check on every rollout | **Use it** — a cheap judge stretches your few expensive checks. |
| Success is genuinely **human-judged** ("neatly", "carefully") | **Use it** — scarce human judgment is the only ground truth, and spending it efficiently is the whole game. |

## What you get

- **Two test methods, one call.** A classic **fixed-horizon** A/B test (the kind run by platforms
  such as [Statsig](https://statsig.com)), and an **anytime-valid** sequential test that lets you
  monitor results live and stop early. Switch with one argument.
- **A familiar results scorecard.** Per-variant success rates with confidence intervals, absolute
  and relative **lift**, a p-value or e-value, a **Sample Ratio Mismatch** health check, achieved
  **power** / **minimum detectable effect**, and a Bayesian "chance to beat" — modelled on how
  leading experimentation platforms report results.
- **Label efficiency.** An optional mode uses a cheap judge on every rollout and spends scarce
  human labels only where they move the decision, reaching a verdict on roughly **3× fewer** human
  labels than auditing at random — without ever trusting the cheap judge for correctness.
- **A real guarantee.** The false-positive rate is held at or below your chosen level regardless of
  how bad the cheap judge is; a useless judge costs you efficiency, never validity.

---

## Installation

```bash
pip install -e .            # core: numpy + scipy only
pip install -e ".[viz]"     # optional: matplotlib, for charts and the example notebooks
pip install -e ".[vlm]"     # optional: torch + transformers, for the vision-language-model judge
pip install -e ".[dev]"     # optional: pytest + ruff, for development
```

All dependencies are declared in [`pyproject.toml`](pyproject.toml); there is no separate
requirements file. The core imports **numpy and scipy only** — every heavy dependency (torch,
matplotlib, lerobot) is optional and lazily imported, so `import lerobot_doctor` works with just
the core installed.

---

## Quickstart

```python
import numpy as np
from lerobot_doctor import compare_success, Trial

# Build a paired dataset: the same scenes attempted by policy A and policy B. A shared
# `key` ties each pair together so the test can use the lower-variance paired analysis.
# In practice, replace these draws with your own per-rollout success labels.
rng = np.random.default_rng(0)
A = [Trial(success=int(rng.random() < 0.65), key=k) for k in range(800)]   # ~65% success
B = [Trial(success=int(rng.random() < 0.55), key=k) for k in range(800)]   # ~55% success

# Default method: the anytime-valid betting e-process (you may peek and stop early).
report = compare_success(A, B, alpha=0.05)
print(report.summary())

# Or the classic fixed-horizon A/B test — change one argument:
report = compare_success(A, B, method="fixed-horizon-proportion-z-test", alpha=0.05)
print(report.summary())
```

`compare_success` accepts either `Trial` objects (carrying match keys / covariates) or plain `0/1`
lists. A worked, runnable walkthrough is in
**[`examples/01_quickstart.ipynb`](examples/01_quickstart.ipynb)**.

The fixed-horizon call prints a scorecard like this:

```text
lerobot-compare | A/B result -- fixed-horizon / proportion z-test, 95% confidence
--------------------------------------------------------------------------------
  Decision: A>B is SIGNIFICANT  (p = 0.0009819)

  Variant      Success rate              95% CI
  A (test)     62.4%  (499/800)          [58.9%, 65.7%]
  B (control)  54.1%  (433/800)          [50.6%, 57.6%]

  Absolute lift (A-B):  +8.2 pts   95% CI [+3.3, +13.2]
  Relative lift:        +15.2%   95% CI [+6.2%, +24.3%]
  Standard error:       0.025
  Bayesian P(A>B):      99.9%
  Health:               sample-ratio OK (n_A=800, n_B=800, SRM p=1)
  Power:                0.91 achieved; MDE at 80% power = 7.0 pts
  Match:                exact (clean paired test)
  Rungs:                {'exact': 800, 'covariate': 0, 'unpaired': 0}
```

---

## Choosing a method: fixed-horizon vs. anytime-valid

The `method` argument takes a **full, descriptive name** that states both the *horizon* (when you
are allowed to look at the data) and the specific *test*:

| `method=` | Horizon | What it reports | You may peek? |
|---|---|---|---|
| `"anytime-valid-betting-e-process"` *(default)* | anytime-valid | **e-value** + widened CI | **Yes** — stop the moment it's conclusive |
| `"fixed-horizon-proportion-z-test"` | fixed-horizon | **p-value**, SE, power, MDE | No — analyse once, at the planned N |

`available_methods()` lists them; the names are designed so future tests (for example
`"anytime-valid-mixture-sprt"` or `"fixed-horizon-welch-t-test"`) slot in **without renaming**
anything that exists today.

**Which should you pick?** A common intuition is that the sequential (e-value) method always
reaches a verdict sooner. That is only half true, and worth understanding:

- At a **fixed sample size**, the fixed-horizon test is the more **powerful** one — it spends its
  entire error budget on a single look.
- The **anytime-valid** test must instead keep its guarantee valid at *every* look, so you may
  monitor continuously and stop early. It pays for that freedom by needing **more** evidence to
  reject. Concretely, a two-sided test at `alpha = 0.05` must drive its "wealth" (evidence) past
  `2 / alpha = 40`, not `1 / alpha = 20`, because it splits the budget across both directions.
- Its real payoff is therefore **early stopping** on clear effects (a strong effect can be called
  in *tens* of rollouts instead of hundreds) and the licence to watch live — **not** extra power
  at a pre-set N.

**[`examples/02_fixed_vs_anytime_valid.ipynb`](examples/02_fixed_vs_anytime_valid.ipynb)**
demonstrates this directly: it sweeps the effect size, plots when each method decides, and shows
the e-process's wealth crossing its bar.

| Situation | Use |
|---|---|
| One look at a pre-planned N | `fixed-horizon-proportion-z-test` (max power per look) |
| Monitor live, stop as soon as it's conclusive | `anytime-valid-betting-e-process` (peeking is free) |
| Effect likely **large** / labels expensive | `anytime-valid-betting-e-process` (stops early, saves data) |
| Effect **subtle**, N limited | `fixed-horizon-proportion-z-test` (resolves smaller effects at fixed N) |

---

## Reading the result

Both methods return the same `Report` object and print the same scorecard, so you can switch
methods and compare like for like. Field by field:

| Scorecard line | `Report` field(s) | Meaning |
|---|---|---|
| **Decision** | `decided`, `direction` | Whether a winner was called, and which way (`"A>B"` / `"B>A"`). |
| **Success rate** + per-variant CI | `rate_A`, `rate_B`, `rate_A_ci`, `rate_B_ci` | Each policy's success rate with an exact (Clopper–Pearson) **confidence interval (CI)**. |
| **Absolute lift (A-B)** | `effect`, `effect_ci_lo/hi` | The gap `rate_A − rate_B` in percentage **points**, with its CI. |
| **Relative lift** | `effect_relative`, `effect_relative_ci_lo/hi` | `(rate_A − rate_B) / rate_B` — Statsig's headline "Delta %". |
| **Standard error** | `standard_error` | **SE** of the effect *(fixed-horizon only)*. |
| (fixed) **p = …** | `p_value`, `z_score` | Fixed-horizon p-value and z-statistic. |
| (anytime-valid) **e-value … ≥ bar** | `e_value`, `e_threshold`, `p_value_equiv` | Evidence against "no difference", the decision bar `2/alpha`, and the anytime-valid p-value. |
| **Bayesian P(A>B)** | `chance_to_beat` | Posterior probability that A's true rate beats B's. |
| **Health: sample-ratio** | `srm_p_value`, `srm_flag` | **Sample Ratio Mismatch (SRM)** check — flags a suspiciously lopsided A/B split. |
| **Power / MDE** | `power`, `mde`, `target_power` | Achieved **power** and **minimum detectable effect (MDE)** *(fixed-horizon only)*. |
| **Match / Rungs** | `match_strength`, `rung_counts` | How the rollouts were paired (exact key, covariate match, or unpaired). |

`report.to_dict()` returns a JSON-serialisable dict of everything above.

---

## Label-efficient mode (a cheap judge + scarce human checks)

The front-door `compare_success` uses a label on every rollout. When trustworthy labels are
expensive, switch to the prediction-powered engine in `lerobot_doctor.core`: it runs a **cheap
judge** on every rollout and spends a **gold** (human) label only on the rollouts an *acquisition
rule* deems worth auditing — then corrects the cheap judge's bias with those few gold labels.

```python
import numpy as np
from lerobot_doctor.core import compare, make_world, CheapVLM

# `make_world` fabricates a paired benchmark with a graded cheap predictor (rho_acc is the
# judge's quality: 0.5 = useless, 1.0 = near-perfect) and a gold labeller. Swap these for your
# own evaluator + gold source on real data.
rng = np.random.default_rng(0)
items, predict, gold = make_world(pA=0.70, pB=0.55, rng=rng, rho_acc=0.8, N=1500)

result = compare(items, CheapVLM(), gold, predict, alpha=0.10, rng=rng)
print(result)   # Result(decided, direction, wealth, n_seen, n_audited, cost_spent)
# e.g. decided=True, direction='A>B', n_seen=672, n_audited=270  -> ~60% of gold labels saved
```

On real robotics data you would replace the three synthetic pieces with:

- an **evaluator** — any object with `score(rollout) -> float in [0, 1]` and a `cost`. A calibrated
  vision-language-model judge is provided in `lerobot_doctor.evaluators.vlm` (`RubricVLM`, which
  reads `P("yes")` from token logits over a rubric of sub-goal questions); a near-free judge built
  from policy internals is in `lerobot_doctor.evaluators.internal` (`InternalState`).
- a **gold source** — `lerobot_doctor.core.GoldSource(cost, label_fn)`, where `label_fn(item)`
  returns the human's `(success_A, success_B)` for an audited pair.
- a **`predict(item, evaluator)`** function returning the evaluator's `(score_A, score_B)`.

The core never knows whether a score came from a vision model, a camera classifier, or a
proprioceptive heuristic — it only needs a number in `[0, 1]` and a cost.

---

## The statistical guarantee

The false-positive rate is **≤ α at the relevant stopping rule** — for the anytime-valid method,
at *every* stopping time simultaneously (via a betting e-process and Ville's inequality), so you
may peek and stop early without penalty. Crucially, this holds **regardless of the cheap judge's
quality**: a useless judge slows you down but cannot make the test wrong, because the scarce gold
labels anchor the truth and the prediction-powered correction is unbiased by construction. The
cost-aware acquisition rule reaches a decision on roughly **3× fewer** gold labels than uniform
auditing at a matched budget. (A documented negative result: classical Neyman allocation, which is
optimal for *estimation*, actually *backfires* for sequential *detection* — see the
[`_legacy/paper.md`](_legacy/paper.md) writeup.)

---

## Command-line interface

```bash
# per-rollout 0/1 labels, one per line (a CSV with a 'success'-like column also works)
lerobot-compare compare A.txt B.txt --method fixed-horizon-proportion-z-test --alpha 0.05
lerobot-compare compare A.txt B.txt --alpha 0.05 --json report.json     # default = anytime-valid
```

Flags: `--method` (see `--help` for the choices), `--alpha`, `--alternative {two-sided,A>B,B>A}`,
`--target-power`, `--json OUT`, `--charts OUT.png`. Exit status is `0` when a decision is reached
and `2` otherwise, which is convenient for gating CI.

## Example notebooks

Runnable, output-populated notebooks live in [`examples/`](examples/) (run them with the `[viz]`
extra plus Jupyter):

- **[`01_quickstart.ipynb`](examples/01_quickstart.ipynb)** — build a dataset, run both methods,
  read the scorecard.
- **[`02_fixed_vs_anytime_valid.ipynb`](examples/02_fixed_vs_anytime_valid.ipynb)** — how the two
  methods differ as the effect size grows, and why early stopping is the anytime-valid payoff.

---

## Repository layout

```
src/lerobot_doctor/
  compare.py        # front door: compare_success(method=...) + the method registry
  fixed_horizon.py  # fixed-horizon paired / two-proportion z-test, power, MDE
  bayesian.py       # Bayesian chance-to-beat (Beta-Binomial)
  intervals.py      # confidence intervals + anytime-valid confidence sequence
  report.py         # the Report dataclass and the Statsig-style scorecard
  matching.py       # pair A/B rollouts (exact key -> covariate -> unpaired)
  checks.py         # preflight diagnostics + Sample Ratio Mismatch check
  cli.py            # the `lerobot-compare` command
  viz.py            # optional matplotlib dashboard ([viz] extra)
  core/             # domain-agnostic engine (numpy only):
                    #   eprocess.py, payoff.py, auditing.py, compare.py
  evaluators/       # pluggable judges: stub, vlm (Qwen2.5-VL), internal, calibrate
examples/           # runnable notebooks
tests/doctor/       # test suite (statistical guarantees + reporting)
```

## Development

```bash
pip install -e ".[dev]"
python -m pytest -q                 # full suite
python -m pytest -q -m "not slow"   # skip the Monte-Carlo validity tests
ruff check src tests                # lint
```

---

## Glossary

Operational definitions — what each term means *in this tool*, not textbook formalism. Acronyms
are expanded on first use throughout the docs and code; this is the consolidated list.

### A/B testing & reporting

- **A/B test.** A controlled experiment comparing variant A against variant B. Here: two policies,
  metric = success rate.
- **horizon.** *When* you may analyse. **Fixed-horizon**: sample size fixed in advance, analyse
  once. **Anytime-valid**: the guarantee holds at every sample size, so you may analyse
  continuously and stop early.
- **method name.** The `method=` value encodes `"<horizon>-<test>"`, e.g.
  `"fixed-horizon-proportion-z-test"`. New tests register under new names so existing ones never
  need renaming.
- **lift / effect.** The difference between policies. **Absolute lift** = `rate_A − rate_B` (in
  percentage *points*). **Relative lift** (a.k.a. **Delta %**) = `(rate_A − rate_B) / rate_B`.
- **CI — confidence interval.** A range of plausible values for an effect at a stated confidence
  level. A 95% CI that excludes 0 means the effect is significant at α = 0.05.
- **SE — standard error.** The standard deviation of an estimate; the CI half-width is roughly
  `1.96 × SE` for a 95% two-sided interval.
- **α (significance level) / confidence level.** α is the tolerated false-positive rate (e.g.
  0.05); the confidence level is `1 − α` (e.g. 95%).
- **power / MDE — minimum detectable effect.** Power is the probability of detecting a real effect
  of a given size; the MDE is the smallest effect a design can detect at a target power (default
  80%). Reported for the fixed-horizon method.
- **SRM — Sample Ratio Mismatch.** A chi-square health check that the A vs. B rollout counts match
  the intended split (e.g. 50/50). A flag means the data collection may be biased — investigate
  before trusting the result.
- **p-value.** The probability, if there were truly no difference, of evidence at least this
  extreme. Reject the null when `p < α`. (Fixed-horizon.)
- **Type-I / Type-II error.** Type-I = a false positive (calling a difference that isn't there);
  Type-II = a false negative (missing a real one). α bounds Type-I; power is `1 − P(Type-II)`.
- **two-proportion z-test / McNemar's test.** The fixed-horizon engine. For paired binary data it
  is McNemar's test (it looks at the pairs where A and B disagree); for unpaired data it reduces to
  the classic two-proportion z-test.
- **Bayesian chance-to-beat / Beta-Binomial.** The posterior probability `P(p_A > p_B)` under a
  Beta-Binomial model (Beta is the conjugate prior for a binomial rate). An intuitive companion to
  the frequentist verdict.
- **SPRT / mSPRT.** The Sequential Probability Ratio Test and its mixture variant (mSPRT) — the
  classical likelihood-ratio sequential tests. Statsig's sequential mode uses mSPRT; lerobot-compare
  uses the closely related **betting e-process**, a modern, assumption-light member of the same
  anytime-valid family.
- **CUPED.** "Controlled-experiment Using Pre-Experiment Data" — a variance-reduction technique.
  lerobot-compare's matching and control-variate payoff play the same role.
- **VLM — vision-language model.** A model that scores an image/video against a text prompt; used
  here as a cheap, fallible judge of task success.
- **Statsig.** A widely used experimentation platform; the results scorecard here is modelled on
  how it (and peers) report A/B results.

### Anytime-valid / betting internals

- **e-value / e-process.** The evidence-against-the-null object: a running **wealth** that starts
  at 1; reject the null when it reaches `1/α` (one-sided) or `2/α` (two-sided). An e-value can be
  monitored continuously and stopped anytime — the contrast with a p-value. Class:
  `core.eprocess.BettingEProcess`.
- **wealth.** Imaginary money in a bet against the null, starting at 1; each rollout multiplies it
  by `(1 + λ_t·g_t)`. Stored in log space for numerical stability.
- **martingale / supermartingale.** A process whose expected next value, given the past, equals
  (martingale) or is at most (supermartingale) its current value. The wealth is a supermartingale
  *under the null* — the single property that makes the guarantee hold.
- **Ville's inequality.** For a non-negative supermartingale starting at 1, the probability it
  *ever* crosses `1/α` is ≤ α. That "ever" is why you may peek and stop early — i.e.
  *anytime-valid*.
- **payoff (`g_t`).** One rollout's bet result in `[−1, 1]`, positive when the evidence favours A.
  Its mean under the null must be ≤ 0.
- **betting fraction (`λ_t`).** How much wealth is staked each round; adapted online (Kelly-style)
  and kept below 1 so a single loss can't bankrupt the bet.
- **predictable (previsible).** A quantity computed from the *past only*, never the current
  rollout's gold label. The audit probability, the tuning weight, and the betting fraction must all
  be predictable, or validity is lost. The #1 source of subtle bugs.
- **gold / audit / π.** "Gold" = the scarce trustworthy (human) label. "Auditing" a rollout =
  spending one gold label on it. `π` is the predictable probability of auditing a rollout.
- **PPI — prediction-powered inference / control variate / λ_PP.** Use the cheap judge on every
  rollout and gold only to *correct* it, so the estimate is unbiased no matter how bad the judge is.
  `λ_PP` is how much to trust the judge (≈1 when good, ≈0 when useless — graceful degradation).
- **IPW — inverse-probability weighting / importance weight (`1/π`).** Audited corrections are
  divided by `π` to stay unbiased; a low audit floor makes `1/π` large and high-variance, which is
  why a sensible audit floor matters.
- **acquisition rule.** The policy that decides which rollouts to audit (e.g.
  `MaxEvidenceAuditor`, the recommended one).
- **rho / rho_acc.** In the synthetic world, the cheap judge's quality: 0.5 = useless coin flip,
  1.0 = near-perfect. Affects efficiency only, never validity.
- **rollout / pair.** A rollout is one whole-task attempt (one label). A pair is the same task done
  by A and B — the unit the comparison consumes.

---

## References

lerobot-compare assembles known statistics rather than inventing new ones, adding a cost-aware
acquisition rule on top.

**Entry points into the anytime-valid / betting worldview**

- Ramdas, Grünwald, Vovk & Shafer, *Game-Theoretic Statistics and Safe Anytime-Valid Inference*,
  **Statistical Science** (2023) — the readable overview of e-values and testing-by-betting.
  <https://doi.org/10.1214/23-STS894>
- Waudby-Smith & Ramdas, *Estimating Means of Bounded Random Variables by Betting*, **JRSS-B**
  (2024) — the wealth picture and adaptive bet used here. <https://arxiv.org/abs/2010.09686>

**Methods assembled**

- Angelopoulos, Bates, Fannjiang, Jordan & Zrnic, *Prediction-Powered Inference*, **Science**
  (2023), and Angelopoulos, Duchi & Zrnic, *PPI++* (2023) — unbiased estimates from a cheap model
  plus a little trustworthy data, with a tuning weight that makes a useless predictor harmless.
  Code: <https://github.com/aangelopoulos/ppi_py>
- Csillag, Struchiner & Goedert, *Prediction-Powered E-Values*, **ICML** (2025) — the direct
  ancestor of the engine here. <https://arxiv.org/abs/2502.04294>
- Deng, Xu, Kohavi & Walker, *Improving the Sensitivity of Online Controlled Experiments by
  Utilizing Pre-Experiment Data (CUPED)*, **WSDM** (2013) — the variance-reduction idea.
- Wald, *Sequential Tests of Statistical Hypotheses*, **Ann. Math. Statist.** (1945), and Robbins,
  *Statistical Methods Related to the Law of the Iterated Logarithm* (1970) — the classical SPRT /
  mSPRT lineage that the betting e-process generalises.
- Chernoff, *Sequential Design of Experiments*, **Ann. Math. Statist.** (1959) — the classical
  "which costly source to query, and when to stop" theory the acquisition rule modernises.
- Ville, *Étude critique de la notion de collectif* (1939) — the inequality the anytime-valid
  guarantee rests on.

**Tools / implementations**

- `ppi-py` — PPI / PPI++: <https://github.com/aangelopoulos/ppi_py>
- `confseq` — betting confidence sequences / e-processes: <https://github.com/gostevehoward/confseq>
- Qwen2.5-VL — the vision-language-model judge: <https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct>
- LeRobot — robot datasets and policies: <https://github.com/huggingface/lerobot>
