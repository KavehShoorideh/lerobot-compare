# lerobot-compare: Cost-Aware, Anytime-Valid Comparison of Robot and Agent Policies from Cheap, Unreliable Signals

*A working paper / reproducible report. All numbers below are produced by
`experiments.py`; every claim is backed by named tests in `test_lerobot-compare.py`
and `test_robotics_adapter.py` (see the reproducibility table in Section 8).*

---

## Abstract

Evaluating whether one robot or agent policy is better than another is bottlenecked
by labels: for tasks without a cheap verifiable success signal ("fold the towel
neatly"), the only trustworthy label is scarce, expensive human judgment, while cheap
signals — a vision-language-model (VLM) judge, the robot's own internal state — are
abundant but unreliable. We present **lerobot-compare**, a small library that compares two
policies with an **anytime-valid** statistical guarantee while spending as few
trustworthy labels as possible. It combines a betting e-process (valid under optional
stopping) with a prediction-powered payoff (unbiased *regardless of how bad the cheap
evaluator is*) and a **cost-aware acquisition rule** that decides which rollouts are
worth a human label. On synthetic benchmarks the full method reaches the same decision
as label-everything evaluation on **~4x fewer** trustworthy labels than a naive
sequential test, and **~3x fewer** than uniform auditing *at matched budget*. We also
report a cautionary **negative result**: the textbook "principled" choice — Neyman /
active-inference allocation, which audits where the evaluator is most uncertain — is
*detection-pessimal* here and underperforms even uniform auditing. The honest scope:
all results are on synthetic data with a graded predictor; the real test is a VLM on
real rollouts, for which an adapter is provided.

---

## 1. Problem and setting

A practitioner has two policies, A and B, and wants to know which is better at a task.
We make three modeling choices that define the scope precisely:

- **A rollout is one whole-task attempt and carries one success label.** Subgoals
  inside an attempt are *dependent* (you cannot place the cube if you did not open the
  drawer), so they are not separate units; they are folded into the single per-rollout
  score (e.g., a graded "progress depth"). Betting at the rollout level contains that
  within-attempt dependence, because validity only needs independence *across*
  attempts.
- **We compare; we do not detect.** lerobot-compare assumes you already have *some* way to
  label a rollout. Its job is the statistics on top: is A better than B, and how few
  expensive labels are needed to be sure.
- **It matters only when trustworthy labels are scarce.** If a cheap reliable sensor
  verifies success, just count sensor hits. lerobot-compare earns its keep on non-verifiable,
  human-judged tasks, where the gold label is costly and the cheap signals are noisy.

The label sources ("routes") differ in cost and reliability: a VLM judge (cheap,
unreliable), the robot's internal state (near-free, indirect), and scarce human gold
(expensive, trusted). The question: spend the gold where it buys the most certainty.

## 2. Method

### 2.1 Estimand

We test the one-sided null **H0: p_A ≤ p_B** (A is not better) against A > B, where
p_P is the success rate of policy P. Two-sided use runs the symmetric process as well.

### 2.2 Betting e-process (where the guarantee comes from)

Inference is a bet against the null. Wealth starts at 1 and updates multiplicatively,
`W_t = W_{t-1}·(1 + λ_t·g_t)`, with payoff `g_t ∈ [-1,1]` and betting fraction
`λ_t ∈ [0, λ_max]`. If the payoff is unfavorable under the null,
`E[g_t | past] ≤ 0`, then `(W_t)` is a nonnegative supermartingale from 1, and Ville's
inequality bounds the probability it *ever* reaches `1/α` by `α`. We declare A > B the
first time `W_t ≥ 1/α`; the false-positive rate is then `≤ α` **at every stopping
time**, which is what licenses peeking and early stopping. The betting fraction `λ_t`
is the Waudby-Smith–Ramdas predictable Kelly approximation, learned online from past
payoffs (validity holds for any *predictable* `λ_t`).

Empirically, under H0 the wealth is a supermartingale (median final wealth ≈ 1.0 and
it crosses the `1/α` threshold with probability ≈ 0), confirming the guarantee.

![Wealth is a supermartingale under H0](../figures/fig6_supermartingale.png)

*Backed by `test_F1_supermartingale_under_h0`.*

### 2.3 Prediction-powered payoff (why a bad evaluator can't break validity)

The payoff is built from a prediction-powered score that uses the cheap evaluator on
every rollout and the gold label only on *audited* rollouts:

```
m_P = λ_PP·f_P + 1{audited}/π · (y_P − λ_PP·f_P),      g ∝ (m_A − m_B)
```

The evaluator's prediction `f_P` **cancels in expectation** — `E[m_P] = p_P` for any
`λ_PP` and any `f` — so a biased, miscalibrated, or useless evaluator changes only the
*variance*, never the mean, and hence never the validity. The tuning weight `λ_PP` (a
predictable regression slope of `y` on `f`) is the PPI++ control variate: it goes to 1
for a good evaluator and to 0 for a useless one, so the method **degrades gracefully**
to gold-only inference and is never hurt by a bad predictor.

### 2.4 Cost-aware acquisition (the contribution)

The novel piece is deciding *which* rollouts to spend gold on. The acquisition rule
must be **predictable** (depend only on the cheap predictions, never on the gold about
to be bought) or validity breaks. The rule that works: tilt audits toward **decisive**
rollouts (large `|f_A − f_B|`), which carry the directional A-vs-B evidence and let the
control variate learn, while keeping a **high audit floor** so importance weights
`1/π` stay bounded. The governing invariant, established empirically below: *spend gold
for wealth growth, not estimator variance, and keep the weights bounded.*

## 3. Three efficiency levers

Three independent choices reduce the gold needed: (i) the **betting fraction** (fixed
0.5 → adaptive Kelly), (ii) the **payoff** (importance weighting → control variate /
PPI++), and (iii) the **acquisition rule** (uniform → decisive-tilt). Their cumulative
effect on median gold labels to decision (H1: p_A = 0.75 vs p_B = 0.55, α = 0.10):

![Efficiency ladder](../figures/fig1_efficiency_ladder.png)

The full method reaches the decision on **~70** gold labels, versus **~292** for the
naive baseline sequential test — and versus **~210** for uniform auditing *at the same
budget* as the principled rule (the fair ~3x comparison). *Backed by
`test_F2_v2_beats_v1`, `test_F2_control_variate_lever_helps`,
`test_F2_adaptive_bet_lever_helps`, `test_F3_principled_beats_uniform_gold`,
`test_F3_principled_beats_uniform_items`, `test_F3_budget_is_actually_matched`.*

## 4. Experiments and results

### 4.1 Validity is unconditional

Across acquisition rules (uniform, principled, Neyman) and evaluator qualities
including a useless coin-flip evaluator (`rho = 0.5`), the false-positive rate under H0
stays below `α`. The maximum observed false-positive rate over all conditions is
**0.036** against `α = 0.10`.

![Validity across evaluator quality and rule](../figures/fig2_validity.png)

*Backed by `test_F1_type_i_good_evaluator`, `test_F1_type_i_useless_evaluator`,
`test_F1_type_i_v1_estimator`, `test_F1_type_i_all_acquisition_rules`,
`test_F1_type_i_second_alpha`, `test_F1_payoff_unbiased_despite_biased_evaluator`.*

### 4.2 The principled rule beats uniform at matched budget

Measuring the principled rule's realized audit rate and running uniform auditing at
*that same rate*, the principled rule reaches the decision on fewer gold labels **and**
fewer rollouts — the win is from *where* gold is spent, not *how much*. This holds at a
second effect size as well. *Backed by the `test_F3_*` family.*

### 4.3 Negative result: classical Neyman allocation backfires

The obvious "principled" rule is Neyman / active-inference allocation: audit in
proportion to residual standard deviation, i.e. where the evaluator is most
*uncertain*. This is variance-optimal for *estimating* a mean — and it is the **worst**
rule for sequential *detection*. At matched budget it detected only **~20%** of true
effects (vs 100% for uniform and the principled rule) and spent **~356** gold labels.

![Neyman backfires](../figures/fig3_neyman_backfires.png)

The mechanism: auditing ambiguous rollouts (`f ≈ 0.5`, where `y` is near a coin flip)
(a) starves the directional A > B signal and (b) collapses the control variate's
`λ_PP` toward 0 (on those rollouts `f` does not predict `y`), while a low audit floor
lets the rare audits of decisive rollouts carry huge importance weight, injecting
variance that sabotages the bet. The lesson — **estimation-optimal is
detection-pessimal** for prediction-powered e-processes — is locked into tests so a
future "improvement" cannot silently regress to it. *Backed by
`test_F4_neyman_detects_worse_than_uniform`,
`test_F4_neyman_spends_more_gold_than_principled`,
`test_F4_mechanism_lam_pp_collapses_on_uncertain_items`.*

### 4.4 Graceful degradation

The PPI++ tuning weight collapses to ~0 for a useless evaluator and rises toward 1 for
a good one, so a bad evaluator costs efficiency but never validity, and never makes the
method worse than gold-only. *Backed by `test_F5_lam_pp_to_zero_for_useless_evaluator`,
`test_F5_lam_pp_to_one_for_good_evaluator`, `test_F5_useless_evaluator_still_valid`.*

### 4.5 Evaluator quality, honestly

A better evaluator reaches the decision in monotonically fewer **rollouts**; the effect
on **gold count** is real but modest in this regime (the smart acquisition rule absorbs
most of the quality dependence into rollouts-seen). We report the robustly true
statement, not the inflated one.

![Evaluator quality effect](../figures/fig4_quality_effect.png)

*Backed by `test_F6_items_monotone_in_rho`, `test_F6_gold_not_increasing_in_rho`.*

### 4.6 Savings across effect sizes

The principled rule spends less gold than uniform-at-matched-budget across a range of
true effect sizes.

![Savings vs effect size](../figures/fig5_savings_vs_effect.png)

## 5. Discussion

The central design insight is that **the objective is wealth growth, not estimation
precision**, and those call for different gold allocations. Minimizing the variance of
an estimate (Neyman) starves the directional evidence a sequential test needs and, with
importance weighting, injects variance that the bet cannot survive. The allocation that
works tilts toward decisive evidence and bounds the weights. Uniform auditing is a
surprisingly strong baseline precisely because it never starves decisive rollouts and
keeps weights homogeneous; beating it required understanding *why* it is strong.

## 6. Limitations and scope

- **Synthetic evaluator.** All results use a graded synthetic predictor whose quality
  is a knob. The decisive-tilt and high-floor constants are tuned to that world; the
  real test is a VLM (e.g., Qwen2.5-VL) on real rollouts, where the tilt/floor must be
  retuned. The `test_F3_*` matched-budget tests are the harness for that retuning.
- **One scalar label per rollout.** The method evaluates a single success-ish number in
  [0,1]; richer per-subgoal verdicts require either an ordinal progress-depth score
  (recommended) or separate comparisons per subgoal.
- **Paired rollouts.** The comparison assumes A and B are run on matched task
  instances. Unpaired streaming is straightforward future work.
- **No counterfactuals.** lerobot-compare evaluates trajectories that actually happened; it
  does not infer "what if the policy had done X," which is a separate, harder problem.

## 7. Relation to prior work

lerobot-compare sits at the intersection of three lineages. Classical **controlled sensing /
active sequential hypothesis testing** (Chernoff 1959 onward) studies which costly
source to query and when to stop, under *known* source models and asymptotic
optimality; we replace known models with *black-box, learned-reliability* evaluators
and asymptotic optimality with *finite-sample, distribution-free* e-value validity.
**Prediction-powered inference** (PPI, PPI++) and **prediction-powered e-values**
supply the unbiased-under-any-predictor payoff and the anytime-valid machinery, for a
*single* predictor; our contribution is the cost-aware allocation across *heterogeneous*
evaluators. The control variate is **CUPED**; the adaptive bet is the
**Waudby-Smith–Ramdas** betting confidence sequence. (`confseq` and `ppi-py` are
drop-in implementations of those pieces; they are not in this sandbox so the two needed
constructions are implemented directly and validated by the test suite.)

## 8. Reproducibility

Run `python3 experiments.py` to regenerate every figure and `results.json`; run
`python3 -m pytest -q` for all 28 + 7 tests. Mapping from claim to evidence:

| Finding | Claim | Tests | Figure |
|---|---|---|---|
| F1 | type-I ≤ α for any evaluator (incl. useless) and any rule | `test_F1_type_i_*` (6), `test_F1_supermartingale_under_h0` | fig2, fig6 |
| F2 | enhanced (adaptive bet + control variate) beats baseline; each lever helps | `test_F2_v2_beats_v1`, `test_F2_*_lever_helps`, `test_F2_v2_beats_v1_second_effect` | fig1 |
| F3 | principled acquisition beats uniform at matched budget | `test_F3_principled_beats_uniform_gold`/`_items`/`_budget_is_actually_matched`/`_holds_at_second_effect` | fig1, fig5 |
| F4 | Neyman allocation backfires (estimation-optimal ≠ detection-optimal) | `test_F4_neyman_detects_worse_than_uniform`, `test_F4_neyman_spends_more_gold_than_principled`, `test_F4_mechanism_lam_pp_collapses_on_uncertain_items` | fig3 |
| F5 | graceful degradation: a useless evaluator never hurts | `test_F5_lam_pp_to_zero_*`, `test_F5_lam_pp_to_one_*`, `test_F5_useless_evaluator_still_valid` | fig2 |
| F6 | better evaluator → fewer rollouts (modest gold effect) | `test_F6_items_monotone_in_rho`, `test_F6_gold_not_increasing_in_rho` | fig4 |
| F7 | predictability, safe betting, determinism, core purity | `test_F7_predictability_*`, `test_F7_betting_lambda_safe_range`, `test_F7_determinism`, `test_F7_core_is_numpy_only` | — |
| robotics | adapter wires real evaluators without breaking core purity or validity | `test_robotics_adapter.py` (7) | — |

Headline numbers (this run): efficiency ladder 292 → 170 → **70** gold; uniform at
matched budget 210; Neyman 356 at 0.20 detection; max false-positive rate 0.036 at
α = 0.10; H0 median wealth 1.00.
