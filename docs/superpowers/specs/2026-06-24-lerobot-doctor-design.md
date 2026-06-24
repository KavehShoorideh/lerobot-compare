# lerobot-compare — unified evaluation framework: design

*Status: approved design, pre-implementation. Date: 2026-06-24.*

## 1. Goal

A single, unified `lerobot-compare` repo that:

1. **Compares two policies** with an anytime-valid statistical guarantee and shows
   one is better than the other, spending as few human labels as possible.
2. **Attaches a VLM judge** and runs a human-in-the-loop training cycle in which
   **every human label is used to the maximum**.
3. **Notifies a human** asynchronously when there is something worth labeling.
4. Is laid out so it can be **PR'd into `huggingface/lerobot` with minimal changes**.

It must run **end-to-end on a Windows laptop today** (synthetic backends, no GPU)
and **graduate to real execution** (real sim, real VLM, real training) by swapping
backends — with no change to the loop logic.

### Non-goals

- No new statistical research / paper. The technique (prediction-powered inference
  + betting e-values + CUPED + cost-aware sensing) is **known prior work**; we are
  bringing it to lerobot, not claiming novelty.
- No counterfactual / off-policy evaluation. We judge trajectories that happened.
- No attempt to run real lerobot sim or VLM weights locally on Windows.

## 2. Starting point (what exists, and what changes)

The repo currently holds **two overlapping families** wrapped around one good core.

- **The core** is duplicated: `engine.py` and `parsimony.py` are the same
  anytime-valid betting e-process + prediction-powered payoff + cost-aware
  acquisition rule. **`parsimony.py` is deleted** (dup).
- **Family A — "parsimony" research artifact:** `paper.md`, `v2-efficiency.md`,
  `experiments.py`, figures, the F1–F7 test suite, and `parsimony_robotics.py`
  (a `RubricVLM` + a LeRobot episode loader). **Demoted** (see §3, §10).
- **Family B — the `lerobot_doctor` package:** `__init__.py`, `cli.py`,
  `matching.py`, `report.py`, `evaluators.py` (incl. a real `VLMJudge`),
  `calibrate.py`, `training.py`, `checks.py`, plus v1/v2 demos and tests. **This is
  the spine we unify onto.**

Everything runnable today is synthetic (`StubVLM`, `make_world`) — numpy only.

### Decision: unify onto Family B (the package), not a clean-room rewrite

Lower risk; keeps every test that already locks in the guarantees. Fold
`parsimony_robotics.py`'s `RubricVLM` + LeRobot loader into the evaluators; keep
the F1–F7 tests (reframed as guarantee tests).

## 3. Naming and terminology

- Package: **`lerobot_doctor`** standalone → **`lerobot.doctor`** vendored (§4).
- CLI: **`lerobot-compare`** (sibling to lerobot's existing `lerobot-eval`). We do
  **not** extend `lerobot.scripts.eval` (that is single-policy sensor-success-rate
  eval — a different thing).
- **`parsimony` → `auditing`** everywhere (filenames, identifiers, docs). The
  acquisition concept *is* auditing: it decides which rollouts are worth a costly
  human label. `core/acquisition.py` → `core/auditing.py`; the `AcquisitionRule`
  protocol → **`Auditor`**; `MaxEvidencePerCost`/`UniformAudit`/`NeymanAcquisition`
  → `MaxEvidenceAuditor` / `UniformAuditor` / `NeymanAuditor`.
- **No paper.** `paper.md` + `v2-efficiency.md` → one short `docs/method.md`: "known
  technique, brought to lerobot," with references retained and the Neyman-backfire
  kept as a one-paragraph caveat (+ its test). No novelty claims.

## 4. Repo layout (standalone now, vendored later via codemod)

"Standalone repo" and "lives inside lerobot at `src/lerobot/doctor/`" conflict (an
installed real `lerobot` would collide). Resolution:

- Develop under **`src/lerobot_doctor/`** — standalone-installable, no collision —
  structured **1:1 identical** to the eventual `lerobot/doctor/`.
- Ship **`scripts/vendor_to_lerobot.py`**: rewrites `lerobot_doctor` →
  `lerobot.doctor` and emits the subtree ready to drop into a lerobot checkout.
- **The lerobot PR = run codemod, copy subtree under `src/lerobot/doctor/`, add two
  `pyproject.toml` entries** (optional `doctor`/`vlm` extras + the `lerobot-compare`
  script). That is the "minimal changes" promise, made concrete and testable.

```
lerobot-compare/                          # standalone repo (this repo)
  src/lerobot_doctor/                    # mirrors eventual lerobot/doctor/ EXACTLY
    __init__.py                          # curated public API (see §11)
    core/                                # domain-agnostic stats, numpy-only
      eprocess.py                        # BettingEProcess  (from engine.py)
      payoff.py                          # Comparison / PPI++ payoff (from engine.py)
      auditing.py                        # Auditor protocol + 3 rules (was acquisition)
      compare.py                         # compare() engine + Result (from engine.py)
    evaluators/
      base.py                            # Evaluator/Rollout protocols, HumanGold
      stub.py                            # StubVLM (synthetic, runs here)
      vlm.py                             # VLMJudge + RubricVLM (Qwen2.5-VL; lazy torch)
      internal.py                        # AutomaticChecker, InternalState
      calibrate.py                       # isotonic Recalibrate / CalibratedEvaluator
    sim/                                 # goal 1's "simulation"
      base.py                            # RolloutSource interface
      synthetic.py                       # success-rate sim (runs here; was make_world)
      lerobot_env.py                     # real gym env + 2 checkpoints (offloaded; lazy)
    loop/                                # goals 2 & 3 — the new heart
      store.py                           # LabelStore (durable jsonl queue + answers)
      notify.py                          # async Notifier + Sink protocol + StdoutSink
      update.py                          # Updater: triple-use of each label
      train.py                           # LoRA retrain handoff (was training.py)
      engine.py                          # batch cycle driver
      online.py                          # thin online driver (feed-one, block-on-queue)
    matching.py                          # covariate pairing (from matching.py)
    report.py                            # Report (from report.py)
    checks.py                            # success-flag preflight policy (from checks.py)
    viz.py                               # optional charts (from viz.py)
    config.py                            # draccus config dataclasses
  src/lerobot/scripts/doctor.py          # `lerobot-compare` CLI: compare | loop | label
  tests/doctor/                          # F1–F7 + v1/v2/checks/viz + new loop tests
  bench/validate.py                      # optional: regenerate guarantee plots (was experiments.py)
  docs/method.md                         # short method note + references (was paper.md)
  examples/                              # v2_demo, offline_compare_stub, robotics_qwen_lerobot
  scripts/vendor_to_lerobot.py           # the codemod
  pyproject.toml  README.md
```

Repo-root cleanup: delete stray build/cache artifacts now tracked-or-untracked
(`*.pyc`, `parsimony.zip`, `CACHEDIR.TAG`, loose `nodeids`/`lastfailed`,
`mnt/`, top-level generated `judge_data.jsonl` / `dataset_info.json` /
`train.yaml` / `sky_task.yaml` / generated `fig*.png` / `sample_report.png`) and
add a `.gitignore`. Keep `robotics_qwen_lerobot.py`, `offline_compare_stub.py`,
`compare_demo.py`, `v2_demo.py` by moving them into `examples/`.

## 5. The core (unchanged behavior, relocated + renamed)

`core/` is lifted from `engine.py` and split by responsibility. **No behavior
change** — the F1–F7 tests must pass identically after the move. The validity
guarantee and the documented Neyman negative result are preserved.

- `BettingEProcess`: wealth supermartingale; reject H0 when wealth ≥ 1/α.
- `Comparison`: PPI++ prediction-powered payoff; unbiased for any evaluator.
- `Auditor` (protocol) + `MaxEvidenceAuditor` (default), `UniformAuditor`,
  `NeymanAuditor` (kept only as the documented-negative baseline).
- `compare(items, evaluator, gold, predict, rule, ...)` → `Result`.

## 6. The human-in-the-loop ("both, one interface")

Batch cycle is the engine; online is a thin driver. Both share three components.

### 6.1 LabelStore (`loop/store.py`)

Durable append-only `jsonl` per run: one record per *item flagged for audit* —
`{item_id, f_A, f_B, question, frames_ref, status: pending|answered, y_A, y_B, ts}`.
- **Dedup:** an item is never enqueued twice (keyed by `item_id`).
- **Persistence:** a crash/restart resumes from the same store; no label lost.
- API: `enqueue(item)`, `pending()`, `answer(item_id, y_A, y_B)`,
  `answered_records()`.

### 6.2 Notifier (`loop/notify.py`) — async, non-blocking

- `notify(notification)` **returns immediately** (puts the event on an internal
  `queue.Queue`); a background **daemon worker thread** drains it to a `Sink`. A
  slow sink (e.g. a future Telegram network call) never stalls the experiment.
- `Sink` protocol: `emit(Notification) -> None`. Ships with **`StdoutSink`**
  (prints a banner + refreshes the run's `queue.jsonl` path). `QueueSink` /
  `TelegramSink` / `WebhookSink` drop in later behind the same interface.
- Context-manager (`__enter__`/`__exit__`) flushes the queue and joins the worker
  on shutdown. (Thread+queue, not asyncio, so sync CLI callers stay sync.)
- `Notification` dataclass: `{event, count, queue_path, payload, ts}`.

### 6.3 Updater (`loop/update.py`) — "use every label to the maximum"

One human answer is used **three ways**, from one click:
1. **Gold** in the e-process bet (the audit itself — drives the A-vs-B decision).
2. **Recalibrate** the cheap judge instantly (isotonic map; no GPU; `calibrate.py`).
3. Appended to the **LoRA retrain pool** for the next VLM fine-tune (`train.py`).

Nothing collected is single-use. `Updater.ingest(record)` fans the label out to all
three; `Updater.refresh_judge()` returns an improved evaluator (recalibrated always;
retrained when the cadence/threshold is met and a GPU backend is available).

### 6.4 Drivers

- **Batch (`loop/engine.py`):**
  `score all → auditor flags a batch → notify once → human answers via store →
  Updater.ingest each → refresh_judge → e-process update over the round → decided?`
  Repeat rounds until decided or items exhausted.
- **Online (`loop/online.py`):** feeds one item at a time, blocks on the store for
  that item's answer, refreshes the judge on a cadence. Same components, same
  guarantees; only batch-size and whether it blocks differ.

### 6.5 Validity invariant (hard requirement, test-locked)

The judge may be recalibrated/retrained **only from labels already in the store
(the past)**, and a refreshed judge is swapped in **only at a round/predictable
boundary** — never using the label of the item currently being bet on. This keeps
`pi_t`, `lam_PP`, `lam_t` predictable, so the anytime-valid guarantee survives a
judge that improves mid-experiment. Locked by a test alongside the existing
`test_F7_predictability_*`.

## 7. Simulation backends (goal 1)

`sim/base.py` defines `RolloutSource` → yields paired `(Rollout_A, Rollout_B)`
plus a `gold(pair)` oracle where available.

- `sim/synthetic.py` — success-rate sim (relocated `make_world`); **runs on the
  laptop now**; gold oracle is the latent truth. This is the demo that shows
  "A > B" today.
- `sim/lerobot_env.py` — runs two pretrained checkpoints in a real lerobot gym env
  (e.g. `gym-pusht`), records camera frames; **offloaded** to SkyPilot/HF Jobs;
  heavy deps lazy. Gold here is a real human via the LabelStore.

In **synthetic mode the gold oracle auto-answers the LabelStore** (and the Notifier
still fires, to exercise that path), so the entire batch/online loop —
notify → label → triple-use update → decision — runs **unattended on the laptop**.
In real mode those same `answer()` calls come from a person via
`lerobot-compare label`. Same code path; only the answer source differs.

## 8. What runs where

| Component        | Laptop (now)              | Offloaded (later, no logic change)     |
|------------------|---------------------------|----------------------------------------|
| Sim              | `synthetic`               | `lerobot_env` (gym-pusht, 2 checkpoints) |
| Judge            | `StubVLM`                 | `VLMJudge` (Qwen2.5-VL)                 |
| Calibration      | isotonic (numpy)          | same                                   |
| Retrain          | dry-run (writes artifacts)| `dry_run=False` LoRA on SkyPilot/HF Jobs |
| Notify sink      | `StdoutSink`              | `StdoutSink` (+ Telegram/webhook later)|

## 9. CLI surface (`lerobot-compare`)

- `lerobot-compare compare A.txt B.txt [--alpha --alternative --json --charts]` —
  the existing flat-file A/B test (keep).
- `lerobot-compare loop --config run.yaml` — run the batch human-in-the-loop cycle;
  emits notifications, writes the run's `queue.jsonl`.
- `lerobot-compare label runs/<id>` — interactive yes/no over the pending batch;
  writes answers back to the LabelStore.

Config via **draccus** dataclasses in `config.py` (matches lerobot conventions):
`DoctorConfig{ sim, judge, auditor, alpha, alternative, train, notify, run_dir }`.

## 10. Documentation & bench (paper demoted)

- `docs/method.md`: ≤2 pages. What the tool does, the honest "assembled from known
  work" framing, references (PPI/PPI++, prediction-powered e-values, WSR betting,
  CUPED, controlled sensing), and the Neyman-backfire caveat. No novelty claims.
- `bench/validate.py`: optional regeneration of validity + efficiency plots (was
  `experiments.py`) — kept as evidence the guarantees hold, not as paper figures.

## 11. Public API (`lerobot_doctor/__init__.py`)

Curated, stable surface (heavy deps lazy):

```
compare_policies, compare_success, Report            # comparisons
Rollout, HumanGold, StubVLM, VLMJudge, RubricVLM,    # evaluators
  AutomaticChecker, InternalState
Recalibrate, CalibratedEvaluator                     # calibration
MaxEvidenceAuditor, UniformAuditor, NeymanAuditor    # auditing
RolloutSource, SyntheticSim                          # sim
LabelStore, Notifier, StdoutSink, Updater            # loop
DoctorLoop (batch), OnlineLoop                        # drivers
TrainingConfig, LlamaFactoryBackend, SkyPilotBackend # training
```

## 12. Testing — robust by construction

Testing is a first-class deliverable, not an afterthought. Every module ships with
tests in the same PR step that creates it (§14); a step is not "done" until its
tests are green. The suite is organized in five tiers under `tests/doctor/`.

**Conventions.** `pytest`; every stochastic test seeds its RNG and a
`test_*_determinism` asserts identical results across two runs with the same seed;
Monte-Carlo tiers are marked `@pytest.mark.slow` with a fast default (small `n`,
trials) and a thorough CI profile. Target ≥90% line coverage on `core/` and
`loop/` (measured, reported in CI). No network, no GPU, no weight downloads in the
default suite — the real `VLMJudge` / `lerobot_env` paths are tested via the stub
and via import-only/mock tests.

### Tier 1 — Statistical guarantees (the ones that must never regress)
Relocated F1–F7, reframed as guarantee tests (not paper evidence):
- **Validity (Type-I ≤ α):** Monte-Carlo under H0 across evaluator quality
  (incl. a useless coin-flip judge), across all three Auditors, at ≥2 alphas;
  asserts empirical false-positive rate ≤ α with margin.
- **Supermartingale under H0:** median final wealth ≈ 1, threshold-cross prob ≈ 0.
- **Payoff unbiasedness** despite a biased/miscalibrated evaluator.
- **Power / efficiency ladder:** v2 beats v1; each lever (adaptive bet, control
  variate, decisive-tilt auditing) helps; principled beats uniform at matched budget.
- **Neyman-backfire** locked in (estimation-optimal ≠ detection-optimal) so no
  future "improvement" silently regresses to it.
- **Graceful degradation:** `lam_PP → 0` for a useless judge, `→ 1` for a good one;
  a useless judge is still valid.

### Tier 2 — Predictability invariants (the subtle, validity-critical ones)
- `pi_t`, `lam_PP`, `lam_t` ignore the current item's gold label (extends
  existing `test_F7_predictability_*`).
- **Mid-loop judge refresh stays valid:** a loop that recalibrates/retrains from
  past-only labels and swaps the judge at round boundaries still has Type-I ≤ α
  (Monte-Carlo). This is the new headline invariant from §6.5.
- `core/` imports numpy only (no torch/lerobot leak) — import-guard test.

### Tier 3 — Component units
- `LabelStore`: enqueue/dedup (same `item_id` never twice), `answer()` round-trip,
  **crash-resume** (reopen a half-written store, pending set is correct, no label
  lost), malformed-line tolerance.
- `Notifier`: `notify()` returns without blocking even with an artificially slow
  sink (assert submit latency ≪ sink latency); worker drains in order; context-
  manager shutdown flushes all pending; exception in a sink doesn't kill the worker
  or the experiment.
- `Updater`: one ingested label provably reaches **all three** consumers
  (e-process gold, isotonic recalibration input, retrain pool); `refresh_judge()`
  returns a recalibrated evaluator and respects the retrain cadence.
- `calibrate`: isotonic map is monotone, clipped to [0,1], identity below
  `min_points`.
- `auditing`: each Auditor's `pi` within its declared floor/ceiling; predictable
  (depends only on `f_A,f_B`).
- `matching`: exact > covariate > unpaired rung assignment; uses no labels; caliper
  respected; rung counts/balance reported.
- `checks`: built-in success flags detected and ignored-by-default; `use_as`
  policy branches.
- `report`/`viz`: `to_dict()` round-trips; `summary()` renders v1 and v2; chart
  smoke test writes a PNG without error.

### Tier 4 — Integration (end-to-end on synthetic, runs on the laptop)
- Full **batch loop** on `SyntheticSim`: decides "A>B", reports human labels spent
  vs. label-everything (asserts a real saving), oracle auto-answers the store.
- Full **online driver** reaches the same decision with the same components.
- **CLI**: `compare` on fixture files exits 0/2 correctly and emits valid JSON;
  `loop` writes a `queue.jsonl`; `label` answers it and the store updates.
- **Real paths without weights:** `VLMJudge` and `lerobot_env` are exercised with
  mocked model/env objects (verify prompt/rubric assembly, frame handling, lazy
  import) — no downloads.

### Tier 5 — Packaging / PR-readiness
- `vendor_to_lerobot.py`: emitted tree rewrites `lerobot_doctor` → `lerobot.doctor`,
  has no stale `lerobot_doctor` strings, and imports cleanly when placed on a path
  as `lerobot/doctor/`.
- Public API: everything in `__init__.__all__` is importable with only the base
  (numpy/scipy) deps installed (lazy heavy deps don't break import).

### CI
GitHub Actions: lint (ruff) + fast suite on push; the `slow` Monte-Carlo tier and
coverage gate on PR. The fast suite must stay under a couple of minutes so it's run
constantly during development.

## 13. Migration map (old file → new home)

| Old | New |
|-----|-----|
| `engine.py` / `parsimony.py` | `core/{eprocess,payoff,auditing,compare}.py` (parsimony.py deleted) |
| `evaluators.py` | `evaluators/{base,stub,vlm,internal}.py` |
| `parsimony_robotics.py` | `evaluators/vlm.py` (RubricVLM) + `sim/lerobot_env.py` (loader) |
| `calibrate.py` | `evaluators/calibrate.py` |
| `training.py` | `loop/train.py` |
| `compare_policies.py` | folded into `loop/engine.py` + `core/compare.py` |
| `compare.py` | `core/compare.py` (v1 path) |
| `matching.py`, `report.py`, `checks.py`, `viz.py`, `cli.py` | same names under package; `cli.py` → `src/lerobot/scripts/doctor.py` |
| `experiments.py` | `bench/validate.py` |
| `paper.md`, `v2-efficiency.md` | `docs/method.md` |
| `test_*.py`, `test_parsimony*.py` | `tests/doctor/` |
| `*_demo.py`, `offline_compare_stub.py`, `robotics_qwen_lerobot.py` | `examples/` |

## 14. Sequencing (for the implementation plan)

1. Scaffold `src/lerobot_doctor/` package + `pyproject.toml` + `.gitignore`; move
   core (`engine.py`) into `core/` with the `auditing` rename; get F1–F7 green.
2. Relocate evaluators + calibrate + matching + report + checks + viz; green tests.
3. `sim/` interface + `synthetic` backend (relocate `make_world`).
4. `loop/`: `store` → `notify` (async + StdoutSink) → `update` (triple-use) →
   `train` (relocate) → `engine` (batch) → `online`. Add loop tests + invariant.
5. CLI (`compare`/`loop`/`label`) + draccus `config.py`.
6. `docs/method.md`, `bench/validate.py`, `examples/`, README rewrite.
7. `scripts/vendor_to_lerobot.py` + its test (the PR-readiness check).
8. `evaluators/vlm.py` real path + `sim/lerobot_env.py` (offloaded; lazy) — last,
   since they can't be exercised on the laptop.

## 15. Open / deferred

- Telegram/webhook sinks (interface ready; not built now).
- Real offloaded runs (SkyPilot/HF Jobs) validated by the user on a GPU box.
- Unpaired streaming comparison (matching already reports the rung honestly).
