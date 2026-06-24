# lerobot-compare Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate the two overlapping code families into one installable `lerobot_doctor` package whose full human-in-the-loop runs end-to-end on synthetic backends locally, with a robust five-tier test suite and a codemod that makes it a near-copy-paste PR into `huggingface/lerobot`.

**Architecture:** A domain-agnostic numpy-only `core/` (betting e-process + prediction-powered payoff + cost-aware auditing) under pluggable `evaluators/`, `sim/`, and a `loop/` of three shared components (durable `LabelStore`, async non-blocking `Notifier`, triple-use `Updater`) driven by a batch engine and a thin online driver. Synthetic backends run on the laptop today; real VLM/gym backends swap in offloaded with no logic change.

**Tech Stack:** Python ≥3.10, numpy, scipy (matching), draccus (config), pytest. Heavy deps (torch, transformers, lerobot) are optional extras, lazily imported.

## Global Constraints

- **Package dir:** `src/lerobot_doctor/`, structured 1:1 identical to the eventual `lerobot/doctor/`. Standalone-installable; no `lerobot` import required for the base package.
- **`core/` imports numpy only.** No torch / transformers / lerobot leak into `core/`. Enforced by a test.
- **Rename `parsimony` → `auditing` everywhere** (filenames, identifiers, docstrings, comments). `AcquisitionRule` → `Auditor`; `MaxEvidencePerCost` → `MaxEvidenceAuditor`; `UniformAudit` → `UniformAuditor`; `NeymanAcquisition` → `NeymanAuditor`. No string "parsimony" remains except in third-party references (the cited papers) inside `docs/method.md`.
- **Predictability invariant:** `pi_t`, `lam_PP`, `lam_t`, and any judge refresh use **past labels only**; a refreshed judge swaps in only at a round/predictable boundary. Validity (Type-I ≤ α at every stopping time) must hold.
- **Heavy deps lazy:** importing `lerobot_doctor` with only numpy/scipy installed must succeed; `VLMJudge`/`lerobot_env` import their deps inside methods.
- **Determinism:** every stochastic test seeds its RNG; Monte-Carlo tests marked `@pytest.mark.slow`.
- **Default test suite:** no network, no GPU, no weight downloads.
- **CLI:** `lerobot-compare` (not an extension of `lerobot.scripts.eval`).
- **Zero `parsimony`:** the string `parsimony` (any case) must not appear in any
  shipped artifact — `src/`, `tests/`, `bench/`, `examples/`, `scripts/`,
  `pyproject.toml`, `README.md`, `docs/method.md` — nor in any filename. (The
  historical `docs/superpowers/` spec/plan keep it as record; they are not shipped.)
  Verified by a grep gate in Task 20. The old `parsimony*.py` files are deleted or
  relocated+renamed; the brand is replaced by `auditing` (the method) / `lerobot-compare`
  (the package).
- **Apache-2.0 license header on EVERY `.py`** (source, tests, `__init__.py`, scripts,
  bench, examples) — lerobot's file format, attributed to the actual author (NOT
  HuggingFace, NOT any AI tool) — verbatim, as the first lines:

  ```python
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
  ```
- **lerobot test format:** tests live in `tests/doctor/` mirroring
  `src/lerobot_doctor/`, files named `test_<module>.py`, **function-based** tests
  named `test_*` each with a **one-line docstring** stating the behavior under test;
  use `pytest.mark.parametrize` where it removes duplication; optional-dependency
  failures raise `ImportError` whose message names the extra (e.g.
  `pip install 'lerobot-compare[vlm]'`). `tests/__init__.py`, `tests/doctor/__init__.py`,
  and `tests/doctor/conftest.py` all carry the license header. (The code blocks shown
  in later tasks omit the header and docstrings for brevity — add them per this rule.)
- **No `v1`/`v2` labels:** the two comparison paths get meaningful names, not version
  numbers. The human-labels-every-rollout path is **`direct`**; the cheap-judge +
  cost-aware-audit path is **`audited`**. This applies to `Report.mode` values
  (`"direct"` / `"audited"`), docstrings, comments, test names, and demo filenames.
  During relocation, rewrite existing `"v1"`/`"v2"` strings accordingly. Verified by a
  grep gate in Task 20. (The legacy source files name things v1/v2 — rename on the way in.)
- **Frequent commits:** one commit per task minimum; TDD (failing test first).

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/lerobot_doctor/__init__.py` | Curated public API; lazy heavy deps |
| `src/lerobot_doctor/core/eprocess.py` | `BettingEProcess`, `Obs` |
| `src/lerobot_doctor/core/payoff.py` | `Comparison` (PPI++ payoff) |
| `src/lerobot_doctor/core/auditing.py` | `Auditor` protocol + 3 auditors |
| `src/lerobot_doctor/core/compare.py` | `compare()` engine, `Result`, `make_world` |
| `src/lerobot_doctor/evaluators/base.py` | `Rollout`, `Evaluator`, `HumanGold` |
| `src/lerobot_doctor/evaluators/stub.py` | `StubVLM` |
| `src/lerobot_doctor/evaluators/internal.py` | `AutomaticChecker`, `InternalState` |
| `src/lerobot_doctor/evaluators/calibrate.py` | `Recalibrate`, `CalibratedEvaluator`, `LabelRecord` |
| `src/lerobot_doctor/evaluators/vlm.py` | `VLMJudge`, `RubricVLM` (lazy torch) |
| `src/lerobot_doctor/sim/base.py` | `RolloutSource` protocol, `Pair` |
| `src/lerobot_doctor/sim/synthetic.py` | `SyntheticSim` (laptop) |
| `src/lerobot_doctor/sim/lerobot_env.py` | gym + 2 checkpoints loader (offloaded, lazy) |
| `src/lerobot_doctor/loop/store.py` | `LabelStore` |
| `src/lerobot_doctor/loop/notify.py` | `Notification`, `Sink`, `StdoutSink`, `Notifier` |
| `src/lerobot_doctor/loop/update.py` | `Updater` (triple-use) |
| `src/lerobot_doctor/loop/train.py` | `TrainingConfig`, `LlamaFactoryBackend`, `SkyPilotBackend` |
| `src/lerobot_doctor/loop/engine.py` | `DoctorLoop` (batch), `compare_policies` |
| `src/lerobot_doctor/loop/online.py` | `OnlineLoop` |
| `src/lerobot_doctor/matching.py` | `match`, `Pairing`, `Trial` |
| `src/lerobot_doctor/report.py` | `Report` |
| `src/lerobot_doctor/checks.py` | success-flag preflight |
| `src/lerobot_doctor/viz.py` | charts (optional matplotlib) |
| `src/lerobot_doctor/config.py` | draccus config dataclasses |
| `src/lerobot_doctor/cli.py` | `lerobot-compare` CLI (codemod → `lerobot/scripts/doctor.py`) |
| `src/lerobot_doctor/loop/factory.py` | `build_loop(cfg)`, `apply_answers` |
| `scripts/vendor_to_lerobot.py` | codemod to `lerobot.doctor` |
| `tests/doctor/...` | five-tier suite |

---

## Phase 0 — Scaffold

### Task 1: Package skeleton, packaging, cleanup

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/lerobot_doctor/__init__.py`, `tests/doctor/__init__.py`, `tests/doctor/conftest.py`
- Delete (tracked/untracked cruft): `parsimony.py`, `*.pyc`, `parsimony.zip`, `CACHEDIR.TAG`, `nodeids`, `lastfailed`, `dataset_info.json`, `judge_data.jsonl`, `train.yaml`, `sky_task.yaml`, `fig*.png`, `sample_report.png`, `mnt/`

**Interfaces:**
- Produces: an installable package `lerobot_doctor` (empty API for now); `pytest` discovers `tests/doctor/`.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "lerobot-compare"
version = "0.3.0"
description = "Label-efficient, anytime-valid comparison of robot/agent policies for lerobot"
requires-python = ">=3.10"
dependencies = ["numpy>=1.24", "scipy>=1.10"]   # scipy: intervals (beta), matching (linear_sum_assignment)

[project.optional-dependencies]
viz = ["matplotlib>=3.7"]
vlm = ["torch>=2.2", "transformers>=4.49", "qwen-vl-utils", "accelerate", "pillow"]
sim = ["lerobot"]
dev = ["pytest>=8", "pytest-cov", "ruff"]

[project.scripts]
lerobot-compare = "lerobot_doctor.cli:main"   # codemod rewrites to lerobot.scripts.doctor:main for the PR

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow: Monte-Carlo / long-running statistical tests"]

[tool.ruff]
line-length = 100
```

- [ ] **Step 2: Write `.gitignore`**

```
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.ruff_cache/
out/
runs/
bench/figures/
*.png
!docs/**/*.png
```

- [ ] **Step 3: Write empty `src/lerobot_doctor/__init__.py`**

```python
"""lerobot-compare: label-efficient, anytime-valid policy comparison for lerobot."""
__version__ = "0.3.0"
```

- [ ] **Step 4: Write `tests/__init__.py`, `tests/doctor/__init__.py` (both empty but with the license header) and `tests/doctor/conftest.py`** (mirrors lerobot's `tests/` package layout)

```python
# tests/doctor/conftest.py  (license header omitted here — add it per Global Constraints)
import numpy as np
import pytest


@pytest.fixture
def rng():
    """Deterministic default RNG for tests."""
    return np.random.default_rng(0)
```

- [ ] **Step 5: Delete cruft**

```bash
git rm -q --ignore-unmatch parsimony.py parsimony.zip CACHEDIR.TAG nodeids lastfailed \
  dataset_info.json judge_data.jsonl train.yaml sky_task.yaml sample_report.png
git rm -q --ignore-unmatch 'fig*.png' '*.pyc'
rm -rf mnt out
```

- [ ] **Step 6: Install and verify import**

Run: `pip install -e ".[dev]"`
Then: `python -c "import lerobot_doctor; print(lerobot_doctor.__version__)"`
Expected: prints `0.3.0`

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "chore: scaffold lerobot_doctor package, drop cruft"
```

---

## Phase 1 — Core (relocate + rename `parsimony`→`auditing`)

### Task 2: Split `engine.py` into `core/`, apply renames, port guarantee tests

**Files:**
- Create: `src/lerobot_doctor/core/__init__.py`, `core/eprocess.py`, `core/payoff.py`, `core/auditing.py`, `core/compare.py`
- Source: existing `engine.py` (379 lines) — split by class, do not rewrite behavior
- Test: `tests/doctor/test_core_guarantees.py` (ported from `test_parsimony.py` + `test_audited.py`)
- Delete: `engine.py` after the split

**Interfaces:**
- Produces:
  - `eprocess.py`: `BettingEProcess(alpha, adaptive, lam_fixed, lam_max, warmup)` with `.update(payoff)`, `.wealth`, `.decided`; `Obs(pred_A, pred_B, audited, pi, gold_A, gold_B)`
  - `payoff.py`: `Comparison(pi_min, control_variate, warmup)` with `.payoff(Obs)->float`, `.observe(Obs)`, `._lam_pp()`
  - `auditing.py`: `Auditor` protocol `audit_prob(pred_A, pred_B, gold_cost)->float`; `MaxEvidenceAuditor`, `UniformAuditor`, `NeymanAuditor` (all carry `.pi_min`)
  - `compare.py`: `compare(items, evaluator, gold, predict, rule=None, alpha=0.10, adaptive=True, control_variate=True, rng=None)->Result`; `Result(decided, direction, wealth, n_seen, n_audited, cost_spent)`; `make_world(pA, pB, rng, rho_acc=0.75, N=1500)`; `GoldSource(cost, label_fn, name)`

- [ ] **Step 1: Create the split with renames.** Move `BettingEProcess` + `Obs` → `core/eprocess.py`; `Comparison` → `core/payoff.py` (imports `Obs` from `.eprocess`); `Auditor`/auditors → `core/auditing.py`; `compare`+`Result`+`make_world`+`GoldSource`+`CheapVLM` → `core/compare.py` (imports from `.eprocess`, `.payoff`, `.auditing`). Apply the Global Constraints renames verbatim. `core/__init__.py` re-exports all public names. Then `git rm engine.py`.

- [ ] **Step 2: Port the guarantee tests.** Copy `test_parsimony.py` (F1–F7) and the v2 cases into `tests/doctor/test_core_guarantees.py`, updating imports to `from lerobot_doctor.core import ...` and the renamed class names. Mark the Monte-Carlo Type-I/efficiency tests `@pytest.mark.slow`.

- [ ] **Step 3: Run the ported suite to verify it fails first only where renamed**

Run: `pytest tests/doctor/test_core_guarantees.py -q`
Expected at first: import/name errors until Step 1's renames are consistent; iterate Step 1 until collected.

- [ ] **Step 4: Run guarantee tests (fast + slow)**

Run: `pytest tests/doctor/test_core_guarantees.py -q -m "slow or not slow"`
Expected: PASS — validity (Type-I ≤ α incl. useless evaluator), supermartingale under H0, efficiency ladder (v2 beats v1; principled beats uniform at matched budget), Neyman backfire, graceful degradation.

- [ ] **Step 5: Add the `core`-purity test**

```python
# tests/doctor/test_core_purity.py
import importlib, sys


def test_core_imports_numpy_only():
    for m in ("eprocess", "payoff", "auditing", "compare"):
        importlib.import_module(f"lerobot_doctor.core.{m}")
    banned = {"torch", "transformers", "lerobot", "qwen_vl_utils"}
    assert banned.isdisjoint(sys.modules), f"core pulled in {banned & set(sys.modules)}"
```

Run: `pytest tests/doctor/test_core_purity.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: split engine into core/, rename parsimony->auditing"
```

---

## Phase 2 — Evaluators & supporting modules (relocate)

### Task 3: Relocate evaluators + calibration

**Files:**
- Create: `src/lerobot_doctor/evaluators/__init__.py`, `evaluators/base.py`, `evaluators/stub.py`, `evaluators/internal.py`, `evaluators/calibrate.py`
- Source: existing `evaluators.py` (split `Rollout`/`Evaluator`/`HumanGold`→base, `StubVLM`→stub, `AutomaticChecker`/`InternalState`→internal) and `calibrate.py` (→ evaluators/calibrate.py)
- Test: `tests/doctor/test_evaluators.py`, `tests/doctor/test_calibrate.py`

**Interfaces:**
- Produces:
  - `base.py`: `Rollout(task, frames, key, covariates, meta)`; `Evaluator` protocol (`cost`, `score(Rollout)->float`); `HumanGold(label_fn, cost=60.0)` with `.label(ra, rb)->(int,int)`
  - `stub.py`: `StubVLM(rho=0.8, cost=1.0)`
  - `internal.py`: `AutomaticChecker(fn, cost)`, `InternalState(fn, cost)`
  - `calibrate.py`: `LabelRecord(f, y, frames, question, meta)`; `CalibratedEvaluator(base, cal)`; `Recalibrate(base, min_points=20).update(records)->Evaluator`; `EvaluatorUpdater` protocol

- [ ] **Step 1: Move the code** into the four files (imports updated to `from .base import ...`). `evaluators/__init__.py` re-exports all. `calibrate.py` imports `Evaluator, Rollout` from `.base`. Delete old `evaluators.py` and `calibrate.py`.

- [ ] **Step 2: Write calibration tests**

```python
# tests/doctor/test_calibrate.py
import numpy as np
from lerobot_doctor.evaluators import StubVLM, Recalibrate, CalibratedEvaluator, LabelRecord, Rollout


def test_isotonic_is_monotone_and_clipped():
    rng = np.random.default_rng(1)
    recs = []
    for _ in range(200):
        f = rng.random()
        y = int(rng.random() < f)  # f correlates with y
        recs.append(LabelRecord(f=f, y=y))
    cal = Recalibrate(StubVLM(), min_points=20).update(recs)
    assert isinstance(cal, CalibratedEvaluator)
    xs = np.linspace(0, 1, 50)
    ys = [cal.cal(x) for x in xs]
    assert all(0.0 <= v <= 1.0 for v in ys)
    assert all(ys[i] <= ys[i + 1] + 1e-9 for i in range(len(ys) - 1))


def test_recalibrate_returns_base_below_min_points():
    base = StubVLM()
    out = Recalibrate(base, min_points=20).update([LabelRecord(f=0.5, y=1)] * 5)
    assert out is base
```

- [ ] **Step 3: Write evaluator tests**

```python
# tests/doctor/test_evaluators.py
import numpy as np
from lerobot_doctor.evaluators import StubVLM, AutomaticChecker, InternalState, HumanGold, Rollout


def test_stubvlm_useless_when_rho_half():
    ev = StubVLM(rho=0.5)
    r = Rollout(meta={"p": 0.9})
    scores = [ev.score(r) for _ in range(200)]
    assert 0.0 <= min(scores) and max(scores) <= 1.0


def test_internal_and_automatic_clip():
    assert AutomaticChecker(fn=lambda r: 2.0).score(Rollout()) == 1.0
    assert InternalState(fn=lambda r: -1.0).score(Rollout()) == 0.0


def test_human_gold_label():
    g = HumanGold(lambda a, b: (a.meta["y"], b.meta["y"]))
    assert g.label(Rollout(meta={"y": 1}), Rollout(meta={"y": 0})) == (1, 0)
```

- [ ] **Step 4: Run**

Run: `pytest tests/doctor/test_evaluators.py tests/doctor/test_calibrate.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: relocate evaluators and calibration into package"
```

### Task 4: Relocate matching, report, checks, viz

**Files:**
- Create: `src/lerobot_doctor/matching.py`, `report.py`, `checks.py`, `viz.py` (moved verbatim from repo root; fix imports to package-relative)
- Test: `tests/doctor/test_matching.py`, `test_report.py`, `test_checks.py`, `test_viz.py` (port `test_checks.py`, `test_viz.py`; add matching/report)

**Interfaces:**
- Produces: `match(trials_A, trials_B, caliper=1.0, rng=None)->Pairing`; `Pairing(pairs, rung_counts, n_unmatched_A, n_unmatched_B, ...)`, `.n_pairs`; `Trial(success, key, covariates, meta)`; `Report(...)` with `.summary()`, `.to_dict()`, `.match_strength`; `scan_success_flags(columns, use_as)`, `preflight_rollouts(rollouts, use_as)`, `emit(diags)`; `viz.dashboard(report, path)`

- [ ] **Step 1: Move the four files**, fixing imports (`report.py` stays dependency-free; `viz.py` imports matplotlib lazily inside `dashboard`). Delete originals.

- [ ] **Step 2: Port `test_checks.py` and `test_viz.py`** into `tests/doctor/` with updated imports.

- [ ] **Step 3: Add a matching test**

```python
# tests/doctor/test_matching.py
import numpy as np
from lerobot_doctor.matching import match, Trial


def test_exact_key_match_preferred():
    A = [Trial(1, key=k) for k in range(5)]
    B = [Trial(0, key=k) for k in range(5)]
    p = match(A, B, rng=np.random.default_rng(0))
    assert p.rung_counts["exact"] == 5
    assert p.n_pairs == 5


def test_unpaired_fallback_still_pairs():
    A = [Trial(1) for _ in range(3)]
    B = [Trial(0) for _ in range(3)]
    p = match(A, B, rng=np.random.default_rng(0))
    assert p.rung_counts["unpaired"] == 3
```

- [ ] **Step 4: Add a report test**

```python
# tests/doctor/test_report.py
from lerobot_doctor.report import Report


def _mk(**kw):
    base = dict(alpha=0.1, alternative="A>B", decided=True, direction="A>B",
               n_to_decision=10, n_pairs=10, n_A=10, n_B=10, successes_A=8,
               successes_B=4, rate_A=0.8, rate_B=0.4, effect=0.4, e_value=12.0,
               p_value_equiv=1/12.0)
    base.update(kw)
    return Report(**base)


def test_summary_and_roundtrip():
    r = _mk()
    assert "DECISION: A>B" in r.summary()
    assert r.to_dict()["e_value"] == 12.0
```

- [ ] **Step 5: Run**

Run: `pytest tests/doctor/test_matching.py tests/doctor/test_report.py tests/doctor/test_checks.py tests/doctor/test_viz.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: relocate matching, report, checks, viz into package"
```

---

## Phase 3 — Simulation backends

### Task 5: `sim/base.py` + `sim/synthetic.py`

**Files:**
- Create: `src/lerobot_doctor/sim/__init__.py`, `sim/base.py`, `sim/synthetic.py`
- Test: `tests/doctor/test_sim_synthetic.py`

**Interfaces:**
- Produces:
  - `base.py`: `Pair(task, A, B, key=None)`; `RolloutSource` protocol `pairs()->Iterator[Pair]` and `gold(pair)->Optional[Tuple[int,int]]`
  - `synthetic.py`: `SyntheticSim(pA=0.75, pB=0.55, n=1500, task="put cube in bin", rho=0.8, rng=...)` implementing `RolloutSource`; latent truth in `meta["y"]`; `gold(pair)` returns `(yA, yB)`

- [ ] **Step 1: Write `sim/base.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Protocol, Tuple
from ..evaluators.base import Rollout


@dataclass
class Pair:
    task: str
    A: Rollout
    B: Rollout
    key: Any = None


class RolloutSource(Protocol):
    def pairs(self) -> Iterator[Pair]: ...
    def gold(self, pair: Pair) -> Optional[Tuple[int, int]]: ...
```

- [ ] **Step 2: Write `sim/synthetic.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterator, Optional, Tuple
import numpy as np
from .base import Pair
from ..evaluators.base import Rollout


@dataclass
class SyntheticSim:
    """Paired success-rate sim. Same task instance (shared `base` draw) is run by
    A and B, so pairs are exact-matched. `gold` is the latent truth oracle."""
    pA: float = 0.75
    pB: float = 0.55
    n: int = 1500
    task: str = "put cube in bin"
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def pairs(self) -> Iterator[Pair]:
        for k in range(self.n):
            base = self.rng.random()
            a = Rollout(self.task, key=k, meta={"p": self.pA, "y": int(base < self.pA)})
            b = Rollout(self.task, key=k, meta={"p": self.pB, "y": int(base < self.pB)})
            yield Pair(self.task, a, b, key=k)

    def gold(self, pair: Pair) -> Optional[Tuple[int, int]]:
        return int(pair.A.meta["y"]), int(pair.B.meta["y"])
```

- [ ] **Step 3: Write the test**

```python
# tests/doctor/test_sim_synthetic.py
import numpy as np
from lerobot_doctor.sim import SyntheticSim, Pair


def test_pairs_are_exact_matched_and_gold_matches_truth():
    sim = SyntheticSim(pA=0.8, pB=0.4, n=50, rng=np.random.default_rng(0))
    ps = list(sim.pairs())
    assert len(ps) == 50
    assert all(isinstance(p, Pair) and p.A.key == p.B.key for p in ps)
    for p in ps:
        assert sim.gold(p) == (p.A.meta["y"], p.B.meta["y"])


def test_better_policy_has_higher_empirical_rate():
    sim = SyntheticSim(pA=0.8, pB=0.4, n=4000, rng=np.random.default_rng(1))
    ps = list(sim.pairs())
    ra = np.mean([p.A.meta["y"] for p in ps])
    rb = np.mean([p.B.meta["y"] for p in ps])
    assert ra > rb
```

- [ ] **Step 4: Run**

Run: `pytest tests/doctor/test_sim_synthetic.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: sim RolloutSource interface + synthetic backend"
```

---

## Phase 4 — The loop

### Task 6: `loop/store.py` — durable LabelStore

**Files:**
- Create: `src/lerobot_doctor/loop/__init__.py`, `loop/store.py`
- Test: `tests/doctor/test_store.py`

**Interfaces:**
- Produces: `LabelItem(item_id, f_A, f_B, question="", frames_ref=None, status="pending", y_A=None, y_B=None, ts=0.0)`; `LabelStore(path)` with `.enqueue(item)->bool` (False if dup), `.answer(item_id, y_A, y_B)`, `.pending()->List[LabelItem]`, `.answered()->List[LabelItem]`

- [ ] **Step 1: Write the failing test**

```python
# tests/doctor/test_store.py
from lerobot_doctor.loop.store import LabelStore, LabelItem


def _item(i):
    return LabelItem(item_id=str(i), f_A=0.6, f_B=0.4, question="ok?")


def test_dedup_enqueue(tmp_path):
    s = LabelStore(str(tmp_path / "q.jsonl"))
    assert s.enqueue(_item(1)) is True
    assert s.enqueue(_item(1)) is False
    assert len(s.pending()) == 1


def test_answer_round_trip(tmp_path):
    s = LabelStore(str(tmp_path / "q.jsonl"))
    s.enqueue(_item(1))
    s.answer("1", 1, 0)
    assert s.pending() == []
    a = s.answered()
    assert len(a) == 1 and (a[0].y_A, a[0].y_B) == (1, 0)


def test_crash_resume_and_torn_line(tmp_path):
    p = str(tmp_path / "q.jsonl")
    s = LabelStore(p)
    s.enqueue(_item(1)); s.enqueue(_item(2)); s.answer("1", 1, 1)
    with open(p, "a") as f:
        f.write('{"item_id": "3", "f_A": 0.5')  # torn last line, no newline
    s2 = LabelStore(p)  # reopen
    assert {it.item_id for it in s2.answered()} == {"1"}
    assert {it.item_id for it in s2.pending()} == {"2"}
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/doctor/test_store.py -q`
Expected: FAIL (module not found)

- [ ] **Step 3: Write `loop/store.py`**

```python
from __future__ import annotations
import json
import os
import threading
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class LabelItem:
    item_id: str
    f_A: float
    f_B: float
    question: str = ""
    frames_ref: Any = None
    status: str = "pending"          # pending | answered
    y_A: Optional[int] = None
    y_B: Optional[int] = None
    ts: float = 0.0


class LabelStore:
    """Append-only JSONL queue; last line per item_id wins on reload."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self._items: Dict[str, LabelItem] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue                      # tolerate a torn final line
                self._items[d["item_id"]] = LabelItem(**d)

    def _append(self, item: LabelItem) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(item)) + "\n")

    def enqueue(self, item: LabelItem) -> bool:
        with self._lock:
            if item.item_id in self._items:
                return False
            self._items[item.item_id] = item
            self._append(item)
            return True

    def answer(self, item_id: str, y_A: int, y_B: int) -> None:
        with self._lock:
            it = self._items[item_id]
            it.status = "answered"
            it.y_A, it.y_B = int(y_A), int(y_B)
            self._append(it)

    def pending(self) -> List[LabelItem]:
        return [it for it in self._items.values() if it.status == "pending"]

    def answered(self) -> List[LabelItem]:
        return [it for it in self._items.values() if it.status == "answered"]
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/doctor/test_store.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: durable LabelStore with dedup and crash-resume"
```

### Task 7: `loop/notify.py` — async non-blocking Notifier

**Files:**
- Create: `src/lerobot_doctor/loop/notify.py`
- Test: `tests/doctor/test_notify.py`

**Interfaces:**
- Produces: `Notification(event, count=0, queue_path="", payload={}, ts=...)`; `Sink` protocol `emit(Notification)`; `StdoutSink(stream=None)`; `Notifier(sink=None)` with `.notify(n)` (non-blocking), `.flush()`, `.close()`, context-manager

- [ ] **Step 1: Write the failing test**

```python
# tests/doctor/test_notify.py
import time
from lerobot_doctor.loop.notify import Notifier, Notification, StdoutSink, Sink


class SlowSink:
    def __init__(self):
        self.seen = []

    def emit(self, n):
        time.sleep(0.05)
        self.seen.append(n)


class BadSink:
    def __init__(self):
        self.after = 0

    def emit(self, n):
        if n.event == "boom":
            raise RuntimeError("sink failure")
        self.after += 1


def test_notify_is_non_blocking():
    sink = SlowSink()
    with Notifier(sink) as nf:
        t0 = time.perf_counter()
        for i in range(5):
            nf.notify(Notification("labels_ready", count=i))
        submit = time.perf_counter() - t0
        assert submit < 0.02            # returns immediately, far less than 5*0.05
    assert len(sink.seen) == 5          # __exit__ flushed all


def test_bad_sink_does_not_kill_worker():
    sink = BadSink()
    with Notifier(sink) as nf:
        nf.notify(Notification("boom"))
        nf.notify(Notification("ok"))
    assert sink.after == 1              # survived the exception, processed the next


def test_stdout_sink_writes_banner(capsys):
    StdoutSink().emit(Notification("labels_ready", count=3, queue_path="runs/x/q.jsonl"))
    out = capsys.readouterr().out
    assert "labels_ready" in out and "3" in out and "runs/x/q.jsonl" in out
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/doctor/test_notify.py -q`
Expected: FAIL (module not found)

- [ ] **Step 3: Write `loop/notify.py`**

```python
from __future__ import annotations
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol


@dataclass
class Notification:
    event: str
    count: int = 0
    queue_path: str = ""
    payload: dict = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


class Sink(Protocol):
    def emit(self, n: Notification) -> None: ...


@dataclass
class StdoutSink:
    stream: Any = None

    def emit(self, n: Notification) -> None:
        out = self.stream or sys.stdout
        tail = f" -> {n.queue_path}" if n.queue_path else ""
        out.write(f">> [doctor] {n.event}: {n.count} item(s) awaiting label{tail}\n")
        out.flush()


class Notifier:
    """Submit-and-forget. notify() enqueues and returns; a daemon worker drains to
    the sink, so a slow/failing sink never stalls or breaks the experiment."""

    def __init__(self, sink: Optional[Sink] = None):
        self.sink: Sink = sink or StdoutSink()
        self._q: "queue.Queue[Optional[Notification]]" = queue.Queue()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def _run(self) -> None:
        while True:
            n = self._q.get()
            try:
                if n is None:
                    return
                self.sink.emit(n)
            except Exception:
                pass                      # never let a bad sink kill the worker
            finally:
                self._q.task_done()

    def notify(self, n: Notification) -> None:
        self._q.put(n)

    def flush(self) -> None:
        self._q.join()

    def close(self) -> None:
        self._q.put(None)
        self._worker.join(timeout=5)

    def __enter__(self) -> "Notifier":
        return self

    def __exit__(self, *exc: object) -> None:
        self.flush()
        self.close()
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/doctor/test_notify.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: async non-blocking Notifier with pluggable sinks"
```

### Task 8: `loop/update.py` — triple-use Updater

**Files:**
- Create: `src/lerobot_doctor/loop/update.py`
- Test: `tests/doctor/test_update.py`

**Interfaces:**
- Consumes: `LabelRecord`, `Recalibrate`, `CalibratedEvaluator` from `evaluators.calibrate`; `Evaluator` from `evaluators.base`
- Produces: `Updater(base_evaluator, retrain_cadence=0, min_recal=20)` with `.ingest(f, y, frames=None, question="")`, `.refresh_judge(retrain_backend=None)->Evaluator`, `.retrain_pool->List[LabelRecord]`

- [ ] **Step 1: Write the failing test**

```python
# tests/doctor/test_update.py
import numpy as np
from lerobot_doctor.evaluators import StubVLM, CalibratedEvaluator
from lerobot_doctor.loop.update import Updater


class RecordingBackend:
    def __init__(self):
        self.calls = []

    def update(self, records):
        self.calls.append(len(records))


def test_ingest_populates_retrain_pool():
    up = Updater(StubVLM(), min_recal=10)
    for i in range(12):
        up.ingest(f=0.6, y=1, frames=[f"{i}.png"], question="ok?")
    assert len(up.retrain_pool) == 12
    assert up.retrain_pool[0].frames == ["0.png"]


def test_refresh_recalibrates_after_min_points():
    rng = np.random.default_rng(0)
    up = Updater(StubVLM(), min_recal=20)
    for _ in range(60):
        f = rng.random()
        up.ingest(f=f, y=int(rng.random() < f))
    judge = up.refresh_judge()
    assert isinstance(judge, CalibratedEvaluator)


def test_retrain_fires_on_cadence_with_past_only():
    up = Updater(StubVLM(), retrain_cadence=10, min_recal=5)
    be = RecordingBackend()
    for i in range(9):
        up.ingest(f=0.5, y=1)
    up.refresh_judge(be)
    assert be.calls == []                 # below cadence
    up.ingest(f=0.5, y=1)                  # 10th
    up.refresh_judge(be)
    assert be.calls == [10]               # fired once, on the whole past pool
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/doctor/test_update.py -q`
Expected: FAIL (module not found)

- [ ] **Step 3: Write `loop/update.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Optional
from ..evaluators.base import Evaluator
from ..evaluators.calibrate import LabelRecord, Recalibrate


@dataclass
class Updater:
    """Fan each human label out to every consumer that can use it. The e-process
    consumes the label as gold in the caller (the audit). Here we cover the other
    two uses: instant isotonic recalibration, and the LoRA retrain pool. Both use
    PAST labels only, preserving the predictability invariant."""
    base_evaluator: Evaluator
    retrain_cadence: int = 0              # 0 => never retrain in-loop (recalibrate only)
    min_recal: int = 20
    records: List[LabelRecord] = field(default_factory=list)
    _since_retrain: int = 0

    def ingest(self, f: float, y: int, frames: Any = None, question: str = "") -> None:
        self.records.append(LabelRecord(f=f, y=y, frames=frames, question=question))
        self._since_retrain += 1

    def refresh_judge(self, retrain_backend: Optional[Any] = None) -> Evaluator:
        judge = Recalibrate(self.base_evaluator, min_points=self.min_recal).update(self.records)
        if (retrain_backend is not None and self.retrain_cadence
                and self._since_retrain >= self.retrain_cadence):
            retrain_backend.update(list(self.records))   # whole past pool
            self._since_retrain = 0
        return judge

    @property
    def retrain_pool(self) -> List[LabelRecord]:
        return self.records
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/doctor/test_update.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: triple-use Updater (gold/recalibrate/retrain pool)"
```

### Task 9: `loop/train.py` — relocate training handoff

**Files:**
- Create: `src/lerobot_doctor/loop/train.py` (moved from `training.py`; import `LabelRecord` from `..evaluators.calibrate`)
- Test: `tests/doctor/test_train.py`
- Delete: `training.py`

**Interfaces:**
- Produces: `TrainingConfig(...)`; `write_dataset(records, path)->int`; `LlamaFactoryBackend(config, dry_run=True).update(records)->TrainResult`; `SkyPilotBackend(config, accelerator, infra, dry_run=True).update(records)->TrainResult`; `TrainResult(dataset_path, config_path, command, adapter_dir, ran, returncode)`

- [ ] **Step 1: Move the file**, updating the import line to `from ..evaluators.calibrate import LabelRecord, EvaluatorUpdater`. Behavior unchanged (dry-run writes dataset+config+command).

- [ ] **Step 2: Write the test**

```python
# tests/doctor/test_train.py
import json
from lerobot_doctor.loop.train import LlamaFactoryBackend, SkyPilotBackend, TrainingConfig, write_dataset
from lerobot_doctor.evaluators import LabelRecord


def _recs(n):
    return [LabelRecord(f=0.6, y=i % 2, frames=[f"{i}.png"], question="cube in bin?")
            for i in range(n)]


def test_write_dataset_is_sharegpt_jsonl(tmp_path):
    p = tmp_path / "judge.jsonl"
    n = write_dataset(_recs(3), str(p))
    assert n == 3
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    assert rows[0]["messages"][1]["content"] in ("yes", "no")
    assert rows[0]["images"] == ["0.png"]


def test_llamafactory_dry_run_writes_artifacts(tmp_path):
    cfg = TrainingConfig(workdir=str(tmp_path / "t"), output_dir=str(tmp_path / "o"))
    res = LlamaFactoryBackend(cfg, dry_run=True).update(_recs(4))
    assert res.ran is False
    assert res.command[0] == "llamafactory-cli"
    assert (tmp_path / "t" / "train.yaml").exists()


def test_skypilot_dry_run_wraps_command(tmp_path):
    cfg = TrainingConfig(workdir=str(tmp_path / "t"), output_dir=str(tmp_path / "o"))
    res = SkyPilotBackend(cfg, accelerator="A100:1", dry_run=True).update(_recs(2))
    assert res.command[0] == "sky"
```

- [ ] **Step 3: Run**

Run: `pytest tests/doctor/test_train.py -q`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "refactor: relocate training handoff to loop/train"
```

### Task 10: `loop/engine.py` — port `compare_policies` (stateless v2)

**Files:**
- Create: `src/lerobot_doctor/loop/engine.py` (port `compare_policies.py`; update imports to `..core`, `..evaluators.base`, `..matching`, `..report`; `MaxEvidencePerCost`→`MaxEvidenceAuditor`)
- Test: `tests/doctor/test_compare_policies.py` (port from `test_audited.py`)
- Delete: `compare_policies.py`

**Interfaces:**
- Produces: `compare_policies(rollouts_A, rollouts_B, evaluator, human_gold, *, alpha=0.10, alternative="two-sided", acquisition=None, control_variate=True, caliper=1.0, rng=None)->Report`

- [ ] **Step 1: Move + rewire imports.** Keep the function body identical except class/import renames.

- [ ] **Step 2: Port the v2 test** into `tests/doctor/test_compare_policies.py` with updated imports; assert it decides "A>B" on the 0.75-vs-0.55 stub world and that `n_audited < n_seen`.

- [ ] **Step 3: Run**

Run: `pytest tests/doctor/test_compare_policies.py -q`
Expected: PASS (decides A>B; reports gold saved)

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "refactor: relocate stateless compare_policies into loop/engine"
```

### Task 11: `loop/engine.py` — `DoctorLoop` (stateful human-in-the-loop, batch)

**Files:**
- Modify: `src/lerobot_doctor/loop/engine.py` (add `DoctorLoop`)
- Test: `tests/doctor/test_doctor_loop.py`

**Interfaces:**
- Consumes: `RolloutSource`/`Pair` (sim), `Auditor` (core.auditing), `BettingEProcess`/`Comparison`/`Obs` (core), `LabelStore`/`LabelItem`, `Notifier`/`Notification`, `Updater`, `Report`
- Produces: `DoctorLoop(source, evaluator, auditor, store, notifier, updater, gold=None, train_backend=None, alpha=0.10, alternative="A>B", batch_size=64, rng=None)` with `.run(max_new_labels=10**9)->Report`. `gold` is a callable `gold(Pair)->(int,int)` or None. When `gold` is None and an audited item has no stored answer, the loop enqueues it, notifies, and returns a Report with `decided=False` and `status="needs_labels"`; re-running after the store is answered resumes deterministically (same RNG ⇒ same audit decisions; stored answers short-circuit).

- [ ] **Step 1: Write the failing test**

```python
# tests/doctor/test_doctor_loop.py
import numpy as np
from lerobot_doctor.sim import SyntheticSim
from lerobot_doctor.evaluators import StubVLM
from lerobot_doctor.core.auditing import MaxEvidenceAuditor
from lerobot_doctor.loop.store import LabelStore
from lerobot_doctor.loop.notify import Notifier
from lerobot_doctor.loop.update import Updater
from lerobot_doctor.loop.engine import DoctorLoop


def _loop(tmp_path, sink_seen, gold_oracle, n=1500):
    sim = SyntheticSim(pA=0.75, pB=0.55, n=n, rng=np.random.default_rng(0))
    judge = StubVLM(rho=0.85, _rng=np.random.default_rng(2))
    store = LabelStore(str(tmp_path / "q.jsonl"))

    class CollectSink:
        def emit(self, note): sink_seen.append(note)

    nf = Notifier(CollectSink())
    up = Updater(judge, min_recal=20)
    return DoctorLoop(sim, judge, MaxEvidenceAuditor(), store, nf, up,
                      gold=gold_oracle, alpha=0.10, alternative="A>B",
                      batch_size=64, rng=np.random.default_rng(7)), nf


def test_auto_mode_decides_A_gt_B_and_saves_labels(tmp_path):
    sim_ref = SyntheticSim(pA=0.75, pB=0.55, n=1500)
    seen = []
    loop, nf = _loop(tmp_path, seen, gold_oracle=sim_ref.gold)
    rep = loop.run()
    nf.flush(); nf.close()
    assert rep.decided and rep.direction == "A>B"
    assert rep.n_audited < rep.n_seen          # cheap judge carried most pairs
    assert rep.gold_saved_frac > 0.3
    assert len(seen) >= 1                       # notifier path exercised


def test_needs_labels_when_no_oracle_then_resumes(tmp_path):
    seen = []
    loop, nf = _loop(tmp_path, seen, gold_oracle=None, n=300)
    rep = loop.run()
    nf.flush(); nf.close()
    assert rep.decided is False and rep.status == "needs_labels"
    pend = loop.store.pending()
    assert len(pend) >= 1                        # something was enqueued for a human
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/doctor/test_doctor_loop.py -q`
Expected: FAIL (`DoctorLoop` not defined)

- [ ] **Step 3: Add `DoctorLoop` to `loop/engine.py`**

```python
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple
import numpy as np
from ..core.eprocess import BettingEProcess, Obs
from ..core.payoff import Comparison
from ..core.auditing import Auditor, MaxEvidenceAuditor
from ..report import Report
from ..sim.base import Pair, RolloutSource
from ..evaluators.base import Evaluator
from .store import LabelStore, LabelItem
from .notify import Notifier, Notification
from .update import Updater


@dataclass
class DoctorLoop:
    source: RolloutSource
    evaluator: Evaluator
    auditor: Auditor
    store: LabelStore
    notifier: Notifier
    updater: Updater
    gold: Optional[Callable[[Pair], Tuple[int, int]]] = None
    train_backend: Any = None
    alpha: float = 0.10
    alternative: str = "A>B"               # "A>B" | "B>A" | "two-sided"
    batch_size: int = 64
    rng: Optional[np.random.Generator] = None
    status: str = "running"

    def run(self, max_new_labels: int = 10 ** 9) -> Report:
        rng = self.rng or np.random.default_rng(0)
        a_each = self.alpha / 2 if self.alternative == "two-sided" else self.alpha
        est = Comparison(pi_min=getattr(self.auditor, "pi_min", 0.05), control_variate=True)
        ep_AgtB = BettingEProcess(alpha=a_each, adaptive=True)
        ep_BgtA = BettingEProcess(alpha=a_each, adaptive=True)
        test_A = self.alternative in ("two-sided", "A>B")
        test_B = self.alternative in ("two-sided", "B>A")

        judge = self.evaluator
        answered = {it.item_id: it for it in self.store.answered()}
        decided, direction, n_to_decision = False, None, None
        n_seen = n_audited = new_labels = 0
        sa = sb = na = nb = 0

        for pair in self.source.pairs():
            n_seen += 1
            fA, fB = judge.score(pair.A), judge.score(pair.B)
            gold_cost = 60.0
            pi = self.auditor.audit_prob(fA, fB, gold_cost)
            audited = bool(rng.random() < pi)
            yA = yB = None
            if audited:
                iid = str(pair.key if pair.key is not None else n_seen)
                if iid in answered:
                    yA, yB = answered[iid].y_A, answered[iid].y_B
                elif self.gold is not None:
                    yA, yB = self.gold(pair)
                    self.store.enqueue(LabelItem(iid, fA, fB, status="answered",
                                                 y_A=int(yA), y_B=int(yB), ts=time.time()))
                else:
                    self.store.enqueue(LabelItem(iid, fA, fB, question=pair.task,
                                                 frames_ref=getattr(pair.A, "frames", None),
                                                 ts=time.time()))
                    self.notifier.notify(Notification(
                        "needs_labels", count=len(self.store.pending()),
                        queue_path=self.store.path))
                    self.status = "needs_labels"
                    return self._report(False, None, None, n_seen, n_audited,
                                        sa, sb, na, nb, est)
                n_audited += 1
                new_labels += 1
                sa += yA; sb += yB; na += 1; nb += 1
                self.updater.ingest(fA, yA, frames=getattr(pair.A, "frames", None),
                                    question=pair.task)
                self.updater.ingest(fB, yB, frames=getattr(pair.B, "frames", None),
                                    question=pair.task)

            o = Obs(fA, fB, audited, pi, yA, yB)
            g = est.payoff(o)
            est.observe(o)
            if test_A:
                ep_AgtB.update(g)
            if test_B:
                ep_BgtA.update(-g)

            if not decided:
                if test_A and ep_AgtB.decided:
                    decided, direction, n_to_decision = True, "A>B", n_seen
                elif test_B and ep_BgtA.decided:
                    decided, direction, n_to_decision = True, "B>A", n_seen
            if decided:
                self.status = "decided"
                return self._report(True, direction, n_to_decision, n_seen, n_audited,
                                    sa, sb, na, nb, est)

            # refresh judge ONLY at a batch boundary, from PAST labels (predictable)
            if audited and self.batch_size and (new_labels % self.batch_size == 0):
                judge = self.updater.refresh_judge(self.train_backend)
                self.notifier.notify(Notification("progress", count=new_labels,
                                                  queue_path=self.store.path))
            if new_labels >= max_new_labels:
                break

        self.status = "exhausted"
        return self._report(False, None, None, n_seen, n_audited, sa, sb, na, nb, est)

    def _report(self, decided, direction, n_to_decision, n_seen, n_audited,
                sa, sb, na, nb, est) -> Report:
        rate_A = sa / max(na, 1)
        rate_B = sb / max(nb, 1)
        rep = Report(
            alpha=self.alpha, alternative=self.alternative, decided=decided,
            direction=direction, n_to_decision=n_to_decision, n_pairs=n_seen,
            n_A=n_seen, n_B=n_seen, successes_A=sa, successes_B=sb,
            rate_A=rate_A, rate_B=rate_B, effect=rate_A - rate_B,
            e_value=1.0, p_value_equiv=1.0, mode="audited", n_audited=n_audited,
            n_seen=n_seen, audit_rate=n_audited / max(n_seen, 1),
            lam_pp=est._lam_pp(), gold_saved_frac=1.0 - n_audited / max(n_seen, 1))
        rep.status = self.status
        return rep
```

> Note: `Report` gains a `status: str = "running"` field (default) in Task 4's `report.py` — add it there as a trailing optional field so `_report` can set it. If not yet added, add it now: `status: str = "running"` after `gold_saved_frac`.

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/doctor/test_doctor_loop.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: DoctorLoop batch human-in-the-loop driver"
```

### Task 12: Validity invariant under mid-loop judge refresh

**Files:**
- Test: `tests/doctor/test_loop_validity.py`

**Interfaces:**
- Consumes: `DoctorLoop`, `SyntheticSim`, `StubVLM`, `MaxEvidenceAuditor`, `LabelStore`, `Notifier`, `Updater`

- [ ] **Step 1: Write the Monte-Carlo invariant test**

```python
# tests/doctor/test_loop_validity.py
import numpy as np
import pytest
from lerobot_doctor.sim import SyntheticSim
from lerobot_doctor.evaluators import StubVLM
from lerobot_doctor.core.auditing import MaxEvidenceAuditor
from lerobot_doctor.loop.store import LabelStore
from lerobot_doctor.loop.notify import Notifier, StdoutSink
from lerobot_doctor.loop.update import Updater
from lerobot_doctor.loop.engine import DoctorLoop


@pytest.mark.slow
def test_type_i_below_alpha_with_mid_loop_refresh(tmp_path):
    """Under H0 (pA == pB) the loop must reject at most alpha of the time, EVEN
    though the judge is recalibrated mid-stream from past labels."""
    alpha, trials, false_pos = 0.10, 200, 0
    for t in range(trials):
        sim = SyntheticSim(pA=0.6, pB=0.6, n=600, rng=np.random.default_rng(t))
        ref = SyntheticSim(pA=0.6, pB=0.6, n=600, rng=np.random.default_rng(t))
        judge = StubVLM(rho=0.85, _rng=np.random.default_rng(1000 + t))
        store = LabelStore(str(tmp_path / f"q{t}.jsonl"))
        with Notifier(StdoutSink(stream=open(tmp_path / "null.txt", "a"))) as nf:
            up = Updater(judge, min_recal=20)
            loop = DoctorLoop(sim, judge, MaxEvidenceAuditor(), store, nf, up,
                              gold=ref.gold, alpha=alpha, alternative="A>B",
                              batch_size=32, rng=np.random.default_rng(5000 + t))
            rep = loop.run()
        false_pos += int(rep.decided)
    assert false_pos / trials <= alpha + 0.04      # MC slack
```

- [ ] **Step 2: Run**

Run: `pytest tests/doctor/test_loop_validity.py -q -m slow`
Expected: PASS (empirical false-positive rate ≤ ~α)

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "test: validity holds under mid-loop judge refresh"
```

### Task 13: `loop/online.py` — thin online driver

**Files:**
- Create: `src/lerobot_doctor/loop/online.py`
- Test: `tests/doctor/test_online.py`

**Interfaces:**
- Produces: `OnlineLoop(...)` — same constructor as `DoctorLoop` but `batch_size` defaults to 1 (refresh judge after every label); `.run()->Report`

- [ ] **Step 1: Write the failing test**

```python
# tests/doctor/test_online.py
import numpy as np
from lerobot_doctor.sim import SyntheticSim
from lerobot_doctor.evaluators import StubVLM
from lerobot_doctor.core.auditing import MaxEvidenceAuditor
from lerobot_doctor.loop.store import LabelStore
from lerobot_doctor.loop.notify import Notifier, StdoutSink
from lerobot_doctor.loop.update import Updater
from lerobot_doctor.loop.online import OnlineLoop


def test_online_reaches_decision(tmp_path):
    sim = SyntheticSim(pA=0.78, pB=0.5, n=1500, rng=np.random.default_rng(0))
    ref = SyntheticSim(pA=0.78, pB=0.5, n=1500, rng=np.random.default_rng(0))
    judge = StubVLM(rho=0.85, _rng=np.random.default_rng(2))
    store = LabelStore(str(tmp_path / "q.jsonl"))
    with Notifier(StdoutSink(stream=open(tmp_path / "n.txt", "a"))) as nf:
        up = Updater(judge, min_recal=20)
        loop = OnlineLoop(sim, judge, MaxEvidenceAuditor(), store, nf, up,
                          gold=ref.gold, alpha=0.10, alternative="A>B",
                          rng=np.random.default_rng(7))
        rep = loop.run()
    assert rep.decided and rep.direction == "A>B"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/doctor/test_online.py -q`
Expected: FAIL (module not found)

- [ ] **Step 3: Write `loop/online.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
from .engine import DoctorLoop


@dataclass
class OnlineLoop(DoctorLoop):
    """Feed-one-at-a-time driver: identical components to DoctorLoop but refreshes
    the judge after every label (batch_size=1). With a real human and gold=None it
    enqueues + notifies + returns 'needs_labels' on the first unanswered audit, then
    resumes deterministically on the next run() once the store is answered."""
    batch_size: int = 1
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/doctor/test_online.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: thin OnlineLoop driver over the batch engine"
```

---

## Phase 5 — v1 path, config, CLI

### Task 14: Relocate v1 `compare_success` + `intervals` (package level)

> **Refinement of spec §13:** `compare_success` and `intervals` pull in `matching`
> (scipy) and `report`, so they cannot live in numpy-only `core/`. They become
> package-level `src/lerobot_doctor/compare.py` and `intervals.py`. `core/compare.py`
> keeps only the generic `compare()` engine.

**Files:**
- Create: `src/lerobot_doctor/compare.py` (from `compare.py`; import `BettingEProcess` from `.core.eprocess`, `match`/`Trial` from `.matching`, `Report` from `.report`)
- Create: `src/lerobot_doctor/intervals.py` (from `intervals.py`; import `BettingEProcess` from `.core.eprocess`)
- Test: `tests/doctor/test_compare_success.py` (port from `test_direct.py`)
- Delete: `compare.py`, `intervals.py`

**Interfaces:**
- Produces: `compare_success(successes_A, successes_B, *, alpha=0.10, alternative="two-sided", adaptive=True, caliper=1.0, rng=None)->Report`; `clopper_pearson(k, n, alpha)`, `betting_confidence_sequence(diffs, alpha, ...)`

- [ ] **Step 1: Move both files**, updating imports. Behavior unchanged.

- [ ] **Step 2: Port `test_direct.py`** to `tests/doctor/test_compare_success.py` (updated imports). Assert a clear-cut input (`A` all 1s, `B` all 0s) decides "A>B"; equal inputs do not decide.

```python
# tests/doctor/test_compare_success.py (key cases)
import numpy as np
from lerobot_doctor.compare import compare_success
from lerobot_doctor.intervals import clopper_pearson


def test_clear_winner_decides():
    rep = compare_success([1] * 60, [0] * 60, alternative="A>B",
                          rng=np.random.default_rng(0))
    assert rep.decided and rep.direction == "A>B"


def test_equal_does_not_decide():
    rng = np.random.default_rng(0)
    a = (rng.random(400) < 0.6).astype(int).tolist()
    b = (rng.random(400) < 0.6).astype(int).tolist()
    rep = compare_success(a, b, alpha=0.10, rng=np.random.default_rng(1))
    assert rep.decided is False


def test_clopper_pearson_bounds():
    lo, hi = clopper_pearson(5, 10, 0.10)
    assert 0.0 <= lo < hi <= 1.0
```

- [ ] **Step 3: Run**

Run: `pytest tests/doctor/test_compare_success.py -q`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "refactor: relocate v1 compare_success + intervals to package level"
```

### Task 15: `config.py` — draccus config dataclasses

**Files:**
- Create: `src/lerobot_doctor/config.py`
- Test: `tests/doctor/test_config.py`

**Interfaces:**
- Produces: `SimConfig`, `JudgeConfig`, `TrainCfg`, `NotifyConfig`, `DoctorConfig` (nested dataclasses with defaults). Plain dataclasses (draccus parses them; no draccus import needed at definition time).

- [ ] **Step 1: Write the failing test**

```python
# tests/doctor/test_config.py
from lerobot_doctor.config import DoctorConfig, SimConfig


def test_defaults_are_synthetic_and_runnable():
    cfg = DoctorConfig()
    assert cfg.sim.kind == "synthetic"
    assert cfg.judge.kind == "stub"
    assert cfg.alternative in ("A>B", "B>A", "two-sided")
    assert cfg.auditor == "max_evidence"


def test_nested_override():
    cfg = DoctorConfig(sim=SimConfig(pA=0.9, pB=0.3, n=100))
    assert (cfg.sim.pA, cfg.sim.pB, cfg.sim.n) == (0.9, 0.3, 100)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/doctor/test_config.py -q`
Expected: FAIL (module not found)

- [ ] **Step 3: Write `config.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class SimConfig:
    kind: str = "synthetic"            # synthetic | lerobot_env
    pA: float = 0.75
    pB: float = 0.55
    n: int = 1500
    task: str = "put cube in bin"
    repo_A: str = ""                   # lerobot_env: dataset/checkpoint A
    repo_B: str = ""


@dataclass
class JudgeConfig:
    kind: str = "stub"                 # stub | vlm
    rho: float = 0.85                  # stub fidelity
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    combine: str = "mean"              # mean | product | depth


@dataclass
class TrainCfg:
    enabled: bool = False
    cadence: int = 0                   # retrain every N labels (0 = never in-loop)
    backend: str = "llamafactory"      # llamafactory | skypilot
    accelerator: str = "A100:1"
    dry_run: bool = True


@dataclass
class NotifyConfig:
    sink: str = "stdout"               # stdout (telegram | webhook later)


@dataclass
class DoctorConfig:
    sim: SimConfig = field(default_factory=SimConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    alpha: float = 0.10
    alternative: str = "A>B"           # A>B | B>A | two-sided
    auditor: str = "max_evidence"      # max_evidence | uniform | neyman
    batch_size: int = 64
    online: bool = False
    run_dir: str = "runs/doctor"
    seed: int = 0
    train: TrainCfg = field(default_factory=TrainCfg)
    notify: NotifyConfig = field(default_factory=NotifyConfig)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/doctor/test_config.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: draccus-style DoctorConfig dataclasses"
```

### Task 16: `cli.py` — `lerobot-compare` (compare | loop | label)

> The CLI lives at `src/lerobot_doctor/cli.py` for the standalone repo; the codemod
> (Task 19) relocates it to `lerobot/scripts/doctor.py` for the vendored PR.

**Files:**
- Create: `src/lerobot_doctor/cli.py` (extends the existing `cli.py` `compare` with `loop` and `label`)
- Create: `src/lerobot_doctor/loop/factory.py` (`build_loop(cfg)->(loop, notifier)`, `apply_answers(store, answers)`)
- Test: `tests/doctor/test_cli.py`

**Interfaces:**
- Consumes: `DoctorConfig`, `compare_success`, `DoctorLoop`/`OnlineLoop`, `LabelStore`, `Notifier`, `StdoutSink`, `Updater`, `SyntheticSim`, `StubVLM`, auditors
- Produces: `main(argv=None)->int`; `build_loop(cfg)->(DoctorLoop, Notifier)`; `apply_answers(store, answers: dict[str, tuple[int,int]])->int`

- [ ] **Step 1: Write the failing test**

```python
# tests/doctor/test_cli.py
import json
from lerobot_doctor.cli import main, build_loop, apply_answers
from lerobot_doctor.config import DoctorConfig, SimConfig
from lerobot_doctor.loop.store import LabelStore, LabelItem


def test_compare_subcommand(tmp_path, capsys):
    a = tmp_path / "A.txt"; b = tmp_path / "B.txt"
    a.write_text("\n".join(["1"] * 60))
    b.write_text("\n".join(["0"] * 60))
    out = tmp_path / "rep.json"
    rc = main(["compare", str(a), str(b), "--alternative", "A>B", "--json", str(out)])
    assert rc == 0
    assert json.loads(out.read_text())["direction"] == "A>B"


def test_build_loop_auto_decides(tmp_path):
    cfg = DoctorConfig(sim=SimConfig(pA=0.78, pB=0.5, n=1500),
                       run_dir=str(tmp_path / "run"))
    loop, nf = build_loop(cfg, gold_from_sim=True)
    rep = loop.run(); nf.flush(); nf.close()
    assert rep.decided and rep.direction == "A>B"


def test_apply_answers(tmp_path):
    s = LabelStore(str(tmp_path / "q.jsonl"))
    s.enqueue(LabelItem("7", 0.6, 0.4))
    n = apply_answers(s, {"7": (1, 0)})
    assert n == 1 and s.pending() == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/doctor/test_cli.py -q`
Expected: FAIL (module not found)

- [ ] **Step 3: Write `loop/factory.py`**

```python
from __future__ import annotations
import os
from typing import Optional, Tuple
import numpy as np
from ..config import DoctorConfig
from ..core.auditing import MaxEvidenceAuditor, UniformAuditor, NeymanAuditor
from ..evaluators.stub import StubVLM
from ..sim.synthetic import SyntheticSim
from .store import LabelStore, LabelItem
from .notify import Notifier, StdoutSink
from .update import Updater
from .train import TrainingConfig, LlamaFactoryBackend, SkyPilotBackend
from .engine import DoctorLoop
from .online import OnlineLoop

_AUDITORS = {"max_evidence": MaxEvidenceAuditor, "uniform": UniformAuditor,
             "neyman": NeymanAuditor}


def _judge(cfg):
    if cfg.judge.kind == "stub":
        return StubVLM(rho=cfg.judge.rho, _rng=np.random.default_rng(cfg.seed + 1))
    from ..evaluators.vlm import RubricVLM
    return RubricVLM(model_id=cfg.judge.model_id, combine=cfg.judge.combine)


def _train_backend(cfg):
    if not cfg.train.enabled:
        return None
    tc = TrainingConfig(workdir=os.path.join(cfg.run_dir, "train"),
                        output_dir=os.path.join(cfg.run_dir, "judge-lora"))
    if cfg.train.backend == "skypilot":
        return SkyPilotBackend(tc, accelerator=cfg.train.accelerator,
                               dry_run=cfg.train.dry_run)
    return LlamaFactoryBackend(tc, dry_run=cfg.train.dry_run)


def build_loop(cfg: DoctorConfig, gold_from_sim: bool = False) -> Tuple[DoctorLoop, Notifier]:
    os.makedirs(cfg.run_dir, exist_ok=True)
    sim = SyntheticSim(pA=cfg.sim.pA, pB=cfg.sim.pB, n=cfg.sim.n, task=cfg.sim.task,
                       rng=np.random.default_rng(cfg.seed))
    judge = _judge(cfg)
    store = LabelStore(os.path.join(cfg.run_dir, "queue.jsonl"))
    nf = Notifier(StdoutSink())
    up = Updater(judge, retrain_cadence=cfg.train.cadence)
    gold = sim.gold if gold_from_sim else None
    Driver = OnlineLoop if cfg.online else DoctorLoop
    loop = Driver(sim, judge, _AUDITORS[cfg.auditor](), store, nf, up, gold=gold,
                  train_backend=_train_backend(cfg), alpha=cfg.alpha,
                  alternative=cfg.alternative, batch_size=cfg.batch_size,
                  rng=np.random.default_rng(cfg.seed + 99))
    return loop, nf


def apply_answers(store: LabelStore, answers) -> int:
    n = 0
    for iid, (yA, yB) in answers.items():
        store.answer(iid, yA, yB)
        n += 1
    return n
```

- [ ] **Step 4: Write `src/lerobot_doctor/cli.py`** (keep the existing `compare` reader; add subcommands)

```python
from __future__ import annotations
import argparse
import json
import sys
from typing import List
from .compare import compare_success
from .config import DoctorConfig
from .loop.factory import build_loop, apply_answers
from .loop.store import LabelStore


def _read_successes(path: str) -> List[int]:
    vals: List[int] = []
    with open(path) as f:
        first = True
        for line in f:
            s = line.strip()
            if not s:
                continue
            if "," in s:
                parts = [p.strip() for p in s.split(",")]
                if first and not parts[0].lstrip("-").isdigit():
                    first = False
                    continue
                s = parts[-1]
            first = False
            vals.append(int(float(s)))
    return vals


def _cmd_compare(args) -> int:
    A, B = _read_successes(args.file_A), _read_successes(args.file_B)
    rep = compare_success(A, B, alpha=args.alpha, alternative=args.alternative)
    print(rep.summary())
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rep.to_dict(), f, indent=2)
    return 0 if rep.decided else 2


def _cmd_loop(args) -> int:
    cfg = DoctorConfig(run_dir=args.run_dir)
    cfg.online = args.online
    loop, nf = build_loop(cfg, gold_from_sim=args.auto)
    rep = loop.run()
    nf.flush(); nf.close()
    print(rep.summary())
    print(f"status: {rep.status}  (queue: {loop.store.path})")
    return 0 if rep.decided else (3 if rep.status == "needs_labels" else 2)


def _cmd_label(args) -> int:
    store = LabelStore(args.run_dir + "/queue.jsonl")
    pend = store.pending()
    if not pend:
        print("nothing to label."); return 0
    answers = {}
    for it in pend:
        line = input(f"[{it.item_id}] A success? B success? (e.g. 'y n'): ").split()
        ya = 1 if line and line[0].lower().startswith("y") else 0
        yb = 1 if len(line) > 1 and line[1].lower().startswith("y") else 0
        answers[it.item_id] = (ya, yb)
    n = apply_answers(store, answers)
    print(f"answered {n}; re-run `lerobot-compare loop` to resume.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="lerobot-compare")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compare", help="anytime-valid A/B success test")
    c.add_argument("file_A"); c.add_argument("file_B")
    c.add_argument("--alpha", type=float, default=0.10)
    c.add_argument("--alternative", default="two-sided",
                   choices=["two-sided", "A>B", "B>A"])
    c.add_argument("--json", metavar="OUT")
    c.set_defaults(fn=_cmd_compare)

    l = sub.add_parser("loop", help="run the human-in-the-loop comparison")
    l.add_argument("--run-dir", dest="run_dir", default="runs/doctor")
    l.add_argument("--online", action="store_true")
    l.add_argument("--auto", action="store_true",
                   help="auto-answer from the synthetic oracle (laptop demo)")
    l.set_defaults(fn=_cmd_loop)

    lb = sub.add_parser("label", help="answer pending items, then resume loop")
    lb.add_argument("--run-dir", dest="run_dir", default="runs/doctor")
    lb.set_defaults(fn=_cmd_label)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run to verify pass**

Run: `pytest tests/doctor/test_cli.py -q`
Expected: PASS

- [ ] **Step 6: Manual smoke (optional)**

Run: `python -m lerobot_doctor.cli loop --auto --run-dir runs/demo`
Expected: prints a `DECISION: A>B` summary + `status: decided`.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: lerobot-compare CLI (compare|loop|label) + loop factory"
```

---

## Phase 6 — Real backends (code + mock tests; real execution is post-plan)

### Task 17: `evaluators/vlm.py` — VLMJudge + RubricVLM (lazy torch)

**Files:**
- Create: `src/lerobot_doctor/evaluators/vlm.py` (consolidate `VLMJudge` from `evaluators.py` and `RubricVLM` from `robotics_adapter.py`; both score a `Rollout` from `.base`; all torch/transformers/qwen imports stay inside methods)
- Test: `tests/doctor/test_vlm_mock.py`

**Interfaces:**
- Produces: `RubricVLM(model_id, rubric, combine="mean", max_frames=8, cost=1.0)` with `.score(Rollout)->float`; `VLMJudge(...)` alias/equivalent. Public combine modes: `mean | product | depth`.

- [ ] **Step 1: Move both classes** into `vlm.py`, importing `Rollout` from `.base`. Keep lazy imports.

- [ ] **Step 2: Write a mock test** (no weights, no torch) that monkeypatches `_p_yes` to verify the combine logic and frame subsampling:

```python
# tests/doctor/test_vlm_mock.py
from lerobot_doctor.evaluators.vlm import RubricVLM
from lerobot_doctor.evaluators.base import Rollout


def test_combine_modes(monkeypatch):
    ev = RubricVLM(rubric=("q1?", "q2?", "q3?"), combine="mean")
    # fake the model: q1->0.9, q2->0.8, q3->0.2 regardless of frames
    answers = iter([0.9, 0.8, 0.2])
    monkeypatch.setattr(ev, "_ensure_loaded", lambda: None)
    monkeypatch.setattr(ev, "_p_yes", lambda task, frames, q: next(answers))
    assert abs(ev.score(Rollout("t", frames=["f"])) - (0.9 + 0.8 + 0.2) / 3) < 1e-9

    ev2 = RubricVLM(rubric=("q1?", "q2?", "q3?"), combine="depth")
    monkeypatch.setattr(ev2, "_ensure_loaded", lambda: None)
    seq = iter([0.9, 0.8, 0.2])
    monkeypatch.setattr(ev2, "_p_yes", lambda task, frames, q: next(seq))
    assert ev2.score(Rollout("t", frames=["f"])) == 2 / 3   # depth stops at first <0.5


def test_subsample_caps_frames():
    ev = RubricVLM(max_frames=4)
    assert len(ev._subsample(list(range(20)))) == 4
    assert ev._subsample([]) == []
```

- [ ] **Step 3: Run**

Run: `pytest tests/doctor/test_vlm_mock.py -q`
Expected: PASS (combine + subsample logic verified without weights)

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: consolidate VLM judges into evaluators/vlm (lazy torch)"
```

### Task 18: `sim/lerobot_env.py` — LeRobot loader (lazy)

**Files:**
- Create: `src/lerobot_doctor/sim/lerobot_env.py` (`load_episodes`, `build_pairs` from `robotics_adapter.py`; lazy-import lerobot; yield `Pair` from `.base` using `Rollout` from `..evaluators.base`)
- Test: `tests/doctor/test_lerobot_env_mock.py`

**Interfaces:**
- Produces: `load_episodes(repo_id, camera_key="observation.images.top", task_key="task")->Iterator[Rollout]`; `build_pairs(rollouts_A, rollouts_B)->List[Pair]`

- [ ] **Step 1: Move + adapt** to emit `sim.base.Pair` / `evaluators.base.Rollout`. Keep the `from lerobot...` import inside `load_episodes`.

- [ ] **Step 2: Write a mock test** (no lerobot installed): test `build_pairs` directly, and that `load_episodes` raises a clean `ImportError`/`ModuleNotFoundError` rather than failing at import time.

```python
# tests/doctor/test_lerobot_env_mock.py
import pytest
from lerobot_doctor.sim.lerobot_env import build_pairs, load_episodes
from lerobot_doctor.evaluators.base import Rollout
from lerobot_doctor.sim.base import Pair


def test_build_pairs_positional():
    A = [Rollout("t", key=i) for i in range(3)]
    B = [Rollout("t", key=i) for i in range(3)]
    pairs = build_pairs(A, B)
    assert len(pairs) == 3 and all(isinstance(p, Pair) for p in pairs)


def test_load_episodes_lazy_import_only_when_called():
    # module imported fine without lerobot; the dep is needed only on use
    gen = load_episodes("nonexistent/repo")
    with pytest.raises((ImportError, ModuleNotFoundError, Exception)):
        next(gen)
```

- [ ] **Step 3: Run**

Run: `pytest tests/doctor/test_lerobot_env_mock.py -q`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: lerobot gym episode loader (lazy import)"
```

---

## Phase 7 — Packaging, public API, docs

### Task 19: `scripts/vendor_to_lerobot.py` — the codemod (PR-readiness)

**Files:**
- Create: `scripts/vendor_to_lerobot.py`
- Test: `tests/doctor/test_vendor_codemod.py`

**Interfaces:**
- Produces: `vendor(src_root="src/lerobot_doctor", out_root=...) -> Path` — copies the package to `<out>/lerobot/doctor/`, rewrites absolute `lerobot_doctor` → `lerobot.doctor` in all `.py`, and relocates `cli.py` → `<out>/lerobot/scripts/doctor.py`. Returns the emitted `lerobot/` dir.

- [ ] **Step 1: Write the failing test**

```python
# tests/doctor/test_vendor_codemod.py
import subprocess, sys, pathlib


def test_codemod_emits_clean_lerobot_doctor(tmp_path):
    out = tmp_path / "vendored"
    subprocess.check_call([sys.executable, "scripts/vendor_to_lerobot.py",
                           "--out", str(out)])
    doctor = out / "lerobot" / "doctor"
    assert (doctor / "core" / "auditing.py").exists()
    assert (out / "lerobot" / "scripts" / "doctor.py").exists()
    # no stale absolute references remain
    hits = []
    for p in doctor.rglob("*.py"):
        if "lerobot_doctor" in p.read_text():
            hits.append(str(p))
    assert hits == [], f"stale lerobot_doctor refs in {hits}"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/doctor/test_vendor_codemod.py -q`
Expected: FAIL (script missing)

- [ ] **Step 3: Write `scripts/vendor_to_lerobot.py`**

```python
#!/usr/bin/env python3
"""Emit src/lerobot_doctor/ as a drop-in lerobot/doctor/ subtree for the lerobot PR.

PR procedure: run this, copy <out>/lerobot/ over a lerobot checkout's src/lerobot/,
add the optional extras + `lerobot-compare = lerobot.scripts.doctor:main` to lerobot's
pyproject. That is the whole change.
"""
from __future__ import annotations
import argparse
import re
import shutil
from pathlib import Path


def vendor(src_root: str = "src/lerobot_doctor", out_root: str = "build/vendored") -> Path:
    src = Path(src_root)
    out = Path(out_root)
    doctor = out / "lerobot" / "doctor"
    scripts = out / "lerobot" / "scripts"
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src, doctor)

    # relocate the CLI: lerobot_doctor/cli.py -> lerobot/scripts/doctor.py
    scripts.mkdir(parents=True, exist_ok=True)
    cli = doctor / "cli.py"
    if cli.exists():
        moved = scripts / "doctor.py"
        moved.write_text(cli.read_text())
        cli.unlink()

    # rewrite absolute references in every emitted .py
    pat = re.compile(r"\blerobot_doctor\b")
    for p in list(doctor.rglob("*.py")) + [scripts / "doctor.py"]:
        text = p.read_text()
        new = pat.sub("lerobot.doctor", text)
        # the moved CLI used package-relative imports (from .compare ...); make them
        # absolute against the new location
        if p.name == "doctor.py" and p.parent.name == "scripts":
            new = new.replace("from .", "from lerobot.doctor.")
        if new != text:
            p.write_text(new)
    return out / "lerobot"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="src/lerobot_doctor")
    ap.add_argument("--out", default="build/vendored")
    a = ap.parse_args()
    dest = vendor(a.src, a.out)
    print(f"vendored -> {dest}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/doctor/test_vendor_codemod.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: vendor_to_lerobot codemod + PR-readiness test"
```

### Task 20: Public API `__init__.py` + lazy-import guarantee

**Files:**
- Modify: `src/lerobot_doctor/__init__.py`
- Test: `tests/doctor/test_public_api.py`

**Interfaces:**
- Produces: top-level names per spec §11. Heavy classes (`VLMJudge`, `RubricVLM`, `lerobot_env` loaders) are exposed via `__getattr__` lazy import so base import needs only numpy/scipy.

- [ ] **Step 1: Write the failing test**

```python
# tests/doctor/test_public_api.py
import importlib


def test_base_api_imports_without_heavy_deps():
    ld = importlib.import_module("lerobot_doctor")
    for name in ("compare_policies", "compare_success", "Report", "Rollout",
                 "HumanGold", "StubVLM", "Recalibrate", "MaxEvidenceAuditor",
                 "UniformAuditor", "NeymanAuditor", "SyntheticSim", "LabelStore",
                 "Notifier", "StdoutSink", "Updater", "DoctorLoop", "OnlineLoop",
                 "TrainingConfig", "LlamaFactoryBackend", "SkyPilotBackend"):
        assert hasattr(ld, name), name


def test_heavy_names_are_lazy(monkeypatch):
    import sys
    # base import must not have imported torch
    importlib.import_module("lerobot_doctor")
    assert "torch" not in sys.modules
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/doctor/test_public_api.py -q`
Expected: FAIL (names not exported)

- [ ] **Step 3: Write `__init__.py`**

```python
"""lerobot-compare: label-efficient, anytime-valid policy comparison for lerobot."""
from .compare import compare_success
from .intervals import clopper_pearson, betting_confidence_sequence
from .report import Report
from .matching import match, Trial, Pairing
from .checks import scan_success_flags, preflight_rollouts, emit
from .core.auditing import MaxEvidenceAuditor, UniformAuditor, NeymanAuditor, Auditor
from .core.compare import compare, Result, make_world, GoldSource
from .evaluators.base import Rollout, Evaluator, HumanGold
from .evaluators.stub import StubVLM
from .evaluators.internal import AutomaticChecker, InternalState
from .evaluators.calibrate import Recalibrate, CalibratedEvaluator, LabelRecord
from .sim.base import RolloutSource, Pair
from .sim.synthetic import SyntheticSim
from .loop.store import LabelStore, LabelItem
from .loop.notify import Notifier, StdoutSink, Sink, Notification
from .loop.update import Updater
from .loop.train import TrainingConfig, LlamaFactoryBackend, SkyPilotBackend, write_dataset
from .loop.engine import DoctorLoop, compare_policies
from .loop.online import OnlineLoop
from .config import DoctorConfig

__version__ = "0.3.0"

_LAZY = {
    "VLMJudge": ("lerobot_doctor.evaluators.vlm", "VLMJudge"),
    "RubricVLM": ("lerobot_doctor.evaluators.vlm", "RubricVLM"),
    "load_episodes": ("lerobot_doctor.sim.lerobot_env", "load_episodes"),
    "build_pairs": ("lerobot_doctor.sim.lerobot_env", "build_pairs"),
}


def __getattr__(name):                       # PEP 562 lazy heavy deps
    if name in _LAZY:
        import importlib
        mod, attr = _LAZY[name]
        return getattr(importlib.import_module(mod), attr)
    raise AttributeError(name)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/doctor/test_public_api.py -q`
Expected: PASS

- [ ] **Step 5: Full suite + coverage gate + Global Constraint gates**

Run: `pytest tests/doctor -q -m "not slow" && pytest tests/doctor -q -m slow`
Then: `pytest tests/doctor --cov=lerobot_doctor --cov-report=term-missing -q -m "not slow"`
Expected: all PASS; `core/` and `loop/` ≥90% lines. (Mark genuinely unreachable
real-execution branches — `train.py` `dry_run=False`/`subprocess.run` — with
`# pragma: no cover`; the coverage gate measures the testable surface.)

Zero-`parsimony` gate — run from repo root:
`git grep -i parsimony -- src tests bench examples scripts pyproject.toml README.md docs/method.md`
Expected: **no output** (exit 1). Any hit is a failure to fix before commit.

Zero-`v1`/`v2`-label gate:
`git grep -nE '\bv1\b|\bv2\b' -- src tests examples`
Expected: **no output**. Mode labels are `direct`/`audited`, not version numbers.

License-header gate — every shipped `.py` starts with the Apache header:
`for f in $(git ls-files 'src/*.py' 'tests/*.py' 'bench/*.py' 'examples/*.py' 'scripts/*.py'); do head -3 "$f" | grep -q "Copyright 2026 Kaveh Shoorideh" || echo "MISSING HEADER: $f"; done`
Expected: no `MISSING HEADER` lines.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: curated public API with lazy heavy deps"
```

### Task 21: Docs, bench, examples, README

**Files:**
- Create: `docs/method.md` (≤2 pages; from `paper.md`/`efficiency-notes.md` content — known-technique framing, references, the Neyman caveat)
- Create: `bench/validate.py` (from `experiments.py`; regenerate validity/efficiency plots into `bench/figures/`)
- Move: `audited_demo.py`, `compare_demo.py`, `offline_compare_stub.py`, `robotics_qwen_lerobot.py` → `examples/` (update imports to `lerobot_doctor`)
- Rewrite: `README.md` (install via extras, the three goals, `lerobot-compare loop --auto` quickstart, the vendoring/PR procedure)
- Delete: `paper.md`, `efficiency-notes.md`, `experiments.py`, root demos

**Interfaces:** none (docs/examples).

- [ ] **Step 1: Write `docs/method.md`** — what the tool does; the honest "assembled from PPI/PPI++, prediction-powered e-values, WSR betting, CUPED, controlled sensing — brought to lerobot, no novelty claim"; references; one paragraph on the Neyman-allocation pitfall (audit decisive, not uncertain).

- [ ] **Step 2: Relocate `experiments.py` → `bench/validate.py`**, fixing imports to `from lerobot_doctor...`. It must run and write figures.

- [ ] **Step 3: Move the four demos to `examples/`**, fixing imports.

- [ ] **Step 4: Rewrite `README.md`** with: extras install, the laptop quickstart (`python -m lerobot_doctor.cli loop --auto`), what runs where, and the `scripts/vendor_to_lerobot.py` PR procedure.

- [ ] **Step 5: Verify examples + bench run**

Run: `python examples/audited_demo.py` and `python bench/validate.py`
Expected: both run to completion (bench writes `bench/figures/*.png`).

- [ ] **Step 6: Run the whole suite once more**

Run: `pytest -q -m "not slow"`
Expected: PASS (green tree)

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "docs: method note, bench, examples, README; drop paper"
```

---

## Self-Review

**Spec coverage:** §2 unify-onto-package → Tasks 2–4, 10, 14. §3 rename → Task 2 + global. §4 layout + codemod → Tasks 1, 19. §5 core → Task 2. §6 loop (store/notify/update/engine/online + invariant) → Tasks 6–13. §7 sim → Tasks 5, 18. §8 runs-where → Tasks 15–18 (backends behind config). §9 CLI → Task 16. §10 docs → Task 21. §11 public API → Task 20. §12 tests (5 tiers) → Tier1 Task 2; Tier2 Tasks 2,12; Tier3 Tasks 3,4,6,7,8,9; Tier4 Tasks 11,13,16; Tier5 Tasks 19,20. §13 migration → all relocate tasks (refined: v1 compare/intervals at package level, CLI at `lerobot_doctor/cli.py`). §14 sequencing → phase order.

**Refinements logged:** (1) `compare_success`/`intervals` live at package level, not `core/` (core purity). (2) CLI is `lerobot_doctor/cli.py` standalone, relocated by the codemod — avoids a `src/lerobot/` namespace collision in the standalone repo. (3) `Report` gains a trailing `status: str = "running"` field (Task 4 / used in Task 11).

**Placeholder scan:** none — every code/test step carries full content.

**Type consistency:** `Auditor.audit_prob(pred_A, pred_B, gold_cost)`, `Comparison.payoff(Obs)`/`.observe(Obs)`, `Obs(pred_A, pred_B, audited, pi, gold_A, gold_B)`, `Updater.ingest(f, y, frames, question)`/`.refresh_judge(retrain_backend)`, `LabelStore.enqueue/answer/pending/answered`, `Notifier.notify/flush/close`, `DoctorLoop`/`OnlineLoop` shared ctor, `build_loop(cfg, gold_from_sim)`, `vendor(src_root, out_root)` — consistent across producer/consumer blocks.

## Post-plan (your validation, not in-plan)

These need a Linux GPU box / cloud and are validated by you after the plan lands:
- Real `RubricVLM` (Qwen2.5-VL) scoring real frames.
- Real `lerobot_env` two-checkpoint rollouts in `gym-pusht`.
- `dry_run=False` LoRA training via LLaMA-Factory on SkyPilot / HF Jobs.
- A real notify sink (Telegram/webhook) behind the existing `Sink` interface.
