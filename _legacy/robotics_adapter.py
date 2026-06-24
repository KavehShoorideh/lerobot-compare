"""
robotics_adapter - robotics adapter for the lerobot_doctor core.

Wires real robot evaluators into the domain-agnostic engine:
  - RubricVLM        : Qwen2.5-VL as a CALIBRATED success evaluator. Asks a
                       rubric of yes/no subgoal questions over the rollout
                       frames and reads P("yes") from the token logits (a graded
                       score in [0,1]), not a parsed string -- so the output is
                       something the calibration layer can actually calibrate.
  - InternalState    : a near-free evaluator from policy internals
                       (action-chunk entropy / proprioceptive heuristic).
  - LeRobot loader   : turn recorded episodes of policy A and policy B into the
                       paired items the core compares.
  - compare_policies : the one-call entry point (paired predict + core.compare).

Boundary discipline: the core (`lerobot_doctor`) imports numpy only. ALL heavy deps
here (torch, transformers, qwen_vl_utils, lerobot, PIL) are imported LAZILY
inside the methods that need them, so `import robotics_adapter` works -- and
the core stays pure -- even when none of them are installed. A torch-free
StubVLM is provided so the full wiring is testable without a GPU or weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple

import lerobot_doctor as core


# ===========================================================================
# Data types
# ===========================================================================

@dataclass
class Rollout:
    """One policy's attempt at a task. `frames` are PIL images or file paths
    (for the real VLM); `meta` carries anything else (e.g. proprio arrays for
    the internal-state evaluator, or a hidden truth for the stub)."""
    task: str
    frames: Optional[List[Any]] = None
    meta: dict = field(default_factory=dict)


@dataclass
class Pair:
    """A paired task instance: the same task done by policy A and policy B."""
    task: str
    A: Rollout
    B: Rollout


def paired_predict(item: Pair, evaluator) -> Tuple[float, float]:
    """The `predict` the core expects: score each policy's rollout."""
    return float(evaluator.score(item.A)), float(evaluator.score(item.B))


# ===========================================================================
# VLM rubric evaluator (Qwen2.5-VL), calibrated via P(yes) logits
# ===========================================================================

@dataclass
class RubricVLM:
    """Cheap, unreliable success evaluator: a vision-language model answering a
    rubric of yes/no subgoal questions over the rollout frames.

    score(rollout) in [0,1]:
      - "mean"    : average P(yes) over subgoals  -> graded partial progress
      - "product" : prod P(yes)                   -> strict all-must-pass
      - "depth"   : normalized progress depth     -> ordinal, encodes ordering

    P(yes) is read from the next-token logits (softmax over the yes/no token
    ids), giving a graded, calibratable probability rather than a parsed word.
    """
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    rubric: Tuple[str, ...] = ("Has the task been completed successfully?",)
    combine: str = "mean"
    max_frames: int = 8
    cost: float = 1.0
    name: str = "qwen2.5-vl"
    _model: Any = None
    _proc: Any = None
    _yes_ids: Any = None
    _no_ids: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch  # noqa: lazy
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype="auto", device_map="auto")
        self._proc = AutoProcessor.from_pretrained(self.model_id)
        tok = self._proc.tokenizer
        # candidate token ids for "yes"/"no" (with and without leading space)
        self._yes_ids = {i for w in ("Yes", " Yes", "yes", " yes")
                         for i in tok(w, add_special_tokens=False).input_ids[:1]}
        self._no_ids = {i for w in ("No", " No", "no", " no")
                        for i in tok(w, add_special_tokens=False).input_ids[:1]}

    def _subsample(self, frames: List[Any]) -> List[Any]:
        if not frames or len(frames) <= self.max_frames:
            return frames or []
        step = len(frames) / self.max_frames
        return [frames[int(i * step)] for i in range(self.max_frames)]

    def _p_yes(self, task: str, frames: List[Any], question: str) -> float:
        import torch  # noqa: lazy
        from qwen_vl_utils import process_vision_info
        frames = self._subsample(frames)
        content = [{"type": "image", "image": f} for f in frames]
        content.append({"type": "text", "text":
                        f"Task: {task}\nQuestion: {question}\n"
                        f"Answer with a single word, Yes or No."})
        messages = [{"role": "user", "content": content}]
        text = self._proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._proc(text=[text], images=image_inputs, videos=video_inputs,
                            padding=True, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            logits = self._model(**inputs).logits[0, -1]      # next-token logits
        yes = torch.logsumexp(logits[list(self._yes_ids)], dim=0)
        no = torch.logsumexp(logits[list(self._no_ids)], dim=0)
        p = torch.softmax(torch.stack([no, yes]), dim=0)[1]
        return float(p)

    def score(self, rollout: Rollout) -> float:
        self._ensure_loaded()
        ps = [self._p_yes(rollout.task, rollout.frames, q) for q in self.rubric]
        if self.combine == "product":
            out = 1.0
            for p in ps:
                out *= p
            return out
        if self.combine == "depth":               # ordinal progress depth in [0,1]
            depth = 0
            for p in ps:
                if p < 0.5:
                    break
                depth += 1
            return depth / len(ps)
        return sum(ps) / len(ps)                   # "mean" graded partial progress


# ===========================================================================
# Internal-state evaluator (near-free route)
# ===========================================================================

@dataclass
class InternalState:
    """Near-free evaluator from policy internals. `fn(rollout) -> [0,1]` reads
    whatever you stashed in rollout.meta (action-chunk entropy, an OOD score, a
    proprioceptive goal-region check, ...). Cost ~ a few flops."""
    fn: Callable[[Rollout], float]
    cost: float = 0.001
    name: str = "internal-state"

    def score(self, rollout: Rollout) -> float:
        return float(self.fn(rollout))


# ===========================================================================
# LeRobot loader  (lazy import)
# ===========================================================================

def load_episodes(repo_id: str, camera_key: str = "observation.images.top",
                  task_key: str = "task"):
    """Yield Rollouts from a LeRobot dataset (one per episode). Lazy-imports
    lerobot so the adapter imports without it."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: lazy
    ds = LeRobotDataset(repo_id)
    from_idx = ds.episode_data_index["from"]
    to_idx = ds.episode_data_index["to"]
    for ep in range(ds.num_episodes):
        i0, i1 = int(from_idx[ep]), int(to_idx[ep])
        frames = [ds[i][camera_key] for i in range(i0, i1)]
        task = ds[i0].get(task_key, "")
        yield Rollout(task=str(task), frames=frames)


def build_pairs(rollouts_A: Sequence[Rollout],
                rollouts_B: Sequence[Rollout]) -> List[Pair]:
    """Pair policy-A and policy-B rollouts positionally (same task order)."""
    return [Pair(task=a.task, A=a, B=b) for a, b in zip(rollouts_A, rollouts_B)]


def human_gold(label_fn: Callable[[Pair], Tuple[int, int]], cost: float = 60.0):
    """A GoldSource for human success labels. label_fn(pair) -> (yA, yB)."""
    return core.GoldSource(cost=cost, label_fn=label_fn)


# ===========================================================================
# One-call entry point
# ===========================================================================

def compare_policies(pairs: Sequence[Pair], evaluator, gold,
                     rule=None, alpha: float = 0.10, adaptive: bool = True,
                     control_variate: bool = True, rng=None) -> core.Result:
    """Is policy A better than B? Streams paired rollouts, leans on the cheap
    evaluator, adaptively spends scarce human gold, stops with an anytime-valid
    guarantee. `evaluator` is any object with .score(Rollout)->[0,1] and .cost."""
    rule = rule or core.MaxEvidencePerCost()
    return core.compare(list(pairs), evaluator, gold, paired_predict,
                        rule=rule, alpha=alpha, adaptive=adaptive,
                        control_variate=control_variate, rng=rng)


# ===========================================================================
# Torch-free stub, for testing the wiring without a GPU or weights
# ===========================================================================

@dataclass
class StubVLM:
    """A fake VLM for tests: returns a graded, noisy score correlated with a
    hidden truth stashed in rollout.meta['p'] (its reliability set by `rho`)."""
    rho: float = 0.8
    cost: float = 1.0
    name: str = "stub-vlm"
    seed: int = 0
    _rng: Any = None

    def score(self, rollout: Rollout) -> float:
        import numpy as np
        if self._rng is None:
            self._rng = np.random.default_rng(self.seed)
        p = float(rollout.meta.get("p", 0.5))
        k = 3.0 * (2.0 * self.rho - 1.0)
        logit = k * (2 * p - 1) + self._rng.normal(0.0, 1.0)
        return float(1.0 / (1.0 + np.exp(-logit)))
