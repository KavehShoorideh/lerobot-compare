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

"""Real VLM judge (lazy: needs the [vlm] extra).

Heavy deps (torch/transformers) are imported lazily inside methods so that
`import lerobot_doctor.evaluators` stays numpy+scipy only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

from .base import Rollout


# ---- real VLM judge (lazy: needs the [vlm] extra) -------------------------

@dataclass
class VLMJudge:
    """Calibrated VLM judge over a rubric of yes/no subgoal questions. Reads
    P('yes') from the model's next-token logits, so the score is graded [0,1]
    rather than a parsed word. combine: 'mean' | 'product' | 'depth'.

    Needs the [vlm] extra (transformers, torch, qwen-vl-utils, pillow) and is
    imported lazily, so it never blocks sandbox/import-only use.
    """
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    rubric: Tuple[str, ...] = ("Did the robot complete the task successfully?",)
    combine: str = "mean"
    max_frames: int = 8
    cost: float = 1.0
    _model: Any = None
    _proc: Any = None

    def _load(self):
        if self._model is not None:
            return
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype="auto", device_map="auto")
        self._proc = AutoProcessor.from_pretrained(self.model_id)

    def _p_yes(self, frames, question: str) -> float:
        import torch
        from qwen_vl_utils import process_vision_info
        msgs = [{"role": "user", "content": (
            [{"type": "image", "image": f} for f in frames]
            + [{"type": "text",
                "text": question + " Answer strictly 'yes' or 'no'."}])}]
        text = self._proc.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=True)
        imgs, vids = process_vision_info(msgs)
        inp = self._proc(text=[text], images=imgs, videos=vids,
                         return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            logits = self._model(**inp).logits[0, -1]
        tok = self._proc.tokenizer
        y = tok(" yes", add_special_tokens=False).input_ids[0]
        n = tok(" no", add_special_tokens=False).input_ids[0]
        p = torch.softmax(logits[[y, n]].float(), 0)[0]
        return float(p)

    def score(self, rollout: Rollout) -> float:
        self._load()
        frames = rollout.frames[: self.max_frames] if rollout.frames else []
        ps = [self._p_yes(frames, q) for q in self.rubric]
        ps = np.asarray(ps, dtype=float)
        if self.combine == "product":
            return float(np.prod(ps))
        if self.combine == "depth":                 # ordinal progress depth
            passed = 0
            for v in ps:
                if v >= 0.5:
                    passed += 1
                else:
                    break
            return passed / len(ps)
        return float(ps.mean())                     # "mean": partial credit
