"""
training.py -- retrain handoff with a clean, swappable interface.

The boundary the open-source ecosystem already standardized: emit a dataset (HF /
LLaMA-Factory JSONL) + a config, and any framework consumes it; checkpoints come
back as interchangeable HF LoRA adapters. So `lerobot-compare` only owns the
dataset + config; WHAT trains and WHERE it runs are swappable plugins.

Decided defaults (see README): training = LLaMA-Factory (native Qwen2.5-VL, CLI),
launcher = SkyPilot (same job runs local or any cloud). Axolotl/Unsloth and Modal
drop in behind the same `update(records) -> TrainResult` surface.

Everything defaults to `dry_run=True`: it WRITES the dataset + config + the exact
command, and returns them, without running. Flip `dry_run=False` on a machine that
has the tools + a GPU (or cloud creds) to actually train. The sandbox can't
download weights or train, so the real path is validated by you, not here.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, List, Optional

from .calibrate import LabelRecord, EvaluatorUpdater


@dataclass
class TrainingConfig:
    base_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    method: str = "lora"
    lora_rank: int = 8
    epochs: float = 2.0
    learning_rate: float = 1.0e-4
    output_dir: str = "out/judge-lora"
    workdir: str = "out/train"


@dataclass
class TrainResult:
    dataset_path: str
    config_path: str
    command: List[str]
    adapter_dir: str
    ran: bool = False
    returncode: Optional[int] = None


def write_dataset(records: List[LabelRecord], path: str) -> int:
    """Write audited labels as a LLaMA-Factory multimodal SFT JSONL.

    Each record becomes a (frames + question) -> "yes"/"no" example, i.e. we teach
    the judge to predict the human verdict it would have given. Returns #examples.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = 0
    with open(path, "w") as fh:
        for r in records:
            imgs = r.frames if isinstance(r.frames, list) else (
                [r.frames] if r.frames else [])
            q = r.question or "Did the robot complete the task successfully?"
            ex = {
                "messages": [
                    {"role": "user",
                     "content": "<image>" * len(imgs) + q + " Answer 'yes' or 'no'."},
                    {"role": "assistant", "content": "yes" if r.y == 1 else "no"},
                ],
                "images": imgs,
            }
            fh.write(json.dumps(ex) + "\n")
            n += 1
    return n


def _write_llamafactory_yaml(cfg: TrainingConfig, dataset_path: str,
                             config_path: str) -> None:
    ds_dir = os.path.dirname(dataset_path) or "."
    ds_name = "lerobot_doctor_judge"
    with open(os.path.join(ds_dir, "dataset_info.json"), "w") as f:
        json.dump({ds_name: {"file_name": os.path.basename(dataset_path),
                             "formatting": "sharegpt",
                             "columns": {"messages": "messages", "images": "images"}}},
                  f, indent=2)
    yaml = f"""### lerobot-compare judge fine-tune (LLaMA-Factory)
model_name_or_path: {cfg.base_model}
trust_remote_code: true
stage: sft
do_train: true
finetuning_type: {cfg.method}
lora_rank: {cfg.lora_rank}
lora_target: all
dataset_dir: {ds_dir}
dataset: {ds_name}
template: qwen2_vl
cutoff_len: 2048
output_dir: {cfg.output_dir}
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: {cfg.learning_rate}
num_train_epochs: {cfg.epochs}
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
"""
    with open(config_path, "w") as f:
        f.write(yaml)


@dataclass
class LlamaFactoryBackend:
    """Default retrain backend. Writes a LoRA YAML for the VLM judge and runs
    `llamafactory-cli train`. dry_run writes artifacts + returns the command."""
    config: TrainingConfig = field(default_factory=TrainingConfig)
    dry_run: bool = True

    def update(self, records: List[LabelRecord]) -> TrainResult:
        c = self.config
        os.makedirs(c.workdir, exist_ok=True)
        ds = os.path.join(c.workdir, "judge_data.jsonl")
        cfgp = os.path.join(c.workdir, "train.yaml")
        write_dataset(records, ds)
        _write_llamafactory_yaml(c, ds, cfgp)
        cmd = ["llamafactory-cli", "train", cfgp]
        res = TrainResult(ds, cfgp, cmd, c.output_dir)
        if not self.dry_run:                          # real path: user's GPU
            p = subprocess.run(cmd)
            res.ran = True; res.returncode = p.returncode
        return res


@dataclass
class SkyPilotBackend:
    """Launcher backend: wraps the LLaMA-Factory job in a SkyPilot task so the
    SAME job runs locally or on any cloud. dry_run writes the task YAML + command."""
    config: TrainingConfig = field(default_factory=TrainingConfig)
    accelerator: str = "A100:1"
    infra: Optional[str] = None                       # e.g. "k8s", "aws", None=auto
    dry_run: bool = True

    def update(self, records: List[LabelRecord]) -> TrainResult:
        c = self.config
        inner = LlamaFactoryBackend(c, dry_run=True).update(records)
        task = os.path.join(c.workdir, "sky_task.yaml")
        infra = f"infra: {self.infra}\n" if self.infra else ""
        with open(task, "w") as f:
            f.write(f"""# lerobot-compare judge fine-tune (SkyPilot launcher)
resources:
  accelerators: {self.accelerator}
  {infra}workdir: {c.workdir}
setup: |
  pip install llamafactory[torch,metrics] "transformers>=4.49"
run: |
  llamafactory-cli train {os.path.basename(inner.config_path)}
""")
        cmd = ["sky", "launch", "-c", "lerobot-compare-judge", task]
        res = TrainResult(inner.dataset_path, task, cmd, c.output_dir)
        if not self.dry_run:                          # real path: any cloud
            p = subprocess.run(cmd)
            res.ran = True; res.returncode = p.returncode
        return res
