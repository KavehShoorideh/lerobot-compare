"""
examples/v2_demo.py  --  run:  python3 examples/v2_demo.py

enhanced in action (no GPU, no download -- uses the torch-free StubVLM):
  1. compare two policies leaning on a cheap evaluator, auditing only some pairs
  2. recalibrate the evaluator from the human labels collected
  3. hand off a retrain job to LLaMA-Factory / SkyPilot (dry-run: writes artifacts)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from lerobot_doctor import (
    compare_policies, Rollout, HumanGold, StubVLM,
    Recalibrate, LabelRecord, TrainingConfig, LlamaFactoryBackend, SkyPilotBackend,
)


def make_world(pA, pB, n, rng):
    A, B = [], []
    for k in range(n):
        base = rng.random()
        A.append(Rollout("put cube in bin", key=k, meta={"p": pA, "y": int(base < pA)}))
        B.append(Rollout("put cube in bin", key=k, meta={"p": pB, "y": int(base < pB)}))
    return A, B


def main():
    rng = np.random.default_rng(0)
    A, B = make_world(0.75, 0.55, 1500, rng)
    vlm = StubVLM(rho=0.85)                       # swap for VLMJudge on real rollouts
    gold = HumanGold(lambda ra, rb: (ra.meta["y"], rb.meta["y"]), cost=60.0)

    print("=== 1. compare (cheap evaluator + cost-aware human auditing) ===")
    rep = compare_policies(A, B, vlm, gold, alpha=0.10, alternative="A>B",
                           rng=np.random.default_rng(1))
    print(rep.summary())

    print("\n=== 2. recalibrate the evaluator from collected human labels ===")
    recs = []
    for ra, rb in zip(A[:400], B[:400]):
        recs += [LabelRecord(f=vlm.score(ra), y=ra.meta["y"]),
                 LabelRecord(f=vlm.score(rb), y=rb.meta["y"])]
    calibrated = Recalibrate(vlm).update(recs)
    print(f"   fit isotonic map on {len(recs)} labels -> {type(calibrated).__name__}")

    print("\n=== 3. retrain handoff (dry-run: writes dataset + config) ===")
    train_recs = [LabelRecord(f=vlm.score(ra), y=ra.meta["y"],
                              frames=[f"ep{i}.png"], question="Is the cube in the bin?")
                  for i, ra in enumerate(A[:50])]
    cfg = TrainingConfig(workdir="out/train", output_dir="out/judge-lora")
    local = LlamaFactoryBackend(cfg, dry_run=True).update(train_recs)
    cloud = SkyPilotBackend(cfg, accelerator="A100:1", dry_run=True).update(train_recs)
    print(f"   local : {' '.join(local.command)}")
    print(f"   cloud : {' '.join(cloud.command)}")
    print(f"   dataset -> {local.dataset_path}")
    print("   (flip dry_run=False on a GPU/cloud machine to actually train)")


if __name__ == "__main__":
    main()
