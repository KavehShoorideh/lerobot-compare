"""
examples/robotics_qwen_lerobot.py

ANNOTATED TEMPLATE for a real machine (GPU or Apple Silicon). Not runnable in a
CPU-only sandbox: it needs Qwen2.5-VL weights and LeRobot data.

    pip install "transformers>=4.49" qwen-vl-utils accelerate lerobot torch pillow
    python3 examples/robotics_qwen_lerobot.py

Pipeline: load recorded rollouts of policy A and B -> wire a calibrated VLM judge
-> define your human gold labeler -> compare with an anytime-valid guarantee.
"""

import robotics_adapter as pr
from robotics_adapter import RubricVLM, InternalState, build_pairs, human_gold, compare_policies


def main():
    # 1) Load recorded rollouts (LeRobot dataset repo ids, or adapt load_episodes
    #    to your local recordings). Each episode -> one Rollout (frames + task).
    rollouts_A = list(pr.load_episodes("you/policyA_rollouts",
                                       camera_key="observation.images.top"))
    rollouts_B = list(pr.load_episodes("you/policyB_rollouts",
                                       camera_key="observation.images.top"))
    pairs = build_pairs(rollouts_A, rollouts_B)        # positional pairing by task order

    # 2) Calibrated VLM judge: a rubric of yes/no subgoal questions. Reads P("yes")
    #    from the token logits -> graded [0,1] score the calibration layer can use.
    #    combine: "mean" (partial credit) | "product" (strict) | "depth" (ordinal,
    #    encodes the open->place->close ordering as normalized progress depth).
    vlm = RubricVLM(
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        rubric=("Is the drawer open?",
                "Is the cube inside the drawer?",
                "Is the drawer closed?"),
        combine="depth",
        max_frames=8,
        cost=1.0,
    )

    # (optional) a near-free second route from policy internals:
    # internal = InternalState(fn=lambda ro: ro.meta["ood_ok"], cost=0.001)

    # 3) Your scarce, trustworthy label: a human confirming success on audited pairs.
    def human_label(pair):
        ya = int(input(f"[{pair.task}] policy A success? 1/0: "))
        yb = int(input(f"[{pair.task}] policy B success? 1/0: "))
        return (ya, yb)
    gold = human_gold(human_label, cost=60.0)          # ~60s of human time per audit

    # 4) Compare. Leans on the VLM, audits adaptively, stops anytime-valid at alpha.
    result = compare_policies(pairs, vlm, gold, alpha=0.10)
    print(result)

    # --- FIRST THING TO MEASURE ON REAL DATA -------------------------------------
    # The decisive-tilt / high-floor acquisition constants are tuned to synthetic
    # data. Verify the ~3x gold saving holds with a real VLM by comparing the
    # principled rule to uniform AT MATCHED budget (mirror test_F3_* in the suite),
    # and retune MaxEvidencePerCost(target_rate, pi_min, pi_max) if needed.


if __name__ == "__main__":
    main()
