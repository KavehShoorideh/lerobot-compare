"""
examples/offline_compare_stub.py

Runnable now, no GPU: compare two policies offline through the robotics adapter
using a torch-free StubVLM. Shows the decision and how few gold labels it spent
relative to auditing every rollout.

    python3 examples/offline_compare_stub.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import robotics_adapter as pr
from robotics_adapter import Rollout, Pair, StubVLM


def make_pairs(n, pA, pB, rng):
    pairs = []
    for _ in range(n):
        a = Rollout("put cube in bin", meta={"p": pA, "y": int(rng.random() < pA)})
        b = Rollout("put cube in bin", meta={"p": pB, "y": int(rng.random() < pB)})
        pairs.append(Pair("put cube in bin", a, b))
    return pairs


def main():
    rng = np.random.default_rng(0)
    pairs = make_pairs(1500, pA=0.75, pB=0.55, rng=rng)         # A is better

    evaluator = StubVLM(rho=0.8, cost=1.0)        # swap for RubricVLM on real rollouts
    gold = pr.human_gold(lambda p: (p.A.meta["y"], p.B.meta["y"]), cost=60.0)

    r = pr.compare_policies(pairs, evaluator, gold, alpha=0.10,
                            rng=np.random.default_rng(1))

    print(f"decision      : {'A > B' if r.decided else 'no call'}  (wealth {r.wealth:.1f})")
    print(f"rollouts seen : {r.n_seen}")
    print(f"gold labels   : {r.n_audited}   (vs {r.n_seen} if you audited everything)")
    print(f"saved         : {100 * (1 - r.n_audited / r.n_seen):.0f}% of trustworthy labels")


if __name__ == "__main__":
    main()
