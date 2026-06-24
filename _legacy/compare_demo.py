"""
examples/compare_demo.py  --  run:  python3 examples/compare_demo.py

baseline in action: an anytime-valid A/B success test you can stop early.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from lerobot_doctor import compare_success, Trial


def main():
    rng = np.random.default_rng(0)

    # --- unpaired: just two lists of 0/1 successes (A is the better policy) ---
    A = [int(rng.random() < 0.75) for _ in range(600)]
    B = [int(rng.random() < 0.55) for _ in range(600)]
    print("=== unpaired ===")
    print(compare_success(A, B, alpha=0.10).summary())

    # --- exact-matched: same task instances (seed/scene id) for A and B ------
    At, Bt = [], []
    for k in range(600):
        base = rng.random()                       # shared per-instance difficulty
        At.append(Trial(int(base < 0.75), key=k))
        Bt.append(Trial(int(base < 0.55), key=k))
    print("\n=== exact-matched (same instances) ===")
    rep = compare_success(At, Bt, alpha=0.10)
    print(rep.summary())

    # render the full chart dashboard (needs the [viz] extra: pip install matplotlib)
    try:
        from lerobot_doctor.viz import dashboard
        out = dashboard(rep, os.path.join(os.path.dirname(__file__), "sample_report.png"))
        print(f"\ncharts -> {out}")
    except ImportError:
        print("\n(install matplotlib for charts: pip install 'lerobot-compare[viz]')")


if __name__ == "__main__":
    main()
