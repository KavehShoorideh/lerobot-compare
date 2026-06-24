"""
cli.py -- `lerobot-compare compare A.txt B.txt`

The standardized eval CLI the community keeps asking for. Reads per-rollout
success labels (0/1, one per line; CSV with a 'success' column also works) for
two policies and prints an anytime-valid verdict.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List

from .compare import compare_success


def _read_successes(path: str) -> List[int]:
    vals: List[int] = []
    with open(path) as f:
        first = True
        for line in f:
            s = line.strip()
            if not s:
                continue
            if "," in s:                              # CSV: take a 'success'-like col
                parts = [p.strip() for p in s.split(",")]
                if first and not parts[0].lstrip("-").isdigit():
                    first = False
                    continue                          # header row
                s = parts[-1]
            first = False
            vals.append(int(float(s)))
    return vals


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="lerobot-compare")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare", help="anytime-valid A/B success test")
    c.add_argument("file_A", help="per-rollout 0/1 successes for policy A")
    c.add_argument("file_B", help="per-rollout 0/1 successes for policy B")
    c.add_argument("--alpha", type=float, default=0.10)
    c.add_argument("--alternative", default="two-sided",
                   choices=["two-sided", "A>B", "B>A"])
    c.add_argument("--json", metavar="OUT", help="write full report JSON to OUT")
    c.add_argument("--charts", metavar="PNG", help="render the dashboard to PNG")
    args = p.parse_args(argv)

    if args.cmd == "compare":
        A, B = _read_successes(args.file_A), _read_successes(args.file_B)
        rep = compare_success(A, B, alpha=args.alpha, alternative=args.alternative)
        print(rep.summary())
        if args.json:
            with open(args.json, "w") as f:
                json.dump(rep.to_dict(), f, indent=2)
            print(f"\nfull report -> {args.json}")
        if args.charts:
            from .viz import dashboard
            dashboard(rep, args.charts)
            print(f"charts -> {args.charts}")
        return 0 if rep.decided else 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
