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

"""
cli.py -- `lerobot-compare compare A.txt B.txt [--method ...]`

The standardised eval CLI: read per-rollout success labels (0/1, one per line; a
CSV with a 'success'-like column also works) for two policies, run the selected
test METHOD, and print the scorecard.

Exit status: 0 if a decision was reached (significant), 2 otherwise -- handy for
shell pipelines and CI gates.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List

from .compare import available_methods, compare_success
from .report import DEFAULT_METHOD


def _read_successes(path: str) -> List[int]:
    """Read 0/1 success labels: one integer per line, or the last column of a CSV
    (a non-numeric first row is treated as a header and skipped)."""
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
    c = sub.add_parser("compare", help="A/B success test (anytime-valid or fixed-horizon)")
    c.add_argument("file_A", help="per-rollout 0/1 successes for policy A")
    c.add_argument("file_B", help="per-rollout 0/1 successes for policy B")
    c.add_argument(
        "--method", default=DEFAULT_METHOD, choices=available_methods(),
        help="test method (default: %(default)s)")
    c.add_argument("--alpha", type=float, default=0.10,
                   help="significance level (default: %(default)s)")
    c.add_argument("--alternative", default="two-sided",
                   choices=["two-sided", "A>B", "B>A"])
    c.add_argument("--target-power", type=float, default=0.80,
                   help="power level the fixed-horizon MDE targets (default: %(default)s)")
    c.add_argument("--json", metavar="OUT", help="write full report JSON to OUT")
    c.add_argument("--charts", metavar="PNG", help="render the dashboard to PNG")
    args = p.parse_args(argv)

    if args.cmd == "compare":
        A, B = _read_successes(args.file_A), _read_successes(args.file_B)
        rep = compare_success(
            A, B, method=args.method, alpha=args.alpha,
            alternative=args.alternative, target_power=args.target_power)
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
