#!/usr/bin/env python3
"""
compare_solutions.py — Compare CPLEX ILP, SCIP ILP, and greedy heuristic solutions.

Usage:
    python compare_solutions.py [--cplex DIR] [--greedy DIR] [--scip DIR] [--csv FILE]

Defaults:
    --cplex   cplex/      (relative to this script, i.e. output/cplex/)
    --greedy  greedy/     (relative to this script, i.e. output/greedy/)
    --scip    scip/       (optional; omitted or missing directory hides SCIP columns)

Parsing and reporting logic lives in the compare/ sub-package:
    compare/parsers.py   — parse_cplex_sol, parse_greedy_txt, parse_scip_txt
    compare/matching.py  — extract_stem, match_instances
    compare/reporting.py — print_table, write_csv
"""

import argparse
import sys
from pathlib import Path

from compare.parsers   import parse_cplex_sol, parse_greedy_txt, parse_scip_txt
from compare.matching  import match_instances
from compare.reporting import print_table, write_csv


def main() -> int:
    script_dir = Path(__file__).parent
    default_cplex  = script_dir / "cplex"
    default_greedy = script_dir / "greedy"
    default_scip   = script_dir / "scip"

    parser = argparse.ArgumentParser(
        description="Compare CPLEX ILP, SCIP ILP, and greedy heuristic solutions."
    )
    parser.add_argument("--cplex",  type=Path, default=default_cplex,
                        metavar="DIR", help="Directory containing CPLEX .sol files")
    parser.add_argument("--greedy", type=Path, default=default_greedy,
                        metavar="DIR", help="Directory containing greedy .txt files")
    parser.add_argument("--scip",   type=Path, default=default_scip,
                        metavar="DIR", help="Directory containing SCIP .txt files (skipped if missing)")
    parser.add_argument("--csv",    type=Path, default=None,
                        metavar="FILE", help="Optional path to write a CSV summary")
    args = parser.parse_args()

    for d, label in [(args.cplex, "CPLEX"), (args.greedy, "greedy")]:
        if not d.exists():
            print(f"Error: {label} directory not found: {d}", file=sys.stderr)
            return 1

    scip_dir: Path | None = args.scip if args.scip.exists() else None
    if not scip_dir:
        msg = f"[warn] SCIP directory not found: {args.scip} — SCIP columns will be omitted."
        print(msg, file=sys.stderr)

    matched = match_instances(args.cplex, args.greedy, scip_dir)
    if not matched:
        print("No matching instance pairs found.", file=sys.stderr)
        return 1

    info = f"[cplex: {args.cplex}]  [greedy: {args.greedy}]"
    if scip_dir:
        info += f"  [scip: {scip_dir}]"
    print(f"\nComparing {len(matched)} instance(s)  {info}\n")

    rows = []
    for stem, cplex_path, greedy_path, scip_path in matched:
        print(f"  Parsing {stem} ...")
        c = parse_cplex_sol(cplex_path)
        g = parse_greedy_txt(greedy_path)
        if c is None or g is None:
            continue
        s = parse_scip_txt(scip_path) if scip_path else None
        rows.append({
            "stem":             stem,
            "cplex_distance":   c["distance"],
            "greedy_distance":  g["distance"],
            "scip_distance":    s["distance"] if s else None,
            "cplex_cost":       c["cost"],
            "greedy_cost":      g["cost"],
            "scip_cost":        s["cost"] if s else None,
            "cplex_loss":       c["loss"],
            "greedy_loss":      g["loss"],
            "scip_loss":        s["loss"] if s else None,
            "cplex_n":          c["n_wells"],
            "greedy_n":         g["n_wells"],
            "scip_n":           s["n_wells"] if s else None,
        })

    if not rows:
        print("No valid solution pairs to compare.", file=sys.stderr)
        return 1

    print()
    print_table(rows)

    if args.csv:
        write_csv(rows, args.csv)

    return 0


if __name__ == "__main__":
    sys.exit(main())
