#!/usr/bin/env python3
"""
compare_solutions.py  —  Compare CPLEX ILP solutions against greedy heuristic solutions.

Usage:
    python compare_solutions.py [--cplex DIR] [--greedy DIR] [--csv FILE]

Defaults:
    --cplex   output/cplex/
    --greedy  output/greedy/

Instance matching:
    CPLEX files  : output/cplex/model_<stem>.sol
    Greedy files : output/greedy/greedy_<stem>.txt
    Stems are matched by the <N_wells>_<N_batteries>_<run> portion of the name.

CPLEX .sol format (XML written by CPLEX):
    <CPLEXSolution>
      <header objectiveValue="..." solutionStatusString="..." .../>
      <variables>
        <variable name="distance"  value="..."/>
        <variable name="cost"      value="..."/>
        <variable name="loss"      value="..."/>
        <variable name="z$<id>$"   value="1"/>   ← selected well (Zimpl LP encoding)
        ...
      </variables>
    </CPLEXSolution>

Greedy .txt format (plain text written by solve_main.cpp):
    === Solution ===
    Selected wells (N): w1 w2 ...
    Per-well regimes:
      Well <id>: current_regime=<r> -> new_regime=<nr>
    Per-crew routes (K crews):
      Crew 1: 0 → w3 → ... → 0   (distance: X)
    Total distance: X
    Total cost: X
    Total loss (actual): X
"""

import argparse
import re
import sys
from pathlib import Path
from xml.etree import ElementTree


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_cplex_sol(path: Path) -> dict | None:
    """Parse a CPLEX XML .sol file.  Returns None if the file cannot be parsed
    or does not represent a feasible solution."""
    try:
        tree = ElementTree.parse(path)
    except ElementTree.ParseError as exc:
        print(f"  [warn] Cannot parse XML in '{path.name}': {exc}", file=sys.stderr)
        return None

    root = tree.getroot()

    # Status check
    header = root.find("header")
    status = (header.attrib.get("solutionStatusString", "") if header is not None else "")
    obj_str = (header.attrib.get("objectiveValue", "") if header is not None else "")

    if not obj_str:
        print(f"  [warn] '{path.name}' has no objectiveValue – skipping.", file=sys.stderr)
        return None

    variables = root.find("variables")
    if variables is None:
        print(f"  [warn] '{path.name}' has no <variables> section.", file=sys.stderr)
        return None

    result = {
        "status": status,
        "distance": None,
        "cost": None,
        "loss": None,
        "selected_wells": [],
    }

    for var in variables.findall("variable"):
        name = var.attrib.get("name", "")
        raw  = var.attrib.get("value", "0")
        try:
            val = float(raw)
        except ValueError:
            continue

        # Scalar objectives defined as variables in the Zimpl model
        if name == "distance":
            result["distance"] = val
        elif name == "cost":
            result["cost"] = val
        elif name == "loss":
            result["loss"] = val
        # Binary selection variable z[i] → encoded by Zimpl as "z$<i>$" in LP
        elif re.fullmatch(r"z\$\d+\$", name) and val > 0.5:
            well_id = int(name.split("$")[1])
            result["selected_wells"].append(well_id)

    # Fall back to header objective if the distance variable is absent
    if result["distance"] is None:
        try:
            result["distance"] = float(obj_str)
        except ValueError:
            pass

    result["n_wells"] = len(result["selected_wells"])
    return result


def parse_greedy_txt(path: Path) -> dict | None:
    """Parse a plain-text greedy solution file."""
    try:
        text = path.read_text()
    except OSError as exc:
        print(f"  [warn] Cannot read '{path.name}': {exc}", file=sys.stderr)
        return None

    result = {
        "distance": None,
        "cost": None,
        "loss": None,
        "selected_wells": [],
        "n_wells": 0,
    }

    # Selected wells (N): w1 w2 ...
    m = re.search(r"Selected wells \((\d+)\):\s*([\d ]+)", text)
    if m:
        result["n_wells"] = int(m.group(1))
        result["selected_wells"] = [int(x) for x in m.group(2).split()]

    # Total distance: X
    m = re.search(r"Total distance:\s*([\d.eE+\-]+)", text)
    if m:
        result["distance"] = float(m.group(1))

    # Total cost: X
    m = re.search(r"Total cost:\s*([\d.eE+\-]+)", text)
    if m:
        result["cost"] = float(m.group(1))

    # Total loss (actual): X
    m = re.search(r"Total loss \(actual\):\s*([\d.eE+\-]+)", text)
    if m:
        result["loss"] = float(m.group(1))

    if result["distance"] is None:
        print(f"  [warn] '{path.name}' missing 'Total distance' – skipping.", file=sys.stderr)
        return None

    return result


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def extract_stem(filename: str) -> str:
    """Return the <N>_<B>_<run> stem from filenames like:
        model_100_2_1.sol  →  100_2_1
        greedy_100_2_1.txt  →  100_2_1
    """
    name = Path(filename).stem          # drop extension
    # Strip known prefixes
    for prefix in ("model_", "greedy_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def match_instances(cplex_dir: Path, greedy_dir: Path) -> list[tuple[str, Path, Path]]:
    """Return list of (stem, cplex_path, greedy_path) for matched instances."""
    cplex_files  = {extract_stem(f.name): f for f in cplex_dir.glob("*.sol")}
    greedy_files = {extract_stem(f.name): f for f in greedy_dir.glob("*.txt")}

    stems = sorted(cplex_files.keys() & greedy_files.keys())

    only_cplex  = cplex_files.keys()  - greedy_files.keys()
    only_greedy = greedy_files.keys() - cplex_files.keys()

    if only_cplex:
        print(f"[info] CPLEX-only instances (no greedy match): {sorted(only_cplex)}")
    if only_greedy:
        print(f"[info] Greedy-only instances (no CPLEX match): {sorted(only_greedy)}")

    return [(s, cplex_files[s], greedy_files[s]) for s in stems]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(val, fmt=".2f"):
    return f"{val:{fmt}}" if val is not None else "N/A"


def _gap(cplex_val, greedy_val):
    """Relative gap: (greedy - cplex) / cplex * 100 %  (positive = greedy is worse)."""
    if cplex_val is None or greedy_val is None or cplex_val == 0:
        return None
    return (greedy_val - cplex_val) / abs(cplex_val) * 100.0


def print_table(rows: list[dict]) -> None:
    col_w = {
        "instance":       12,
        "cplex_d":         9,
        "greedy_d":        9,
        "gap_d":           8,
        "cplex_cost":      9,
        "greedy_cost":     9,
        "cplex_loss":      9,
        "greedy_loss":     9,
        "cplex_n":         7,
        "greedy_n":        7,
    }

    header = (
        f"{'Instance':<{col_w['instance']}} "
        f"{'CPLEX dist':>{col_w['cplex_d']}} "
        f"{'Grdy dist':>{col_w['greedy_d']}} "
        f"{'Gap dist%':>{col_w['gap_d']}} "
        f"{'CPLEX cost':>{col_w['cplex_cost']}} "
        f"{'Grdy cost':>{col_w['greedy_cost']}} "
        f"{'CPLEX loss':>{col_w['cplex_loss']}} "
        f"{'Grdy loss':>{col_w['greedy_loss']}} "
        f"{'CPLEX n':>{col_w['cplex_n']}} "
        f"{'Grdy n':>{col_w['greedy_n']}}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for r in rows:
        gap_d = _gap(r["cplex_distance"], r["greedy_distance"])
        gap_str = f"{gap_d:+.1f}" if gap_d is not None else "N/A"
        print(
            f"{r['stem']:<{col_w['instance']}} "
            f"{_fmt(r['cplex_distance']):>{col_w['cplex_d']}} "
            f"{_fmt(r['greedy_distance']):>{col_w['greedy_d']}} "
            f"{gap_str:>{col_w['gap_d']}} "
            f"{_fmt(r['cplex_cost']):>{col_w['cplex_cost']}} "
            f"{_fmt(r['greedy_cost']):>{col_w['greedy_cost']}} "
            f"{_fmt(r['cplex_loss']):>{col_w['cplex_loss']}} "
            f"{_fmt(r['greedy_loss']):>{col_w['greedy_loss']}} "
            f"{r['cplex_n']:>{col_w['cplex_n']}} "
            f"{r['greedy_n']:>{col_w['greedy_n']}}"
        )

    print(sep)

    # Summary averages (omit N/A)
    def avg(key, rows):
        vals = [r[key] for r in rows if r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    all_gaps = [_gap(r["cplex_distance"], r["greedy_distance"]) for r in rows]
    valid_gaps = [g for g in all_gaps if g is not None]
    avg_gap = sum(valid_gaps) / len(valid_gaps) if valid_gaps else None

    print(
        f"{'AVERAGE':<{col_w['instance']}} "
        f"{_fmt(avg('cplex_distance', rows)):>{col_w['cplex_d']}} "
        f"{_fmt(avg('greedy_distance', rows)):>{col_w['greedy_d']}} "
        f"{(_fmt(avg_gap, '.1f') + '%') if avg_gap is not None else 'N/A':>{col_w['gap_d']}} "
        f"{_fmt(avg('cplex_cost', rows)):>{col_w['cplex_cost']}} "
        f"{_fmt(avg('greedy_cost', rows)):>{col_w['greedy_cost']}} "
        f"{_fmt(avg('cplex_loss', rows)):>{col_w['cplex_loss']}} "
        f"{_fmt(avg('greedy_loss', rows)):>{col_w['greedy_loss']}} "
        f"{'':>{col_w['cplex_n']}} "
        f"{'':>{col_w['greedy_n']}}"
    )
    print(sep)


def write_csv(rows: list[dict], path: Path) -> None:
    import csv
    fieldnames = [
        "instance",
        "cplex_distance", "greedy_distance", "distance_gap_pct",
        "cplex_cost",     "greedy_cost",
        "cplex_loss",     "greedy_loss",
        "cplex_n_wells",  "greedy_n_wells",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            gap = _gap(r["cplex_distance"], r["greedy_distance"])
            writer.writerow({
                "instance":          r["stem"],
                "cplex_distance":    r["cplex_distance"],
                "greedy_distance":   r["greedy_distance"],
                "distance_gap_pct":  f"{gap:.4f}" if gap is not None else "",
                "cplex_cost":        r["cplex_cost"],
                "greedy_cost":       r["greedy_cost"],
                "cplex_loss":        r["cplex_loss"],
                "greedy_loss":       r["greedy_loss"],
                "cplex_n_wells":     r["cplex_n"],
                "greedy_n_wells":    r["greedy_n"],
            })
    print(f"\n[info] CSV written to: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    script_dir = Path(__file__).parent
    default_cplex  = script_dir / "output" / "cplex"
    default_greedy = script_dir / "output" / "greedy"

    parser = argparse.ArgumentParser(
        description="Compare CPLEX ILP solutions against greedy heuristic solutions."
    )
    parser.add_argument("--cplex",  type=Path, default=default_cplex,
                        metavar="DIR", help="Directory containing CPLEX .sol files")
    parser.add_argument("--greedy", type=Path, default=default_greedy,
                        metavar="DIR", help="Directory containing greedy .txt files")
    parser.add_argument("--csv",    type=Path, default=None,
                        metavar="FILE", help="Optional path to write a CSV summary")
    args = parser.parse_args()

    for d, label in [(args.cplex, "CPLEX"), (args.greedy, "greedy")]:
        if not d.exists():
            print(f"Error: {label} directory not found: {d}", file=sys.stderr)
            return 1

    matched = match_instances(args.cplex, args.greedy)
    if not matched:
        print("No matching instance pairs found.", file=sys.stderr)
        return 1

    print(f"\nComparing {len(matched)} instance(s)  "
          f"[cplex: {args.cplex}]  [greedy: {args.greedy}]\n")

    rows = []
    for stem, cplex_path, greedy_path in matched:
        print(f"  Parsing {stem} ...")
        c = parse_cplex_sol(cplex_path)
        g = parse_greedy_txt(greedy_path)
        if c is None or g is None:
            continue
        rows.append({
            "stem":             stem,
            "cplex_distance":   c["distance"],
            "greedy_distance":  g["distance"],
            "cplex_cost":       c["cost"],
            "greedy_cost":      g["cost"],
            "cplex_loss":       c["loss"],
            "greedy_loss":      g["loss"],
            "cplex_n":          c["n_wells"],
            "greedy_n":         g["n_wells"],
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
