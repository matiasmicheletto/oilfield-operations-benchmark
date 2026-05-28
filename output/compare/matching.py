"""
matching.py — Instance discovery and stem-based matching across solver output directories.
"""

import sys
import re
from pathlib import Path


def extract_stem(filename: str) -> str:
    """Return the <N>_<B>_<run> stem from output filenames.

    Examples:
        model_100_2_1.sol  →  100_2_1
        greedy_100_2_1.txt →  100_2_1
        model_100_2_1.txt  →  100_2_1   (SCIP output)
    """
    name = Path(filename).stem
    for prefix in ("model_", "greedy_", "hs_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def extract_join_stem(stem: str) -> str:
    """Normalize stems so heuristic variants can match model/scip outputs.

    Examples:
        10_2_1_1_priority_cost -> 10_2_1_1
        10_2_1_1_loss          -> 10_2_1_1
        10_2_1_1_route         -> 10_2_1_1
    """
    m = re.fullmatch(r"(\d+_\d+_\d+_\d+)_(priority_cost|loss|route)", stem)
    if m:
        return m.group(1)
    return stem


def match_instances(
    greedy_dir: Path,
    cplex_dir: Path | None = None,
    scip_dir: Path | None = None,
) -> list[tuple]:
    """Return a list of (stem, cplex_path_or_None, greedy_path, scip_path_or_None).

    greedy is the required base — every greedy file produces one row.
    CPLEX and SCIP files are joined by stem when available; their path is None otherwise.
    Stems present in CPLEX/SCIP but absent from greedy are reported and skipped.
    """
    # Heuristic directories can contain auxiliary route dumps (routes_*.txt);
    # only method solution files should be considered for comparison.
    greedy_items = []
    for f in greedy_dir.glob("*.txt"):
        if f.name.startswith("routes_"):
            continue
        full_stem = extract_stem(f.name)
        join_stem = extract_join_stem(full_stem)
        greedy_items.append((full_stem, join_stem, f))

    cplex_files = {
        extract_join_stem(extract_stem(f.name)): f for f in cplex_dir.glob("*.sol")
    } if cplex_dir else {}
    scip_files = {
        extract_join_stem(extract_stem(f.name)): f for f in scip_dir.glob("*.txt")
    } if scip_dir else {}

    greedy_join_stems = {join_stem for _, join_stem, _ in greedy_items}

    only_cplex = cplex_files.keys() - greedy_join_stems
    only_scip  = scip_files.keys()  - greedy_join_stems

    if only_cplex:
        print(f"[info] CPLEX-only instances (no greedy match): {sorted(only_cplex)}")
    if only_scip:
        print(f"[info] SCIP-only instances (no greedy match): {sorted(only_scip)}")

    greedy_items.sort(key=lambda x: x[0])
    return [
        (full_stem, cplex_files.get(join_stem), greedy_path, scip_files.get(join_stem))
        for full_stem, join_stem, greedy_path in greedy_items
    ]
