"""
matching.py — Instance discovery and stem-based matching across solver output directories.
"""

import sys
from pathlib import Path


def extract_stem(filename: str) -> str:
    """Return the <N>_<B>_<run> stem from output filenames.

    Examples:
        model_100_2_1.sol  →  100_2_1
        greedy_100_2_1.txt →  100_2_1
        model_100_2_1.txt  →  100_2_1   (SCIP output)
    """
    name = Path(filename).stem
    for prefix in ("model_", "greedy_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


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
    greedy_files = {extract_stem(f.name): f for f in greedy_dir.glob("*.txt")}
    cplex_files  = {extract_stem(f.name): f for f in cplex_dir.glob("*.sol")} if cplex_dir else {}
    scip_files   = {extract_stem(f.name): f for f in scip_dir.glob("*.txt")}  if scip_dir  else {}

    stems = sorted(greedy_files.keys())

    only_cplex = cplex_files.keys() - greedy_files.keys()
    only_scip  = scip_files.keys()  - greedy_files.keys()

    if only_cplex:
        print(f"[info] CPLEX-only instances (no greedy match): {sorted(only_cplex)}")
    if only_scip:
        print(f"[info] SCIP-only instances (no greedy match): {sorted(only_scip)}")

    return [(s, cplex_files.get(s), greedy_files[s], scip_files.get(s)) for s in stems]
