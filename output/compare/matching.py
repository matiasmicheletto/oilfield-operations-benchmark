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
    cplex_dir: Path,
    greedy_dir: Path,
    scip_dir: Path | None = None,
) -> list[tuple]:
    """Return a list of (stem, cplex_path, greedy_path, scip_path) tuples.

    Only stems present in both cplex_dir and greedy_dir are returned.
    scip_path is None when scip_dir is not provided or no matching file exists.
    Unmatched files are reported as informational warnings.
    """
    cplex_files  = {extract_stem(f.name): f for f in cplex_dir.glob("*.sol")}
    greedy_files = {extract_stem(f.name): f for f in greedy_dir.glob("*.txt")}
    scip_files   = {extract_stem(f.name): f for f in scip_dir.glob("*.txt")} if scip_dir else {}

    stems = sorted(cplex_files.keys() & greedy_files.keys())

    only_cplex  = cplex_files.keys()  - greedy_files.keys()
    only_greedy = greedy_files.keys() - cplex_files.keys()

    if only_cplex:
        print(f"[info] CPLEX-only instances (no greedy match): {sorted(only_cplex)}")
    if only_greedy:
        print(f"[info] Greedy-only instances (no CPLEX match): {sorted(only_greedy)}")

    return [(s, cplex_files[s], greedy_files[s], scip_files.get(s)) for s in stems]
