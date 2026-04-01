"""
reporting.py — Console table and CSV output for solution comparison results.

Each row dict is expected to have keys:
    stem
    cplex_distance, greedy_distance, scip_distance  (float | None)
    cplex_cost,     greedy_cost,     scip_cost      (float | None)
    cplex_loss,     greedy_loss,     scip_loss      (float | None)
    cplex_n,        greedy_n,        scip_n         (int | None)
"""

import csv
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(val, fmt: str = ".2f") -> str:
    """Format a numeric value or return 'N/A' for None."""
    return f"{val:{fmt}}" if val is not None else "N/A"


def _gap(reference: float | None, other: float | None) -> float | None:
    """Relative gap: (other - reference) / |reference| * 100 %.
    Positive means *other* is worse (larger distance / cost / loss).
    Returns None when either value is missing or reference is zero."""
    if reference is None or other is None or reference == 0:
        return None
    return (other - reference) / abs(reference) * 100.0


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

# Column widths shared by both the with-SCIP and without-SCIP layouts.
_COL = {
    "instance":   12,
    "cplex_d":     9,
    "greedy_d":    9,
    "gap_g":       8,
    "scip_d":      9,
    "gap_s":       8,
    "cplex_cost":  9,
    "greedy_cost": 9,
    "scip_cost":   9,
    "cplex_loss":  9,
    "greedy_loss": 9,
    "scip_loss":   9,
    "cplex_n":     7,
    "greedy_n":    7,
    "scip_n":      7,
}


def _header_with_scip() -> str:
    c = _COL
    return (
        f"{'Instance':<{c['instance']}} "
        f"{'CPLEX dist':>{c['cplex_d']}} "
        f"{'Grdy dist':>{c['greedy_d']}} "
        f"{'Gap g%':>{c['gap_g']}} "
        f"{'SCIP dist':>{c['scip_d']}} "
        f"{'Gap s%':>{c['gap_s']}} "
        f"{'CPLEX cost':>{c['cplex_cost']}} "
        f"{'Grdy cost':>{c['greedy_cost']}} "
        f"{'SCIP cost':>{c['scip_cost']}} "
        f"{'CPLEX loss':>{c['cplex_loss']}} "
        f"{'Grdy loss':>{c['greedy_loss']}} "
        f"{'SCIP loss':>{c['scip_loss']}} "
        f"{'CPLEX n':>{c['cplex_n']}} "
        f"{'Grdy n':>{c['greedy_n']}} "
        f"{'SCIP n':>{c['scip_n']}}"
    )


def _header_without_scip() -> str:
    c = _COL
    return (
        f"{'Instance':<{c['instance']}} "
        f"{'CPLEX dist':>{c['cplex_d']}} "
        f"{'Grdy dist':>{c['greedy_d']}} "
        f"{'Gap dist%':>{c['gap_g']}} "
        f"{'CPLEX cost':>{c['cplex_cost']}} "
        f"{'Grdy cost':>{c['greedy_cost']}} "
        f"{'CPLEX loss':>{c['cplex_loss']}} "
        f"{'Grdy loss':>{c['greedy_loss']}} "
        f"{'CPLEX n':>{c['cplex_n']}} "
        f"{'Grdy n':>{c['greedy_n']}}"
    )


def _row_with_scip(r: dict) -> str:
    c = _COL
    gap_g     = _gap(r["cplex_distance"], r["greedy_distance"])
    gap_s     = _gap(r["cplex_distance"], r.get("scip_distance"))
    gap_g_str = f"{gap_g:+.1f}" if gap_g is not None else "N/A"
    gap_s_str = f"{gap_s:+.1f}" if gap_s is not None else "N/A"
    scip_n    = str(r["scip_n"]) if r.get("scip_n") is not None else "N/A"
    return (
        f"{r['stem']:<{c['instance']}} "
        f"{_fmt(r['cplex_distance']):>{c['cplex_d']}} "
        f"{_fmt(r['greedy_distance']):>{c['greedy_d']}} "
        f"{gap_g_str:>{c['gap_g']}} "
        f"{_fmt(r.get('scip_distance')):>{c['scip_d']}} "
        f"{gap_s_str:>{c['gap_s']}} "
        f"{_fmt(r['cplex_cost']):>{c['cplex_cost']}} "
        f"{_fmt(r['greedy_cost']):>{c['greedy_cost']}} "
        f"{_fmt(r.get('scip_cost')):>{c['scip_cost']}} "
        f"{_fmt(r['cplex_loss']):>{c['cplex_loss']}} "
        f"{_fmt(r['greedy_loss']):>{c['greedy_loss']}} "
        f"{_fmt(r.get('scip_loss')):>{c['scip_loss']}} "
        f"{r['cplex_n']:>{c['cplex_n']}} "
        f"{r['greedy_n']:>{c['greedy_n']}} "
        f"{scip_n:>{c['scip_n']}}"
    )


def _row_without_scip(r: dict) -> str:
    c = _COL
    gap_g     = _gap(r["cplex_distance"], r["greedy_distance"])
    gap_g_str = f"{gap_g:+.1f}" if gap_g is not None else "N/A"
    return (
        f"{r['stem']:<{c['instance']}} "
        f"{_fmt(r['cplex_distance']):>{c['cplex_d']}} "
        f"{_fmt(r['greedy_distance']):>{c['greedy_d']}} "
        f"{gap_g_str:>{c['gap_g']}} "
        f"{_fmt(r['cplex_cost']):>{c['cplex_cost']}} "
        f"{_fmt(r['greedy_cost']):>{c['greedy_cost']}} "
        f"{_fmt(r['cplex_loss']):>{c['cplex_loss']}} "
        f"{_fmt(r['greedy_loss']):>{c['greedy_loss']}} "
        f"{r['cplex_n']:>{c['cplex_n']}} "
        f"{r['greedy_n']:>{c['greedy_n']}}"
    )


def _avg(key: str, rows: list[dict]) -> float | None:
    vals = [r.get(key) for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def _avg_gap(ref_key: str, other_key: str, rows: list[dict]) -> float | None:
    gaps = [_gap(r.get(ref_key), r.get(other_key)) for r in rows]
    valid = [g for g in gaps if g is not None]
    return sum(valid) / len(valid) if valid else None


def _averages_with_scip(rows: list[dict]) -> str:
    c = _COL
    avg_gap_g = _avg_gap("cplex_distance", "greedy_distance", rows)
    avg_gap_s = _avg_gap("cplex_distance", "scip_distance",   rows)
    return (
        f"{'AVERAGE':<{c['instance']}} "
        f"{_fmt(_avg('cplex_distance',  rows)):>{c['cplex_d']}} "
        f"{_fmt(_avg('greedy_distance', rows)):>{c['greedy_d']}} "
        f"{(_fmt(avg_gap_g, '.1f') + '%') if avg_gap_g is not None else 'N/A':>{c['gap_g']}} "
        f"{_fmt(_avg('scip_distance',   rows)):>{c['scip_d']}} "
        f"{(_fmt(avg_gap_s, '.1f') + '%') if avg_gap_s is not None else 'N/A':>{c['gap_s']}} "
        f"{_fmt(_avg('cplex_cost',  rows)):>{c['cplex_cost']}} "
        f"{_fmt(_avg('greedy_cost', rows)):>{c['greedy_cost']}} "
        f"{_fmt(_avg('scip_cost',   rows)):>{c['scip_cost']}} "
        f"{_fmt(_avg('cplex_loss',  rows)):>{c['cplex_loss']}} "
        f"{_fmt(_avg('greedy_loss', rows)):>{c['greedy_loss']}} "
        f"{_fmt(_avg('scip_loss',   rows)):>{c['scip_loss']}} "
        f"{'':>{c['cplex_n']}} "
        f"{'':>{c['greedy_n']}} "
        f"{'':>{c['scip_n']}}"
    )


def _averages_without_scip(rows: list[dict]) -> str:
    c = _COL
    avg_gap_g = _avg_gap("cplex_distance", "greedy_distance", rows)
    return (
        f"{'AVERAGE':<{c['instance']}} "
        f"{_fmt(_avg('cplex_distance',  rows)):>{c['cplex_d']}} "
        f"{_fmt(_avg('greedy_distance', rows)):>{c['greedy_d']}} "
        f"{(_fmt(avg_gap_g, '.1f') + '%') if avg_gap_g is not None else 'N/A':>{c['gap_g']}} "
        f"{_fmt(_avg('cplex_cost',  rows)):>{c['cplex_cost']}} "
        f"{_fmt(_avg('greedy_cost', rows)):>{c['greedy_cost']}} "
        f"{_fmt(_avg('cplex_loss',  rows)):>{c['cplex_loss']}} "
        f"{_fmt(_avg('greedy_loss', rows)):>{c['greedy_loss']}} "
        f"{'':>{c['cplex_n']}} "
        f"{'':>{c['greedy_n']}}"
    )


def print_table(rows: list[dict]) -> None:
    """Print a formatted comparison table to stdout."""
    has_scip = any(r.get("scip_distance") is not None for r in rows)

    header   = _header_with_scip()   if has_scip else _header_without_scip()
    sep      = "-" * len(header)
    row_fn   = _row_with_scip        if has_scip else _row_without_scip
    avg_line = _averages_with_scip   if has_scip else _averages_without_scip

    print(sep)
    print(header)
    print(sep)
    for r in rows:
        print(row_fn(r))
    print(sep)
    print(avg_line(rows))
    print(sep)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def write_csv(rows: list[dict], path: Path) -> None:
    """Write comparison data to a CSV file."""
    fieldnames = [
        "instance",
        "cplex_distance", "greedy_distance", "greedy_distance_gap_pct",
        "scip_distance",  "scip_distance_gap_pct",
        "cplex_cost",     "greedy_cost",     "scip_cost",
        "cplex_loss",     "greedy_loss",     "scip_loss",
        "cplex_n_wells",  "greedy_n_wells",  "scip_n_wells",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            gap_g = _gap(r["cplex_distance"], r["greedy_distance"])
            gap_s = _gap(r["cplex_distance"], r.get("scip_distance"))
            writer.writerow({
                "instance":                r["stem"],
                "cplex_distance":          r["cplex_distance"],
                "greedy_distance":         r["greedy_distance"],
                "greedy_distance_gap_pct": f"{gap_g:.4f}" if gap_g is not None else "",
                "scip_distance":           r.get("scip_distance") or "",
                "scip_distance_gap_pct":   f"{gap_s:.4f}" if gap_s is not None else "",
                "cplex_cost":              r["cplex_cost"],
                "greedy_cost":             r["greedy_cost"],
                "scip_cost":               r.get("scip_cost") or "",
                "cplex_loss":              r["cplex_loss"],
                "greedy_loss":             r["greedy_loss"],
                "scip_loss":               r.get("scip_loss") or "",
                "cplex_n_wells":           r["cplex_n"],
                "greedy_n_wells":          r["greedy_n"],
                "scip_n_wells":            r.get("scip_n") or "",
            })
    print(f"\n[info] CSV written to: {path}")
