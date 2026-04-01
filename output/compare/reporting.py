"""
reporting.py — Console table and CSV output for solution comparison results.

Each row dict is expected to have keys:
    stem
    cplex_distance, greedy_distance, scip_distance  (float | None)
    cplex_cost,     greedy_cost,     scip_cost      (float | None)
    cplex_loss,     greedy_loss,     scip_loss      (float | None)
    cplex_n,        greedy_n,        scip_n         (int | None)

CPLEX and SCIP values are None when those solvers were not run.
print_table() automatically selects one of four layouts based on which
solvers are present in the data:
    - CPLEX + greedy + SCIP  (full)
    - CPLEX + greedy         (no SCIP)
    - greedy + SCIP          (no CPLEX)
    - greedy only
"""

import csv
from pathlib import Path


# ---------------------------------------------------------------------------
# Primitive helpers
# ---------------------------------------------------------------------------

def _fmt(val, fmt: str = ".2f") -> str:
    """Format a numeric value, or return 'N/A' for None."""
    return f"{val:{fmt}}" if val is not None else "N/A"


def _gap(reference: float | None, other: float | None) -> float | None:
    """Relative gap vs CPLEX: (other - reference) / |reference| * 100 %.
    Positive means *other* is worse. Returns None when either value is absent."""
    if reference is None or other is None or reference == 0:
        return None
    return (other - reference) / abs(reference) * 100.0


def _avg(key: str, rows: list[dict]) -> float | None:
    vals = [r.get(key) for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def _avg_gap(ref_key: str, other_key: str, rows: list[dict]) -> float | None:
    gaps = [_gap(r.get(ref_key), r.get(other_key)) for r in rows]
    valid = [g for g in gaps if g is not None]
    return sum(valid) / len(valid) if valid else None


def _gap_str(gap: float | None) -> str:
    return f"{gap:+.1f}" if gap is not None else "N/A"


def _avg_gap_str(gap: float | None) -> str:
    return f"{_fmt(gap, '.1f')}%" if gap is not None else "N/A"


def _print_table(header: str, data_rows: list[str], avg_row: str) -> None:
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for row in data_rows:
        print(row)
    print(sep)
    print(avg_row)
    print(sep)


# ---------------------------------------------------------------------------
# Layout: CPLEX + greedy + SCIP  (full)
# ---------------------------------------------------------------------------

def _layout_full(rows: list[dict]) -> None:
    W = dict(inst=12, cd=9, gd=9, gg=8, sd=9, gs=8,
             cc=9, gc=9, sc=9, cl=9, gl=9, sl=9, cn=7, gn=7, sn=7)
    h = (f"{'Instance':<{W['inst']}} "
         f"{'CPLEX dist':>{W['cd']}} {'Grdy dist':>{W['gd']}} {'Gap g%':>{W['gg']}} "
         f"{'SCIP dist':>{W['sd']}} {'Gap s%':>{W['gs']}} "
         f"{'CPLEX cost':>{W['cc']}} {'Grdy cost':>{W['gc']}} {'SCIP cost':>{W['sc']}} "
         f"{'CPLEX loss':>{W['cl']}} {'Grdy loss':>{W['gl']}} {'SCIP loss':>{W['sl']}} "
         f"{'CPLEX n':>{W['cn']}} {'Grdy n':>{W['gn']}} {'SCIP n':>{W['sn']}}")

    def row(r):
        gg = _gap_str(_gap(r["cplex_distance"], r["greedy_distance"]))
        gs = _gap_str(_gap(r["cplex_distance"], r.get("scip_distance")))
        sn = str(r["scip_n"]) if r.get("scip_n") is not None else "N/A"
        return (f"{r['stem']:<{W['inst']}} "
                f"{_fmt(r['cplex_distance']):>{W['cd']}} {_fmt(r['greedy_distance']):>{W['gd']}} {gg:>{W['gg']}} "
                f"{_fmt(r.get('scip_distance')):>{W['sd']}} {gs:>{W['gs']}} "
                f"{_fmt(r['cplex_cost']):>{W['cc']}} {_fmt(r['greedy_cost']):>{W['gc']}} {_fmt(r.get('scip_cost')):>{W['sc']}} "
                f"{_fmt(r['cplex_loss']):>{W['cl']}} {_fmt(r['greedy_loss']):>{W['gl']}} {_fmt(r.get('scip_loss')):>{W['sl']}} "
                f"{r['cplex_n']:>{W['cn']}} {r['greedy_n']:>{W['gn']}} {sn:>{W['sn']}}")

    avg_gg = _avg_gap_str(_avg_gap("cplex_distance", "greedy_distance", rows))
    avg_gs = _avg_gap_str(_avg_gap("cplex_distance", "scip_distance",   rows))
    avg_row = (f"{'AVERAGE':<{W['inst']}} "
               f"{_fmt(_avg('cplex_distance',  rows)):>{W['cd']}} {_fmt(_avg('greedy_distance', rows)):>{W['gd']}} {avg_gg:>{W['gg']}} "
               f"{_fmt(_avg('scip_distance',   rows)):>{W['sd']}} {avg_gs:>{W['gs']}} "
               f"{_fmt(_avg('cplex_cost', rows)):>{W['cc']}} {_fmt(_avg('greedy_cost', rows)):>{W['gc']}} {_fmt(_avg('scip_cost', rows)):>{W['sc']}} "
               f"{_fmt(_avg('cplex_loss', rows)):>{W['cl']}} {_fmt(_avg('greedy_loss', rows)):>{W['gl']}} {_fmt(_avg('scip_loss', rows)):>{W['sl']}} "
               f"{'':>{W['cn']}} {'':>{W['gn']}} {'':>{W['sn']}}")

    _print_table(h, [row(r) for r in rows], avg_row)


# ---------------------------------------------------------------------------
# Layout: CPLEX + greedy  (no SCIP)
# ---------------------------------------------------------------------------

def _layout_cplex_greedy(rows: list[dict]) -> None:
    W = dict(inst=12, cd=9, gd=9, gap=8, cc=9, gc=9, cl=9, gl=9, cn=7, gn=7)
    h = (f"{'Instance':<{W['inst']}} "
         f"{'CPLEX dist':>{W['cd']}} {'Grdy dist':>{W['gd']}} {'Gap%':>{W['gap']}} "
         f"{'CPLEX cost':>{W['cc']}} {'Grdy cost':>{W['gc']}} "
         f"{'CPLEX loss':>{W['cl']}} {'Grdy loss':>{W['gl']}} "
         f"{'CPLEX n':>{W['cn']}} {'Grdy n':>{W['gn']}}")

    def row(r):
        g = _gap_str(_gap(r["cplex_distance"], r["greedy_distance"]))
        return (f"{r['stem']:<{W['inst']}} "
                f"{_fmt(r['cplex_distance']):>{W['cd']}} {_fmt(r['greedy_distance']):>{W['gd']}} {g:>{W['gap']}} "
                f"{_fmt(r['cplex_cost']):>{W['cc']}} {_fmt(r['greedy_cost']):>{W['gc']}} "
                f"{_fmt(r['cplex_loss']):>{W['cl']}} {_fmt(r['greedy_loss']):>{W['gl']}} "
                f"{r['cplex_n']:>{W['cn']}} {r['greedy_n']:>{W['gn']}}")

    avg_g = _avg_gap_str(_avg_gap("cplex_distance", "greedy_distance", rows))
    avg_row = (f"{'AVERAGE':<{W['inst']}} "
               f"{_fmt(_avg('cplex_distance', rows)):>{W['cd']}} {_fmt(_avg('greedy_distance', rows)):>{W['gd']}} {avg_g:>{W['gap']}} "
               f"{_fmt(_avg('cplex_cost', rows)):>{W['cc']}} {_fmt(_avg('greedy_cost', rows)):>{W['gc']}} "
               f"{_fmt(_avg('cplex_loss', rows)):>{W['cl']}} {_fmt(_avg('greedy_loss', rows)):>{W['gl']}} "
               f"{'':>{W['cn']}} {'':>{W['gn']}}")

    _print_table(h, [row(r) for r in rows], avg_row)


# ---------------------------------------------------------------------------
# Layout: greedy + SCIP  (no CPLEX)
# ---------------------------------------------------------------------------

def _layout_greedy_scip(rows: list[dict]) -> None:
    W = dict(inst=12, gd=9, sd=9, gc=9, sc=9, gl=9, sl=9, gn=7, sn=7)
    h = (f"{'Instance':<{W['inst']}} "
         f"{'Grdy dist':>{W['gd']}} {'SCIP dist':>{W['sd']}} "
         f"{'Grdy cost':>{W['gc']}} {'SCIP cost':>{W['sc']}} "
         f"{'Grdy loss':>{W['gl']}} {'SCIP loss':>{W['sl']}} "
         f"{'Grdy n':>{W['gn']}} {'SCIP n':>{W['sn']}}")

    def row(r):
        sn = str(r["scip_n"]) if r.get("scip_n") is not None else "N/A"
        return (f"{r['stem']:<{W['inst']}} "
                f"{_fmt(r['greedy_distance']):>{W['gd']}} {_fmt(r.get('scip_distance')):>{W['sd']}} "
                f"{_fmt(r['greedy_cost']):>{W['gc']}} {_fmt(r.get('scip_cost')):>{W['sc']}} "
                f"{_fmt(r['greedy_loss']):>{W['gl']}} {_fmt(r.get('scip_loss')):>{W['sl']}} "
                f"{r['greedy_n']:>{W['gn']}} {sn:>{W['sn']}}")

    avg_row = (f"{'AVERAGE':<{W['inst']}} "
               f"{_fmt(_avg('greedy_distance', rows)):>{W['gd']}} {_fmt(_avg('scip_distance', rows)):>{W['sd']}} "
               f"{_fmt(_avg('greedy_cost', rows)):>{W['gc']}} {_fmt(_avg('scip_cost', rows)):>{W['sc']}} "
               f"{_fmt(_avg('greedy_loss', rows)):>{W['gl']}} {_fmt(_avg('scip_loss', rows)):>{W['sl']}} "
               f"{'':>{W['gn']}} {'':>{W['sn']}}")

    _print_table(h, [row(r) for r in rows], avg_row)


# ---------------------------------------------------------------------------
# Layout: greedy only
# ---------------------------------------------------------------------------

def _layout_greedy_only(rows: list[dict]) -> None:
    W = dict(inst=12, gd=9, gc=9, gl=9, gn=7)
    h = (f"{'Instance':<{W['inst']}} "
         f"{'Grdy dist':>{W['gd']}} {'Grdy cost':>{W['gc']}} "
         f"{'Grdy loss':>{W['gl']}} {'Grdy n':>{W['gn']}}")

    def row(r):
        return (f"{r['stem']:<{W['inst']}} "
                f"{_fmt(r['greedy_distance']):>{W['gd']}} {_fmt(r['greedy_cost']):>{W['gc']}} "
                f"{_fmt(r['greedy_loss']):>{W['gl']}} {r['greedy_n']:>{W['gn']}}")

    avg_row = (f"{'AVERAGE':<{W['inst']}} "
               f"{_fmt(_avg('greedy_distance', rows)):>{W['gd']}} {_fmt(_avg('greedy_cost', rows)):>{W['gc']}} "
               f"{_fmt(_avg('greedy_loss', rows)):>{W['gl']}} {'':>{W['gn']}}")

    _print_table(h, [row(r) for r in rows], avg_row)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def print_table(rows: list[dict]) -> None:
    """Print a formatted comparison table, choosing the layout based on available data."""
    has_cplex = any(r.get("cplex_distance") is not None for r in rows)
    has_scip  = any(r.get("scip_distance")  is not None for r in rows)

    if   has_cplex and has_scip:  _layout_full(rows)
    elif has_cplex:               _layout_cplex_greedy(rows)
    elif has_scip:                _layout_greedy_scip(rows)
    else:                         _layout_greedy_only(rows)


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
            gap_g = _gap(r.get("cplex_distance"), r["greedy_distance"])
            gap_s = _gap(r.get("cplex_distance"), r.get("scip_distance"))
            writer.writerow({
                "instance":                r["stem"],
                "cplex_distance":          r.get("cplex_distance") or "",
                "greedy_distance":         r["greedy_distance"],
                "greedy_distance_gap_pct": f"{gap_g:.4f}" if gap_g is not None else "",
                "scip_distance":           r.get("scip_distance") or "",
                "scip_distance_gap_pct":   f"{gap_s:.4f}" if gap_s is not None else "",
                "cplex_cost":              r.get("cplex_cost") or "",
                "greedy_cost":             r.get("greedy_cost") or "",
                "scip_cost":               r.get("scip_cost") or "",
                "cplex_loss":              r.get("cplex_loss") or "",
                "greedy_loss":             r.get("greedy_loss") or "",
                "scip_loss":               r.get("scip_loss") or "",
                "cplex_n_wells":           r.get("cplex_n") or "",
                "greedy_n_wells":          r["greedy_n"],
                "scip_n_wells":            r.get("scip_n") or "",
            })
    print(f"\n[info] CSV written to: {path}")
