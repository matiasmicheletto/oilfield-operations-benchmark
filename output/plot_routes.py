#!/usr/bin/env python3
"""
Overlay solver-computed crew routes on an instance's spatial map.

Workflow:
  1. Run the instance generator (main.py) – produces spatial_data_<id>.npz.
  2. Run the solver with -f routes and -o <routes_file>.
  3. Run this script to produce a route-overlay PNG.

Usage:
    python plot_routes.py \\
        --spatial  path/to/spatial_data.npz \\
        --routes   path/to/routes.txt \\
        [--output  path/to/route_overlay.png]

Routes file format (solver -f routes):
    Each line:  <crew_id> [well_id ...]
    Depot (well 0) is omitted.  Empty crews have no well IDs after the crew number.

    Example:
        1 6 7 2
        2 5 4 1
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from skimage.graph import MCP_Geometric

# Up to 10 visually distinct crew colours
_CREW_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
    "#fabed4", "#469990",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_routes(path: Path) -> dict[int, list[int]]:
    """Parse solver ROUTES format → {crew_id: [well_id, ...]}."""
    routes: dict[int, list[int]] = {}
    with open(path) as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens:
                continue
            crew_id = int(tokens[0])
            routes[crew_id] = [int(t) for t in tokens[1:]]
    return routes


def trace_geodesic(cost_map: np.ndarray, start: tuple, end: tuple) -> np.ndarray:
    """Return pixel path (N,2) tracing the least-cost geodesic route."""
    mcp = MCP_Geometric(cost_map)
    mcp.find_costs(starts=[start], ends=[end])
    return np.array(mcp.traceback(end))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Overlay solver crew routes on the instance spatial map."
    )
    ap.add_argument(
        "--spatial", required=True,
        help="spatial_data_<id>.npz produced by the instance generator.",
    )
    ap.add_argument(
        "--routes", required=True,
        help="Solver output file generated with -f routes.",
    )
    ap.add_argument(
        "--output", default=None,
        help="Output PNG path.  Defaults to route_overlay_<id>.png next to the NPZ.",
    )
    args = ap.parse_args()

    spatial_path = Path(args.spatial)
    routes_path  = Path(args.routes)

    if not spatial_path.exists():
        sys.exit(f"[error] Spatial file not found: {spatial_path}")
    if not routes_path.exists():
        sys.exit(f"[error] Routes file not found: {routes_path}")

    # ---- Load spatial data -------------------------------------------------
    data           = np.load(spatial_path)
    elevation      = data["elevation"]
    cost_map       = data["cost_map"]
    well_positions = data["well_positions"]   # (n_wells, 2) – row i → well ID i+1
    ops_center     = tuple(int(v) for v in data["ops_center"])
    road_mask      = data["road_mask"]

    # ---- Parse routes file -------------------------------------------------
    routes = parse_routes(routes_path)
    if not any(wids for wids in routes.values()):
        sys.exit("[error] Routes file contains no well assignments.")

    # ---- Trace geodesic segments per crew ----------------------------------
    print("[plot_routes] Tracing geodesic paths …")
    crew_segments: dict[int, list[np.ndarray]] = {}
    for crew_id, well_ids in sorted(routes.items()):
        if not well_ids:
            continue
        sequence = (
            [ops_center]
            + [tuple(int(v) for v in well_positions[wid - 1]) for wid in well_ids]
            + [ops_center]
        )
        segments = []
        for src, dst in zip(sequence[:-1], sequence[1:]):
            segments.append(trace_geodesic(cost_map, src, dst))
        crew_segments[crew_id] = segments
        print(f"  Crew {crew_id}: {well_ids}")

    # ---- Build plot --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 12))

    # Terrain background
    ax.imshow(elevation, cmap="terrain", origin="lower", alpha=0.6)

    # Existing road network as a semi-transparent brown pixel overlay
    road_rgba = np.zeros((*road_mask.shape, 4), dtype=float)
    road_rgba[road_mask] = [0.47, 0.33, 0.28, 0.35]
    ax.imshow(road_rgba, origin="lower", zorder=2)

    # All wells (context)
    ax.scatter(
        well_positions[:, 1], well_positions[:, 0],
        color="silver", marker="o", s=18,
        edgecolors="black", linewidths=0.4,
        zorder=4, label="Wells (unvisited)",
    )

    # Crew routes
    visited_ids: set[int] = set()
    for idx, (crew_id, segments) in enumerate(sorted(crew_segments.items())):
        color = _CREW_COLORS[idx % len(_CREW_COLORS)]
        label_used = False
        for seg in segments:
            ax.plot(
                seg[:, 1], seg[:, 0],
                color=color, linewidth=2.5, alpha=0.92, zorder=5,
                label=f"Crew {crew_id}" if not label_used else None,
            )
            label_used = True
        visited_ids.update(routes[crew_id])

    # Highlight visited wells
    for wid in visited_ids:
        r, c = well_positions[wid - 1]
        ax.scatter(c, r, color="red", marker="o", s=40,
                   edgecolors="black", linewidths=0.6, zorder=6)
        ax.annotate(
            str(wid), (c, r),
            fontsize=6.5, color="white", fontweight="bold",
            ha="center", va="center", zorder=7,
        )

    # Ops centre
    ax.scatter(
        ops_center[1], ops_center[0],
        color="blue", marker="s", s=120,
        edgecolors="white", linewidths=1.2,
        zorder=8, label="Ops Centre",
    )

    instance_id = spatial_path.stem.replace("spatial_data_", "")
    ax.set_title(f"Solver Crew Routes – Instance {instance_id}", fontsize=15)
    ax.legend(loc="upper right", fontsize=9)
    ax.axis("off")
    plt.tight_layout()

    out_path = Path(args.output) if args.output else \
               spatial_path.parent / f"route_overlay_{instance_id}.png"
    plt.savefig(out_path, dpi=200)
    print(f"[plot_routes] Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
