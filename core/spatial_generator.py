# core/spatial_generator.py

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.graph import MCP_Geometric
from skimage.filters import gaussian, sobel
from scipy.ndimage import zoom


class SpatialGenerator:
    def __init__(self, rng, config):
        self.rng = rng
        self.config = config
        self.size = config["spatial"]["grid_size"]

    # ------------------------------------------------------------------
    # Terrain
    # ------------------------------------------------------------------
    def generate_terrain(self):
        low_res = self.rng.random((20, 20))
        elevation = zoom(low_res, self.size / 20, order=3) * 100
        elevation = gaussian(elevation, sigma=2)

        grad_x = sobel(elevation, axis=0)
        grad_y = sobel(elevation, axis=1)
        slope = np.sqrt(grad_x**2 + grad_y**2)

        cost_map = 1.0 + (slope ** 2)
        return elevation, cost_map

    # ------------------------------------------------------------------
    # Well Coordinates (clustered like original example)
    # ------------------------------------------------------------------
    def generate_well_positions(self):
        n_wells = self.config["general"]["n_wells"]

        n_clusters = 5
        centers = self.rng.uniform(50, self.size - 50, size=(n_clusters, 2))

        wells = []
        for c in centers:
            w = self.rng.normal(loc=c, scale=15,
                                size=(n_wells // n_clusters, 2))
            wells.extend(w)

        wells = np.clip(np.array(wells), 0, self.size - 1)
        return wells.astype(int)

    # ------------------------------------------------------------------
    # Road Network Growth (corridor reuse)
    # ------------------------------------------------------------------
    def build_network(self, cost_map, wells):
        ops_center = (self.size // 10, self.size // 10)

        wells_sorted = sorted(
            wells,
            key=lambda w: np.linalg.norm(w - np.array(ops_center))
        )

        dynamic_cost = cost_map.copy()
        all_paths = []

        for well in wells_sorted:
            start_node = tuple(well)
            mcp = MCP_Geometric(dynamic_cost)

            _, _ = mcp.find_costs(starts=[start_node],
                                  ends=[ops_center])
            path = mcp.traceback(ops_center)

            all_paths.append(path)

            # Corridor reuse
            for r, c in path:
                dynamic_cost[r, c] = 0.001

        return ops_center, all_paths

    # ------------------------------------------------------------------
    # Distance Matrix (from terrain cost only)
    # ------------------------------------------------------------------
    def compute_distance_matrix(self, cost_map, wells):
        n = len(wells)
        D = np.zeros((n + 1, n + 1))

        depot = (self.size // 10, self.size // 10)
        nodes = [depot] + [tuple(w) for w in wells]

        mcp = MCP_Geometric(cost_map)

        for i, start in enumerate(nodes):
            costs, _ = mcp.find_costs(starts=[start])
            for j, end in enumerate(nodes):
                D[i, j] = costs[end]

        return D

    # ------------------------------------------------------------------
    # Plot (same format as original prompt)
    # ------------------------------------------------------------------
    def plot_network(self, elevation, wells, paths, ops_center, instance_id):
        plot_cfg = self.config["general"].get("plot", {})
        if not plot_cfg.get("save") and not plot_cfg.get("show"):
            return

        fig, ax = plt.subplots(figsize=(12, 12))

        # Elevation background
        ax.imshow(elevation, cmap='terrain',
                  origin='lower', alpha=0.5)

        # Roads
        for path in paths:
            pts = np.array(path)
            ax.plot(pts[:, 1], pts[:, 0],
                    color="#5d4037", linewidth=2.5,
                    alpha=0.6, zorder=2)
            ax.plot(pts[:, 1], pts[:, 0],
                    color="#d7ccc8", linewidth=1.0,
                    alpha=0.9, zorder=3)

        # Wells
        wells = np.array(wells)
        ax.scatter(wells[:, 1], wells[:, 0],
                   color="white", marker="s", s=20,
                   edgecolors="black", linewidths=0.5,
                   zorder=5, label="Wells")

        # Ops Center
        ax.scatter(ops_center[1], ops_center[0],
                   color="#d32f2f", marker="s", s=100,
                   edgecolors="white", zorder=6,
                   label="Ops Center")

        ax.set_title(
            f"Optimized Oilfield Road Evolution - Instance {instance_id}",
            fontsize=15
        )

        ax.legend(loc='upper right')
        ax.axis('off')
        plt.tight_layout()

        if plot_cfg.get("save"):
            out_dir = Path(self.config["general"]["output_dir"])
            fname = f"spatial_network_{instance_id}.png"
            plt.savefig(out_dir / fname, dpi=300)

        if plot_cfg.get("show"):
            plt.show()

        plt.close()