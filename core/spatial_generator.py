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
        self.config = config["spatial"]
        self.ops_center = None

    # ------------------------------------------------------------------
    # Terrain
    # ------------------------------------------------------------------
    def generate_terrain(self):
        size = self.config["grid_size"]
        seed_res = self.config.get("seed_resolution", 20)
        smooth_sigma = self.config.get("smooth_sigma", 2)
        cost_exponent = self.config.get("cost_exponent", 2)
        num_peaks = max(1, int(self.config.get("num_peaks", 6)))
        elevation_amplitude = float(self.config.get("elevation_amplitude", 100.0))

        low_res = self.rng.random((seed_res, seed_res))
        base_terrain = zoom(low_res, size / seed_res, order=3)

        # Explicit Gaussian peaks make terrain morphology directly configurable.
        x = np.arange(size)
        y = np.arange(size)
        xx, yy = np.meshgrid(x, y)
        peaks = np.zeros((size, size), dtype=float)
        for _ in range(num_peaks):
            cx = self.rng.uniform(0, size - 1)
            cy = self.rng.uniform(0, size - 1)
            spread = self.rng.uniform(size * 0.04, size * 0.12)
            height = self.rng.uniform(0.6, 1.0)
            peaks += height * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * spread ** 2))

        base_terrain = (base_terrain - base_terrain.min()) / (np.ptp(base_terrain) + 1e-9)
        peaks = (peaks - peaks.min()) / (np.ptp(peaks) + 1e-9)
        elevation = (0.35 * base_terrain + 0.65 * peaks) * elevation_amplitude
        elevation = gaussian(elevation, sigma=smooth_sigma)

        grad_x = sobel(elevation, axis=0)
        grad_y = sobel(elevation, axis=1)
        slope = np.sqrt(grad_x**2 + grad_y**2)

        cost_map = 1.0 + (slope ** cost_exponent)
        return elevation, cost_map

    # ------------------------------------------------------------------
    # Well Coordinates (clustered like original example)
    # ------------------------------------------------------------------
    def generate_well_positions(self, n_wells):
        size = self.config["grid_size"]
        n_clusters = self.config.get("n_clusters", 4)
        padding = size * 0.1
        centers = self.rng.uniform(padding, size - padding, size=(n_clusters, 2))

        wells = []
        for c in centers:
            w = self.rng.normal(loc=c, scale=15,
                                size=(n_wells // n_clusters, 2))
            wells.extend(w)

        wells = np.clip(np.array(wells), 0, size - 1)
        return wells.astype(int)

    # ------------------------------------------------------------------
    # Ops Center Placement
    # ------------------------------------------------------------------
    def _place_ops_center(self):
        """Pick a random position near the grid centre (within ~12% radius)."""
        size = self.config["grid_size"]
        center = size // 2
        spread = size // 8
        r = int(self.rng.integers(center - spread, center + spread + 1))
        c = int(self.rng.integers(center - spread, center + spread + 1))
        return (r, c)

    # ------------------------------------------------------------------
    # Road Network Growth (corridor reuse)
    # ------------------------------------------------------------------
    def build_network(self, cost_map, wells):
        if self.ops_center is None:
            self.ops_center = self._place_ops_center()
        ops_center = self.ops_center

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

        depot = self.ops_center if self.ops_center is not None else self._place_ops_center()
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
    def plot_network(self, elevation, wells, paths, ops_center, instance_id, output_dir):
        if not self.config["save_plot"] and not self.config["show_plot"]:
            return

        fig, ax = plt.subplots(figsize=(12, 12))

        # Elevation background
        ax.imshow(elevation, cmap='terrain',
                  origin='lower', alpha=0.5)

        # Roads
        for path in paths:
            pts = np.array(path)
            ax.plot(pts[:, 1], pts[:, 0],
                    color="#5d4037", linewidth=3,
                    alpha=0.6, zorder=2)
            ax.plot(pts[:, 1], pts[:, 0],
                    color="#d7ccc8", linewidth=1.5,
                    alpha=0.9, zorder=3)

        # Wells
        wells = np.array(wells)
        ax.scatter(wells[:, 1], wells[:, 0],
                   color="red", marker="o", s=20,
                   edgecolors="black", linewidths=0.5,
                   zorder=5, label="Wells")

        # Ops Center
        ax.scatter(ops_center[1], ops_center[0],
                   color="blue", marker="s", s=100,
                   edgecolors="white", zorder=6,
                   label="Ops Center")

        ax.set_title(
            f"Optimized Oilfield Road Evolution - Instance {instance_id}",
            fontsize=15
        )

        ax.legend(loc='upper right')
        ax.axis('off')
        plt.tight_layout()

        if self.config["save_plot"]:
            fname = f"spatial_network_{instance_id}.png"
            plt.savefig(output_dir / fname, dpi=300)

        if self.config["show_plot"]:
            plt.show()

        plt.close()