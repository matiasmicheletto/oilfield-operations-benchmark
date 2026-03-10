import numpy as np
import matplotlib.pyplot as plt
from skimage.graph import MCP_Geometric
from skimage.filters import gaussian, sobel
from scipy.ndimage import zoom


class SpatialGenerator:
    def __init__(self, rng, config):
        self.rng = rng
        self.config = config["spatial"]
        self.general_cfg = config.get("general", {})
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
        n_clusters = max(1, min(int(n_clusters), int(n_wells)))
        padding = size * 0.1
        centers = self.rng.uniform(padding, size - padding, size=(n_clusters, 2))

        # Distribute wells across clusters while preserving the exact total.
        base = n_wells // n_clusters
        remainder = n_wells % n_clusters
        wells_per_cluster = np.full(n_clusters, base, dtype=int)
        wells_per_cluster[:remainder] += 1

        wells = []
        for c, count in zip(centers, wells_per_cluster):
            w = self.rng.normal(loc=c, scale=15,
                                size=(count, 2))
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
    # Well-to-Depot Network (with corridor reuse)
    # ------------------------------------------------------------------
    def _build_well_connectors(self, cost_map, wells, existing_road_mask=None):
        size = self.config["grid_size"]
        max_dist_cfg = float(self.config.get("max_dist_fraction", 0.25))
        max_dist = max_dist_cfg * size if max_dist_cfg <= 1.0 else max_dist_cfg
        min_links = max(0, int(self.config.get("min_per_well", 2)))
        max_links = max(min_links, int(self.config.get("max_per_well", 3)))
        max_path_factor = float(self.config.get("max_path_factor", 1.8))
        reuse_penalty = float(self.config.get("connector_reuse_penalty", 3.0))

        connectors = []
        wells = np.array(wells)
        n = len(wells)
        degrees = np.zeros(n, dtype=int)
        used_pairs = set()

        connector_cost = cost_map.copy()
        if existing_road_mask is not None:
            connector_cost = connector_cost + existing_road_mask.astype(float) * np.mean(cost_map) * reuse_penalty

        for i in range(n):
            if degrees[i] >= max_links:
                continue

            dists = np.linalg.norm(wells - wells[i], axis=1)
            candidates = np.where((dists > 0) & (dists < max_dist))[0]

            if len(candidates) == 0:
                continue

            # Prioritize close neighbors to increase feasible secondary links.
            candidates = candidates[np.argsort(dists[candidates])]

            for j in candidates:
                if degrees[i] >= max_links:
                    break
                if i == j or degrees[j] >= max_links:
                    continue

                pair = (min(i, j), max(i, j))
                if pair in used_pairs:
                    continue

                start = tuple(wells[i])
                end = tuple(wells[j])

                mcp = MCP_Geometric(connector_cost)
                costs, _ = mcp.find_costs(starts=[start], ends=[end])
                path_cost = costs[end]

                if not np.isfinite(path_cost):
                    continue

                # Scale admissible path cost by euclidean separation.
                euclid_dist = max(dists[j], 1.0)
                max_path_cost = euclid_dist * max_path_factor
                if path_cost > max_path_cost and degrees[i] >= min_links:
                    continue

                path = mcp.traceback(end)
                connectors.append(path)
                used_pairs.add(pair)
                degrees[i] += 1
                degrees[j] += 1

                # Encourage connector corridor reuse among secondary roads.
                for r, c in path:
                    connector_cost[r, c] = 0.001

                if degrees[i] >= max_links:
                    break

        return connectors


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
        primary_road_mask = np.zeros_like(cost_map, dtype=bool)

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
                primary_road_mask[r, c] = True

        connectors = self._build_well_connectors(cost_map, wells, existing_road_mask=primary_road_mask)

        for path in connectors:
            for r, c in path:
                dynamic_cost[r, c] = 0.001

        all_paths.extend(connectors)

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
        save_plot = self.general_cfg.get("save_plot", True)
        show_plot = self.general_cfg.get("show_plot", False)
        if not save_plot and not show_plot:
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

        if save_plot:
            fname = f"spatial_network_{instance_id}.png"
            plt.savefig(output_dir / fname, dpi=300)

        if show_plot:
            plt.show()

        plt.close()