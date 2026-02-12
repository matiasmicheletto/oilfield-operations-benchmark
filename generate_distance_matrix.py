#!/usr/bin/env python3

import sys
import yaml
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree


# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------

def load_config(path):
    if not Path(path).exists():
        print(f"Configuration file '{path}' not found.")
        sys.exit(1)

    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    if seed is not None:
        np.random.seed(seed)


# ------------------------------------------------------------
# Spatial generation
# ------------------------------------------------------------

def generate_clusters(n_wells, cfg):
    area_km = cfg["area_size_km"]
    n_clusters = cfg["n_clusters"]
    radius_m = cfg["cluster_radius_m"]

    area_m = area_km * 1000

    centers = np.random.uniform(0, area_m, size=(n_clusters, 2))

    wells_per_cluster = int(np.ceil(n_wells / n_clusters))

    coords = []
    for c in centers:
        for _ in range(wells_per_cluster):
            point = c + np.random.normal(0, radius_m, size=2)
            coords.append(point)

    coords = np.array(coords[:n_wells])
    return coords, centers


def add_operations_center(coords, centers, cfg):
    loc_type = cfg["operations_center"]["location"]

    if loc_type == "random_cluster":
        center = centers[np.random.randint(len(centers))]
    elif loc_type == "center":
        center = np.mean(coords, axis=0)
    else:
        raise ValueError("Invalid operations_center location")

    return np.vstack([center, coords])


# ------------------------------------------------------------
# Road network
# ------------------------------------------------------------

def build_road_graph(coords, road_cfg):
    n = len(coords)

    eucl_dist = distance_matrix(coords, coords)
    mst_sparse = minimum_spanning_tree(eucl_dist)
    mst = mst_sparse.toarray()

    G = nx.Graph()

    # Add MST edges
    for i in range(n):
        for j in range(n):
            if mst[i, j] > 0:
                detour = np.random.uniform(
                    road_cfg["detour_min"],
                    road_cfg["detour_max"]
                )
                G.add_edge(i, j, weight=mst[i, j] * detour)

    # Add extra edges
    possible_edges = [
        (i, j) for i in range(n) for j in range(i + 1, n)
        if not G.has_edge(i, j)
    ]

    n_extra = int(len(G.edges) * road_cfg["extra_edge_ratio"])

    if possible_edges and n_extra > 0:
        chosen = np.random.choice(
            len(possible_edges),
            size=min(n_extra, len(possible_edges)),
            replace=False
        )

        for idx in chosen:
            i, j = possible_edges[idx]
            detour = np.random.uniform(
                road_cfg["detour_min"],
                road_cfg["detour_max"]
            )
            G.add_edge(i, j, weight=eucl_dist[i, j] * detour)

    return G


# ------------------------------------------------------------
# Distance matrix
# ------------------------------------------------------------

def compute_shortest_path_matrix(G):
    n = len(G.nodes)
    dist_matrix = np.zeros((n, n))

    lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))

    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = lengths[i][j]

    return dist_matrix


def save_distance_matrix(D, output_path):
    with open(output_path, "w") as f:
        f.write(f"{D.shape[0]}\n")
        for row in D:
            row_str = " ".join(f"{int(round(x))}" for x in row)
            f.write(row_str + "\n")


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

def plot_graph(G, coords, save_path=None, show=False):
    plt.figure(figsize=(8, 8))

    for (i, j) in G.edges:
        x = [coords[i, 0], coords[j, 0]]
        y = [coords[i, 1], coords[j, 1]]
        plt.plot(x, y, alpha=0.5)

    plt.scatter(coords[1:, 0], coords[1:, 1])
    plt.scatter(coords[0, 0], coords[0, 1], marker="s")

    plt.title("Synthetic Oilfield Road Network")
    plt.xlabel("Meters")
    plt.ylabel("Meters")
    plt.axis("equal")

    if save_path:
        plt.savefig(save_path, dpi=300)

    if show:
        plt.show()

    plt.close()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    # Use default_config.yaml if no argument provided
    if len(sys.argv) == 1:
        config_path = "default_config.yaml"
    else:
        config_path = sys.argv[1]

    config = load_config(config_path)

    set_seed(config["general"].get("seed"))

    n_wells = config["general"]["n_wells"]
    spatial_cfg = config["spatial"]

    coords, centers = generate_clusters(n_wells, spatial_cfg)
    coords = add_operations_center(coords, centers, spatial_cfg)

    G = build_road_graph(coords, spatial_cfg["road_network"])
    D = compute_shortest_path_matrix(G)

    output_dir = Path(spatial_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / spatial_cfg["output_file_name"]

    save_distance_matrix(D, output_file)

    # Plot behavior from YAML
    save_plot = spatial_cfg["plot"].get("save", False)
    show_plot = spatial_cfg["plot"].get("show", False)

    if save_plot or show_plot:
        plot_path = None
        if save_plot:
            plot_path = output_dir / "road_network.png"

        plot_graph(G, coords, save_path=plot_path, show=show_plot)


if __name__ == "__main__":
    main()
