import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from skimage.graph import MCP_Geometric
from skimage.filters import gaussian, sobel
from scipy.ndimage import zoom

# ---------------------------------------------------------------------------
# 1. Efficient Terrain & Cost Generation
# ---------------------------------------------------------------------------

def generate_efficient_terrain(size=300):
    """Creates elevation and a slope-based cost map."""
    # Generate low-res noise and upscale for smoothness (saves memory)
    low_res = np.random.rand(20, 20)
    elevation = zoom(low_res, size / 20, order=3) * 100
    elevation = gaussian(elevation, sigma=2)

    # Slope-based cost: roads hate steep inclines
    grad_x = sobel(elevation, axis=0)
    grad_y = sobel(elevation, axis=1)
    slope = np.sqrt(grad_x**2 + grad_y**2)
    
    # Base cost is 1.0, plus a penalty for slope
    cost_map = 1.0 + (slope ** 2)
    return elevation, cost_map

# ---------------------------------------------------------------------------
# 2. Incremental Growth with MCP (Fast Pathfinding)
# ---------------------------------------------------------------------------

def build_network_efficiently(n_wells=50, size=300):
    elevation, cost_map = generate_efficient_terrain(size)
    
    # Starting point (Ops Center)
    ops_center = (size // 10, size // 10)
    
    # Place wells in clusters
    rng = np.random.default_rng()
    n_clusters = 5
    centers = rng.uniform(50, size-50, size=(n_clusters, 2))
    wells = []
    for c in centers:
        w = rng.normal(loc=c, scale=15, size=(n_wells // n_clusters, 2))
        wells.extend(w)
    wells = np.clip(np.array(wells), 0, size-1).astype(int)

    # Sort wells by distance to Ops Center to simulate organic growth
    wells = sorted(wells, key=lambda w: np.linalg.norm(w - np.array(ops_center)))

    # Track road geometry
    road_mask = np.zeros((size, size), dtype=bool)
    road_mask[ops_center] = True
    
    # We use a copy of the cost map that we modify as we build
    dynamic_cost = cost_map.copy()
    all_paths = []

    for well in wells:
        start_node = tuple(well)
        
        # MCP finds the path from the well to ANY point where dynamic_cost is low
        # To reuse corridors, we set the goal as the Ops Center, 
        # but because existing roads have ~0 cost, it will 'snap' to them.
        mcp = MCP_Geometric(dynamic_cost)
        
        # Find path from well back to Ops Center
        # MCP is extremely fast compared to manual A*
        cum_costs, traceback = mcp.find_costs(starts=[start_node], ends=[ops_center])
        path = mcp.traceback(ops_center)
        
        all_paths.append(path)
        
        # "Pave" the road: set cost of these pixels to near-zero 
        # so the NEXT well reuses this corridor.
        for r, c in path:
            dynamic_cost[r, c] = 0.001 
            road_mask[r, c] = True

    return elevation, wells, all_paths, ops_center

# ---------------------------------------------------------------------------
# 3. Visualization
# ---------------------------------------------------------------------------

size_px = 300
elev, wells_pos, road_paths, oc = build_network_efficiently(n_wells=40, size=size_px)

fig, ax = plt.subplots(figsize=(12, 12))
# Background: Elevation map
ax.imshow(elev, cmap='terrain', origin='lower', alpha=0.5)

# Plot roads
for path in road_paths:
    pts = np.array(path)
    # Road casing (brown)
    ax.plot(pts[:, 1], pts[:, 0], color="#5d4037", linewidth=2.5, alpha=0.6, zorder=2)
    # Road surface (tan)
    ax.plot(pts[:, 1], pts[:, 0], color="#d7ccc8", linewidth=1.0, alpha=0.9, zorder=3)

# Plot Wells
wells_pos = np.array(wells_pos)
ax.scatter(wells_pos[:, 1], wells_pos[:, 0], color="white", marker="s", s=20, 
           edgecolors="black", linewidths=0.5, zorder=5, label="Wells")

# Ops Center
ax.scatter(oc[1], oc[0], color="#d32f2f", marker="s", s=100, edgecolors="white", 
           zorder=6, label="Ops Center")

ax.set_title("Optimized Oilfield Road Evolution (Corridor Reuse & A*)", fontsize=15)
ax.legend(loc='upper right')
ax.axis('off')
plt.tight_layout()
plt.show()