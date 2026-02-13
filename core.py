import numpy as np
import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree


class WellGenerator:
    def __init__(self, rng, config):
        self.rng = rng
        self.config = config

    def _sample_dist(self, n, dist_cfg):
        """Internal helper to sample from NumPy distributions based on YAML."""
        dist_name = dist_cfg.get("distribution", "uniform")
        params = {k: v for k, v in dist_cfg.items() if k != "distribution"}
        
        # Standardize parameter names for NumPy compatibility
        if dist_name == "lognormal" and "mu" in params:
            params["mean"] = params.pop("mu")
        if dist_name == "beta":
            if "alpha" in params: params["a"] = params.pop("alpha")
            if "beta" in params: params["b"] = params.pop("beta")
            
        generator = getattr(self.rng, dist_name)
        return generator(size=n, **params)

    def _correlated_sample(self, base, cfg):
        """Samples using either an independent distribution or correlation logic."""
        # Mode 1: Independent Distribution
        if "distribution" in cfg:
            return self._sample_dist(len(base), cfg)
        
        # Mode 2: Correlated with Production (Legacy)
        # Supports both 'rho' and 'rho_g' keys for flexibility
        rho = cfg.get("rho", cfg.get("rho_g", 0.8))
        noise_std = cfg.get("noise_std", 0.05)
        
        noise = self.rng.normal(0, noise_std, size=len(base))
        # Protect against division by zero if production is empty
        base_norm = base / (np.max(base) + 1e-9)
        
        return np.clip(rho * base_norm + noise, 0, 1)

    def _apply_scaling(self, G, N, r, R, P, C):
        """Scales raw [0,1] values to YAML-defined ranges and applies rounding."""
        round_cfg = self.config.get("rounding", {})
        scale_cfg = self.config.get("scaling", {})
        
        def process(arr, key, scale_key=None):
            s = scale_cfg.get(scale_key or key, 1)
            d = round_cfg.get(key, 0) # Defaults to 0 (integer)
            
            val_scaled = arr * s
            if d == 0:
                return np.round(val_scaled).astype(int)
            return np.round(val_scaled, d)

        return {
            "G": process(G, "production"),
            "N": process(N, "production"),
            "r": process(r, "regime"),
            "R": process(R, "risk"),
            "P": process(P, "priority"),
            "C": process(C, "cost")
        }

    def generate(self):
        """Main orchestrator for well data generation."""
        n_wells = self.config["general"]["n_wells"]
        prod_cfg = self.config["production"]
        rp_cfg = self.config["risk_priority"]
        
        # 1. Base Production Variables
        G_raw = self._sample_dist(n_wells, prod_cfg["gross"])
        net_ratio = self._sample_dist(n_wells, prod_cfg["net_ratio"])
        N_raw = G_raw * net_ratio
        r_raw = self._sample_dist(n_wells, prod_cfg["regime"])

        # 2. Risk & Priority (Independent vs Correlated Switch)
        R_raw = self._correlated_sample(G_raw, rp_cfg["risk"])
        P_raw = self._correlated_sample(N_raw, rp_cfg["priority"])
        
        # 3. Cost Calculation (Raw C is calculated before final scaling)
        c_cfg = rp_cfg["cost"]
        C_raw = c_cfg["kappa"] + c_cfg["w_r"] * R_raw + c_cfg["w_p"] * (1 - P_raw)

        # 4. Final Processing
        return self._apply_scaling(G_raw, N_raw, r_raw, R_raw, P_raw, C_raw)

    def plot_distributions(self, data, instance_id):
        """Generates 2x2 Bar Charts for well-by-well inspection."""
        self._plot_engine(data, instance_id, mode="bar")

    def plot_histograms(self, data, instance_id):
        """Generates 2x2 Histograms to visualize statistical shapes (Beta/LogNormal)."""
        self._plot_engine(data, instance_id, mode="hist")

    def _plot_engine(self, data, instance_id, mode="bar"):
        """Shared plotting logic for both bars and histograms."""
        plot_cfg = self.config["general"].get("plot", {})
        if not plot_cfg.get("save") and not plot_cfg.get("show"):
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        title_suffix = "Bar Charts" if mode == "bar" else "Histograms"
        fig.suptitle(f"Well Parameters ({title_suffix}) - Instance {instance_id}", fontsize=16)

        well_ids = np.arange(1, len(data["G"]) + 1)
        colors = {"G": "lightgray", "N": "green", "r": "blue", "R": "red", "P": "gold"}

        if mode == "bar":
            # 1. Gross/Net
            axes[0, 0].bar(well_ids, data["G"], label='Gross', color=colors["G"])
            axes[0, 0].bar(well_ids, data["N"], label='Net', color=colors["N"], alpha=0.7)
            # 2. Regime
            axes[0, 1].bar(well_ids, data["r"], color=colors["r"], alpha=0.6)
            # 3. Risk
            axes[1, 0].bar(well_ids, data["R"], color=colors["R"], alpha=0.6)
            # 4. Priority
            axes[1, 1].bar(well_ids, data["P"], color=colors["P"], alpha=0.7)
        else:
            # Histograms
            axes[0, 0].hist(data["G"], bins=15, color=colors["G"], edgecolor='black')
            axes[0, 1].hist(data["r"], bins=15, color=colors["r"], alpha=0.6, edgecolor='black')
            axes[1, 0].hist(data["R"], bins=15, color=colors["R"], alpha=0.6, edgecolor='black')
            axes[1, 1].hist(data["P"], bins=15, color=colors["P"], alpha=0.7, edgecolor='black')

        # Formatting
        titles = ["Gross vs Net", "Regime (%)", "Risk Factor", "Priority Level"]
        for i, ax in enumerate(axes.flat):
            ax.set_title(titles[i])
            if i > 0 and mode == "bar": # Apply Y-limits for scaled variables
                key = ["regime", "risk", "priority"][i-1]
                ax.set_ylim(0, self.config["scaling"].get(key, 100) * 1.1)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if plot_cfg.get("save"):
            out_dir = Path(self.config["general"].get("output_dir", "instances"))
            fname = f"well_{mode}_{instance_id}.png"
            plt.savefig(out_dir / fname, dpi=300)
        if plot_cfg.get("show"): plt.show()
        plt.close()


class SpatialGenerator:
    def __init__(self, rng, config):
        self.rng = rng
        self.config = config["spatial"]
        self.general_config = config["general"]
        self.n_wells = config["general"]["n_wells"]

    def generate_distance_matrix(self):
        # 1. Generate Coordinates
        area_m = self.config["area_size_km"] * 1000
        centers = self.rng.uniform(0, area_m, size=(self.config["n_clusters"], 2))
        w_per_c = int(np.ceil(self.n_wells / self.config["n_clusters"]))
        
        coords = []
        for c in centers:
            for _ in range(w_per_c):
                coords.append(c + self.rng.normal(0, self.config["cluster_radius_m"], size=2))
        coords = np.array(coords[:self.n_wells])

        # 2. Add Ops Center (Index 0)
        oc_loc = self.config["operations_center"]["location"]
        oc_coord = centers[self.rng.integers(len(centers))] if oc_loc == "random_cluster" else np.mean(coords, axis=0)
        full_coords = np.vstack([oc_coord, coords])

        # 3. Build Road Network (Graph)
        G = self._build_graph(full_coords)
        
        # 4. Shortest Paths
        lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
        size = len(full_coords)
        dist_mat = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                dist_mat[i, j] = lengths[i][j]
        
        return dist_mat, full_coords, G

    def _build_graph(self, coords):
        road_cfg = self.config["road_network"]
        eucl = distance_matrix(coords, coords)
        mst = minimum_spanning_tree(eucl).toarray()
        
        G = nx.Graph()
        for i, j in zip(*np.where(mst > 0)):
            detour = self.rng.uniform(road_cfg["detour_min"], road_cfg["detour_max"])
            G.add_edge(i, j, weight=mst[i, j] * detour)
            
        # Add extra edges for redundancy
        possible = [(i, j) for i in range(len(coords)) for j in range(i+1, len(coords)) if not G.has_edge(i, j)]
        n_extra = int(len(G.edges) * road_cfg["extra_edge_ratio"])
        if possible and n_extra > 0:
            for idx in self.rng.choice(len(possible), size=min(n_extra, len(possible)), replace=False):
                i, j = possible[idx]
                detour = self.rng.uniform(road_cfg["detour_min"], road_cfg["detour_max"])
                G.add_edge(i, j, weight=eucl[i, j] * detour)
        return G

    def plot_graph(self, G, coords, instance_id):
        plot_cfg = self.general_config.get("plot", {})
        if not plot_cfg.get("save") and not plot_cfg.get("show"):
            return

        plt.figure(figsize=(10, 10))
        
        # Draw edges (Roads)
        for (i, j) in G.edges:
            plt.plot(
                [coords[i, 0], coords[j, 0]], 
                [coords[i, 1], coords[j, 1]], 
                color='gray', linestyle='-', alpha=0.4, zorder=1
            )

        # Draw Wells (Index 1 onwards)
        plt.scatter(
            coords[1:, 0], coords[1:, 1], 
            c='blue', s=30, label='Wells', zorder=2
        )

        # Draw Operations Center (Index 0)
        plt.scatter(
            coords[0, 0], coords[0, 1], 
            c='red', marker='s', s=100, label='Ops Center', zorder=3
        )

        plt.title(f"Oilfield Infrastructure - Instance {instance_id}")
        plt.xlabel("Meters (X)")
        plt.ylabel("Meters (Y)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.axis("equal")

        if plot_cfg.get("save"):
            out_dir = Path(self.general_config.get("output_dir", "instances"))
            out_dir.mkdir(parents=True, exist_ok=True)
            save_path = out_dir / f"spatial_map_{instance_id}.png"
            plt.savefig(save_path, dpi=300)
            print(f"Map saved to: {save_path}")

        if plot_cfg.get("show"):
            plt.show()

        plt.close()


class BatteryGenerator:
    def __init__(self, rng, config):
        self.rng = rng
        self.config = config

    def generate(self, well_data):
        n_wells = self.config["general"]["n_wells"]
        n_bats = self.config["general"]["n_batteries"]
        noise_std = self.config["batteries"]["noise_std"]
        rounding = self.config["rounding"].get("total_gross", 0)

        # 1. Assign Wells to Batteries
        reps = int(np.ceil(n_wells / n_bats))
        battery_ids = np.repeat(np.arange(1, n_bats + 1), reps)[:n_wells]
        
        # 2. Compute G_t (Sum + Noise)
        battery_targets = []
        for b_id in range(1, n_bats + 1):
            mask = (battery_ids == b_id)
            total_gross = np.sum(well_data["G"][mask])
            noise = self.rng.normal(0, noise_std)
            g_t = total_gross * (1 + noise)
            if rounding == 0:
                battery_targets.append(int(round(g_t)))
            else:
                battery_targets.append(round(g_t, rounding))
            
        return battery_ids, battery_targets


class ZPLGenerator:
    def __init__(self, config):
        self.config = config

    def generate(self, output_path, param_file, bat_file, dist_file):
        # Extract constants from config
        limits = self.config.get("limits", {})
        res = self.config.get("resources", {})
        
        # Zimpl content with dynamic file paths
        zpl_content = f"""# Auto-generated Zimpl model for {param_file}

# Sets and Data Files
set P := {{ read "{param_file}" as "<1n>" skip 1 }};
set B := {{ read "{bat_file}" as "<1n>" skip 1 }};

# Parameters
param G[P]   := read "{param_file}" as "<1n> 2n" skip 1;
param N[P]   := read "{param_file}" as "<1n> 3n" skip 1;
param R[P]   := read "{param_file}" as "<1n> 4n" skip 1;
param C[P]   := read "{param_file}" as "<1n> 7n" skip 1;
param Bat[P] := read "{param_file}" as "<1n> 8n" skip 1;

param D[P*P] := read "{dist_file}" as "n+" skip 1;
param Gpt[B] := read "{bat_file}" as "<1n> 2n" skip 1;

# Bounds from YAML
param maxloss := {limits.get('max_loss', 500)};
param maxcost := {limits.get('max_cost', 900)};
param maxquantity := {limits.get('max_quantity', 9)};
param crews := {res.get('crews', 1)};

# Variables
var newregime[P] >= 0 <= 100;
var z[P] binary;
var x[P*P] binary;
var y[P] >= 0 <= card(P);

# Objectives
var distance >= 0;
var loss >= 0;

# Objective Function
minimize objfunc: distance;

# Constraints
subto lossbound: loss <= maxloss;
subto costbound: cost <= maxcost;
subto quantitybound: (sum <i> in P: z[i]) <= maxquantity;

subto grossproduction: forall <k> in B:
    sum <i> in P with i > 0 and Bat[i] == k: G[i] * newregime[i] / 100 == Gpt[k];

subto linkzwupper: forall <i> in P with i > 0:
    newregime[i] >= R[i] - 100 * z[i];
subto linkzwlower: forall <i> in P with i > 0:
    newregime[i] <= R[i] + 100 * z[i];

subto routeentry: forall <i> in P with i > 0:
    sum <j> in P: x[j,i] == z[i];
subto routeexit: forall <i> in P with i > 0:
    sum <j> in P: x[i,j] == z[i];

subto departure: sum <i> in P with i > 0: x[0,i] == crews;
subto arrival:   sum <i> in P with i > 0: x[i,0] == crews;

subto flow: forall <i> in P with i > 0:
    sum <j> in P: x[j,i] == sum <j> in P: x[i,j];

subto mtz: forall <i,j> in P*P with i != j and i > 0 and j > 0:
    y[i] + 1 <= y[j] + card(P) * (1 - x[i,j]);

subto defdistance: distance == sum <i,j> in P*P: D[i,j] * x[i,j];
subto defloss: loss == sum <i> in P: (G[i] - N[i]) * newregime[i] / 100;

subto nonsensex: forall <i> in P: x[i,i] == 0;
"""
        with open(output_path, "w") as f:
            f.write(zpl_content)