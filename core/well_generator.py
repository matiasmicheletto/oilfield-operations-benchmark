import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


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
