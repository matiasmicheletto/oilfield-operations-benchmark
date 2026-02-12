#!/usr/bin/env python3

"""
generate_instances.py

Generates synthetic oilfield optimization instances based on a YAML configuration file.
Supports dynamic probability distribution definitions.

Usage:
    python generate_instances.py config.yaml
"""

import sys
import numpy as np
from pathlib import Path
from util import load_config, create_rng


# ============================================================
# Utility
# ============================================================

def process_array(arr, decimals, scale=1):
    """
    Apply rounding and optional scaling.
    """
    if decimals > 0:
        return np.round(arr, decimals)
    else:
        return np.round(arr * scale).astype(int)


# ============================================================
# Dynamic Sampling Engine
# ============================================================

def sample_distribution(rng, n, config):
    """
    Generic sampler that calls numpy functions dynamically based on config.
    
    Expects config to have:
      distribution: string (e.g., 'uniform', 'lognormal', 'beta')
      ...params: arguments matching the numpy function (e.g., low, high, mu, sigma)
    """
    # 1. Extract distribution name
    dist_name = config.get("distribution", "uniform")
    
    # 2. Extract parameters (everything except 'distribution')
    params = {k: v for k, v in config.items() if k != "distribution"}
    
    # 3. Parameter Mapping (Quality of Life improvements)
    # Numpy uses 'mean' for lognormal, but users often write 'mu'
    if dist_name == "lognormal" and "mu" in params:
        params["mean"] = params.pop("mu")
    
    # Numpy's beta uses 'a'/'b', user might use 'alpha'/'beta'
    if dist_name == "beta":
        if "alpha" in params: params["a"] = params.pop("alpha")
        if "beta" in params: params["b"] = params.pop("beta")

    # 4. Call the numpy function
    # e.g., rng.uniform(low=0, high=10, size=n)
    try:
        generator = getattr(rng, dist_name)
        return generator(size=n, **params)
    except AttributeError:
        raise ValueError(f"Unknown distribution '{dist_name}'. Check NumPy documentation.")
    except TypeError as e:
        raise TypeError(f"Invalid parameters for '{dist_name}': {params}. Error: {e}")


# ============================================================
# Specialized Logic (Correlated Generation)
# ============================================================

def generate_correlated_variable(rng, base_array, config):
    """
    Legacy logic: Generates a variable correlated with a base array (Gross or Net).
    Used if 'distribution' is NOT specified in the config section.
    """
    rho = config.get("rho", 0.0)
    noise_std = config.get("noise_std", 0.05)
    
    # Normalize base array
    base_norm = base_array / (np.max(base_array) + 1e-9)
    
    noise = rng.normal(0, noise_std, size=len(base_array))
    
    # Formula: Val = rho * base + noise
    generated = rho * base_norm + noise
    
    # Default clip to [0,1] for risk/priority
    return np.clip(generated, 0, 1)


def compute_cost(R, P, w_r, w_p, kappa):
    return kappa + w_r * R + w_p * (1 - P)


def assign_batteries(n_wells, n_batteries):
    reps = int(np.ceil(n_wells / n_batteries))
    batteries = np.repeat(np.arange(1, n_batteries + 1), reps)
    return batteries[:n_wells]


# ============================================================
# Instance Generation
# ============================================================

def generate_instance(config, rng):

    general = config["general"]
    rounding = config.get("rounding", {})
    scaling = config.get("scaling", {})

    n_wells = general["n_wells"]
    n_batteries = general["n_batteries"]

    # ----- 1. Production Generation (Dynamic)
    prod_cfg = config["production"]
    
    # Gross
    G_raw = sample_distribution(rng, n_wells, prod_cfg["gross"])
    
    # Net (Ratio * Gross)
    net_ratio = sample_distribution(rng, n_wells, prod_cfg["net_ratio"])
    N_raw = G_raw * net_ratio
    
    # Regime
    r_raw = sample_distribution(rng, n_wells, prod_cfg["regime"])

    # ----- 2. Risk & Priority (Dynamic OR Correlated)
    rp_cfg = config["risk_priority"]
    
    # Risk Generation
    if "distribution" in rp_cfg["risk"]:
        # Independent distribution defined in yaml
        R_raw = sample_distribution(rng, n_wells, rp_cfg["risk"])
    else:
        # Legacy: Correlated with Gross
        # Expects: rho, noise_std
        R_raw = generate_correlated_variable(rng, G_raw, rp_cfg["risk"])

    # Priority Generation
    if "distribution" in rp_cfg["priority"]:
        # Independent distribution defined in yaml
        P_raw = sample_distribution(rng, n_wells, rp_cfg["priority"])
    else:
        # Legacy: Correlated with Net
        P_raw = generate_correlated_variable(rng, N_raw, rp_cfg["priority"])

    # Cost Calculation
    cost_cfg = rp_cfg["cost"]
    C_raw = compute_cost(R_raw, P_raw, cost_cfg["w_r"], cost_cfg["w_p"], cost_cfg["kappa"])

    # ----- 3. Scaling & Rounding
    
    # Defaults
    dec_prod = rounding.get("production", 2)
    dec_reg = rounding.get("regime", 2)
    dec_risk = rounding.get("risk", 2)
    dec_prio = rounding.get("priority", 2)
    dec_cost = rounding.get("cost", 2)

    scale_reg = scaling.get("regime", 1)
    scale_risk = scaling.get("risk", 1)
    scale_prio = scaling.get("priority", 1)
    scale_cost = scaling.get("cost", 1)

    G = process_array(G_raw, dec_prod)
    N = process_array(N_raw, dec_prod)
    r = process_array(r_raw, dec_reg, scale_reg)
    R = process_array(R_raw, dec_risk, scale_risk)
    P = process_array(P_raw, dec_prio, scale_prio)
    C = process_array(C_raw, dec_cost, scale_cost)

    B = assign_batteries(n_wells, n_batteries)

    return G, N, r, R, P, C, B


# ============================================================
# Main
# ============================================================

def main(config):

    general = config["general"]
    num_instances = general.get("num_instances", 1)
    seed = general.get("seed", None)
    output_dir = Path(general.get("output_dir", "instances"))
    output_prefix = general.get("output_prefix", "instance")
    n_wells = general["n_wells"]
    n_batteries = general["n_batteries"]

    output_dir.mkdir(parents=True, exist_ok=True)

    rng = create_rng(seed)

    for k in range(1, num_instances + 1):
        G, N, r, R, P, C, B = generate_instance(config, rng)
        
        filename = output_dir / f"{output_prefix}_{n_wells}_{n_batteries}_{k}.dat"
        
        with open(filename, "w") as f:
            f.write("ID\tG\tN\tr\tR\tP\tC\tB\n")
            for i in range(n_wells):
                f.write(
                    f"{i+1}\t{G[i]}\t{N[i]}\t{r[i]}\t"
                    f"{R[i]}\t{P[i]}\t{C[i]}\t{B[i]}\n"
                )
        print(f"Generated: {filename}")

    print(f"\nAll {num_instances} instances generated successfully.")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        config_path = sys.argv[1]
    else:
        print("No configuration file provided. Using 'default_config.yaml'.")
        config_path = "default_config.yaml"

    config = load_config(config_path)
    main(config)