# Oilfield Instance Generator & Packager

A modular Python library for generating synthetic oilfield datasets tailored for MILP (Mixed-Integer Linear Programming) optimization. This tool consolidates statistical well generation, battery requirement modeling, and spatial road network synthesis into a single, automated pipeline.

## 🚀 Features

### 1. Well Generator (`WellGenerator`)
* **Dynamic Distributions:** Configure LogNormal, Beta, or any NumPy distribution via YAML.
* **Correlated Variables:** Realistic Risk and Priority values tied to production levels.
* **Visual Diagnostics:** Automatically generates distribution dashboards (Histograms/Bar charts) to verify statistical integrity.

### 2. Spatial & Road Network (`SpatialGenerator`)
* **Clustered Placement:** Simulates realistic field geography.
* **Graph-Based Roads:** Built on Minimum Spanning Trees (MST) with configurable detour factors and redundancy edges.
* **Shortest-Path Matrices:** Computes full routing distances between the Operations Center and all wells.

### 3. Battery Logic (`BatteryGenerator`)
* **Smart Assignment:** Evenly distributes wells to production batteries.
* **Noisy Targets:** Calculates total battery requirements ($G_t$) by summing well production and adding Gaussian noise.

### 4. Zimpl Model Generator (`ZPLGenerator`)
* **Solver Ready:** Automatically writes `.zpl` files with relative paths to data files.
* **Integrated Constraints:** Injects YAML limits (max loss, cost, crews) directly into the model.

---

## 📁 Library Structure
```
oilfield-operations-benchmark/
├── core.py              # Main logic for data generation and visualization
├── main.py              # Entry point for running the packager
├── LICENSE              # License information (GNU GPL v3)
├── requirements.txt     # Python dependencies
├── default_config.yaml  # Default configuration for instance generation
└── README.md            # This documentation
```

### Setup
Ensure you have the required dependencies:

`pip install numpy pyyaml networkx scipy matplotlib`

Or install from [`requirements.txt`](requirements.txt):

`pip install -r requirements.txt`

### 📖 Usage
To generate the entire dataset (Parameters, Batteries, Distances, Zimpl Models, and Plots):

```bash
python main.py [config.yaml]
```

If no configuration is provided, it defaults to [`default_config.yaml`](default_config.yaml).


## ⚙️ Configuration Guide

All configuration is done through a single YAML file. Below is a comprehensive breakdown of all sections.

### General Settings

Controls overall instance generation parameters:

```yaml
general:
  num_instances: 10      # Number of instances to generate
  seed: 42               # Random seed (use null for non-deterministic)
  output_dir: "instances"
  param_name_prefix: "parameters"  # Prefix for generated file names.
  bat_name_prefix: "batteries"   # Prefix for generated file names.
  dist_name_prefix: "distance" # File name for distance matrix output.
  zpl_name_prefix: "model"     # Prefix for generated ZPL files.
  n_wells: 50            # Number of wells in the instance.
  n_batteries: 2         # Number of production batteries.
  plot:
    save: true  # Whether to save the plot of the spatial configuration.
    show: false # Whether to display the plot interactively.
```

**Key Parameters:**
- `num_instances`: Recommended 5–20 for testing, 20–100 for benchmarking
- `n_wells`: Problem size (Small: 20–50, Medium: 100–300, Large: 500+)
- `n_batteries`: Wells are assigned evenly across batteries (typically 1–5)

### Rounding

Number of decimal places for each variable (0 = integers):

```yaml
rounding:
  production: 0   # Gross and Net production
  regime: 0       # Production regime
  risk: 0         # Risk factor
  priority: 0     # Priority index
  cost: 0         # Intervention cost
  total_gross: 0   # Total gross production per battery
```

**Recommendations:**
- Use `0` for MILP models requiring integer conditioning
- Use `2` or higher for continuous/realistic values

### Scaling

Multipliers applied after generation to improve MILP numerical conditioning:

```yaml
scaling:
  risk: 100       # Converts [0,1] → [0,100]
  priority: 100
  cost: 1
  regime: 100
```

### Production Distributions

Define probability distributions for production-related variables:

#### Gross Production
```yaml
production:
  gross:
    distribution: lognormal
    mu: 3.8       # Location parameter (log scale)
    sigma: 0.6    # Dispersion parameter
```

**Alternative distributions:**
```yaml
# Uniform distribution
gross:
  distribution: uniform
  low: 50
  high: 150

# Normal distribution
gross:
  distribution: normal
  loc: 100
  scale: 20
```

#### Net Production Ratio
```yaml
  net_ratio:
    distribution: beta
    alpha: 2      # Shape parameter α
    beta: 3       # Shape parameter β
```

The actual net production is computed as: **N = G × ratio**

#### Production Regime
```yaml
  regime:
    distribution: beta
    alpha: 4
    beta: 2
```

### Risk & Priority

Two modes are supported for generating Risk and Priority:

#### Mode 1: Independent Distributions
```yaml
risk_priority:
  risk:
    distribution: beta
    alpha: 2
    beta: 5
  
  priority:
    distribution: uniform
    low: 0
    high: 1
```

#### Mode 2: Correlated with Production (Legacy)

If `distribution` is omitted, values are correlated with production:

```yaml
risk_priority:
  risk:
    rho_g: 0.8        # Correlation with Gross production
    noise_std: 0.05   # Gaussian noise level
  
  priority:
    rho: 0.7          # Correlation with Net production
    noise_std: 0.05
```

**Correlation formula:**
$$\text{Value} = \text{clip}(\rho \cdot \text{NormalizedProduction} + \mathcal{N}(0, \sigma_{\text{noise}}), 0, 1)$$

### Cost Model

Intervention costs are computed from Risk (R) and Priority (P):

$$C = \kappa + w_r \cdot R + w_p \cdot (1 - P)$$

```yaml
risk_priority:
  cost:
    w_r: 10       # Weight of risk
    w_p: 5        # Weight of (1 - priority)
    kappa: 2      # Base/fixed cost
```

**Note:** If R and P are scaled (e.g., to 0–100), adjust weights accordingly.

### Limits & Resources

Operational constraints (used in optimization, not generation):

```yaml
limits:
  max_loss: 500       # Maximum production loss allowed
  max_cost: 900       # Budget constraint
  max_quantity: 9     # Max wells that can be selected

resources:
  crews: 2            # Number of intervention crews
```

### Spatial Configuration

Controls well placement and road network generation:

```yaml
spatial:
  area_size_km: 20          # Square area side length (km)
  n_clusters: 4             # Number of well clusters
  cluster_radius_m: 600     # Std dev within clusters (meters)
```

**Spatial Distribution Parameters:**
- `area_size_km`: Larger values → more spread out wells
- `n_clusters`: 
  - `1` → all wells in one cluster (high spatial correlation)
  - `n_wells` → each well separate (no spatial correlation)
- `cluster_radius_m`: Smaller → tighter clusters

#### Road Network

```yaml
  road_network:
    extra_edge_ratio: 0.2   # Extra edges beyond MST (0–1)
    detour_min: 1.2         # Min road/Euclidean ratio
    detour_max: 1.8         # Max road/Euclidean ratio
```

**Road Network Algorithm:**
1. Generate Minimum Spanning Tree (MST) connecting all locations
2. Apply detour factors to simulate non-straight roads
3. Add extra edges based on `extra_edge_ratio` for network redundancy

**Detour factors:**
- `1.0` = perfectly straight roads
- `1.2–1.8` = realistic road networks
- Higher values = more winding/indirect routes

#### Operations Center

```yaml
  operations_center:
    location: "random_cluster"  # or "center"
```

Options:
- `random_cluster`: Place at a random cluster centroid (realistic)
- `center`: Place at geometric center of area (less realistic)

---

---
## 📂 Output Structure
For each instance from 1 to `num_instances`, the following files are generated in the `instances/` directory:
```
instances/
├── parameters_[num_wells]_[num_batteries]_[instance_id].dat/
├── batteries_[num_wells]_[num_batteries]_[instance_id].dat/
├── distance_[num_wells]_[num_batteries]_[instance_id].dat
├── model_[num_wells]_[num_batteries]_[instance_id].zpl
├── param_hist_[instance_id].png
├── spatial_map_[instance_id].png
└── well_stats_[instance_id].png
```

### Parameters Data Files (`.dat`)

Tab-separated files with the following columns:

| Column | Description | Typical Range |
|--------|-------------|---------------|
| `ID` | Unique well identifier | 1, 2, 3, ... |
| `G` | Gross Production | 0–500+ |
| `N` | Net Production | 0–400+ |
| `r` | Production Regime | 0–100 |
| `R` | Risk Factor | 0–100 |
| `P` | Priority Index | 0–100 |
| `C` | Intervention Cost | 0–1000+ |
| `B` | Battery Assignment | 1, 2, ... |

**Example output:**
```
ID	G	N	r	R	P	C	B
1	156	89	72	45	67	487	1
2	203	121	81	58	72	612	2
3	98	54	43	31	45	389	1
...
```

### Distance Matrix File
A space-separated matrix where the entry at row `i` and column `j` represents the shortest road distance from the Operations Center to well `j`.

```
51
0 1245 2103 1876 ...
1245 0 987 1543 ...
2103 987 0 1234 ...
...
```

**Format details:**
- First row/column (index 0) represents the operations center
- Remaining rows/columns represent wells (1 through n_wells)
- Values are shortest-path distances in meters (rounded to integers)
- Matrix is symmetric

### Battery File
A tab-separated file with two columns:
| Column | Description |
|--------|-------------|
| `BatteryID` | Unique battery identifier (1, 2, ...) |
| `Target` | Total production target for the battery (sum of assigned wells + noise)


---

## 📊 Best Practices

### For Research Reproducibility
- Always use a fixed `seed` value
- Document your configuration file in version control
- Include generated instances in supplementary materials

### For MILP Models
- Set `rounding` to `0` for all variables
- Use `scaling` to avoid small coefficients (e.g., 0.001)
- Scale risk/priority to 0–100 range

### For Realistic Scenarios
- Use `lognormal` for production (realistic heavy-tailed distribution)
- Set correlation `rho` between 0.5–0.9 for risk/priority
- Use 2–5 clusters with radius 300–1000m
- Set detour factors between 1.2–1.8

### For Benchmarking
- Generate 20–100 instances per configuration
- Vary `n_wells` systematically (e.g., 20, 50, 100, 200, 500)
- Test both clustered and dispersed spatial configurations
- Include both high and low correlation scenarios

---

## 📝 License

General Public License v3.0 (GPL-3.0)

## 🤝 Contributing

Contributions are welcome! Please submit issues or pull requests on GitHub.

## 📧 Contact

For questions or support, please contact the maintainer at matias.micheletto@uns.edu.ar

---