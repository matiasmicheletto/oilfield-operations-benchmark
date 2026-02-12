# Oilfield Instance Generator

A Python-based utility for generating synthetic oilfield datasets for MILP (Mixed-Integer Linear Programming) optimization and benchmarking. This tool provides two complementary scripts: one for generating well production and operational data, and another for creating realistic spatial configurations with road networks.

## 🚀 Features

### Instance Generation (`generate_instances.py`)
* **Dynamic Distributions:** Define any NumPy-supported distribution (Uniform, Beta, LogNormal, etc.) for well properties via YAML.
* **Correlated Variables:** Generate Risk and Priority values that correlate with production levels to simulate realistic field data.
* **Automatic Scaling & Rounding:** Built-in support for scaling values (e.g., 0–1 to 0–100) and rounding to integers for MILP solver compatibility.
* **Reproducibility:** Seed-based random number generation ensures the same instances can be recreated for research consistency.

### Spatial Network Generation (`generate_distance_matrix.py`)
* **Clustered Well Placement:** Generates realistic spatial distributions with configurable clustering parameters.
* **Road Network Synthesis:** Creates road networks based on Minimum Spanning Tree (MST) with detour factors and optional extra connections.
* **Distance Matrix Output:** Computes and exports shortest-path distance matrices for routing optimization.
* **Visualization:** Optional plotting of the spatial configuration and road network.
* **Operations Center Placement:** Configurable placement of central operations hub.

---

## 🛠 Installation

1. **Requirements:**
   * Python 3.8+
   * `numpy`
   * `PyYAML`
   * `networkx`
   * `scipy`
   * `matplotlib`

2. **Setup:**
   ```bash
   pip install numpy pyyaml networkx scipy matplotlib
   ```
   
   Or if you have a requirements.txt:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📖 Usage

### Generating Well Instances

To generate well production and operational data:

```bash
python generate_instances.py config.yaml
```

If no configuration file is provided, the script defaults to `default_config.yaml`:

```bash
python generate_instances.py
```

The script will create an output directory (as specified in the YAML) containing `.dat` files with the following header:

```
ID    G    N    r    R    P    C    B
```

### Generating Spatial Networks

To generate spatial well locations and road networks:

```bash
python generate_distance_matrix.py config.yaml
```

Similarly, if no configuration is provided:

```bash
python generate_distance_matrix.py
```

The script will:
1. Generate clustered well locations within the specified area
2. Create a road network connecting wells and the operations center
3. Compute shortest-path distances between all locations
4. Save the distance matrix to the specified output file
5. Optionally display and/or save a visualization of the network

---

## ⚙️ Configuration Guide

All configuration is done through a single YAML file. Below is a comprehensive breakdown of all sections.

### General Settings

Controls overall instance generation parameters:

```yaml
general:
  num_instances: 10      # Number of instances to generate
  seed: 42               # Random seed (use null for non-deterministic)
  output_dir: "instances"
  output_prefix: "instance"
  n_wells: 50            # Number of wells in each instance
  n_batteries: 2         # Number of production batteries
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
  output_dir: "distances"
  output_file_name: "distance_matrix.dat"
  
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

#### Visualization

```yaml
  plot:
    save: false   # Save plot to file
    show: true    # Display plot interactively
```

---

## 📂 Output Formats

### Instance Data Files (`.dat`)

Generated by `generate_instances.py`. Tab-separated files with the following columns:

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

**File naming convention:**
```
{output_prefix}_{n_wells}_{n_batteries}_{instance_number}.dat
```

Example: `instance_50_2_1.dat`

### Distance Matrix Files (`.dat`)

Generated by `generate_distance_matrix.py`. First line is the matrix dimension (including operations center), followed by the distance matrix:

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

### Visualization Output

If `plot.save: true`, generates `road_network.png` showing:
- Well locations (blue circles)
- Operations center (orange square)
- Road network edges (gray lines)
- Scale in meters

---

## 🔧 Advanced Usage

### Custom Distributions

The dynamic sampling engine supports any NumPy random distribution. Examples:

```yaml
# Exponential
distribution: exponential
scale: 50

# Gamma
distribution: gamma
shape: 2
scale: 20

# Triangular
distribution: triangular
left: 0
mode: 50
right: 100

# Pareto
distribution: pareto
a: 3
```

**Parameter mapping:** The script automatically maps common aliases:
- `mu` → `mean` (for lognormal)
- `alpha`/`beta` → `a`/`b` (for beta distribution)

### Generating Multiple Scenarios

Create different configuration files for different scenarios:

```bash
python generate_instances.py config_high_risk.yaml
python generate_instances.py config_low_production.yaml
python generate_distance_matrix.py config_dense_network.yaml
```

### Integration Workflow

Typical workflow for optimization studies:

1. **Generate spatial configuration:**
   ```bash
   python generate_distance_matrix.py config.yaml
   ```
   Creates `distances/distance_matrix.dat`

2. **Generate instance data:**
   ```bash
   python generate_instances.py config.yaml
   ```
   Creates `instances/instance_50_2_1.dat`, etc.

3. **Use outputs in optimization model:**
   - Load well data from `.dat` files
   - Load distance matrix for routing constraints
   - Apply limits and resources from config

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

## 🐛 Troubleshooting

### "Unknown distribution" error
- Check that the distribution name matches NumPy's random API
- Common distributions: `uniform`, `normal`, `lognormal`, `beta`, `gamma`, `exponential`

### "Invalid parameters" error
- Verify parameter names match NumPy requirements
- Example: `normal` uses `loc` and `scale`, not `mean` and `std`

### Distance matrix seems incorrect
- Check that `area_size_km` and `cluster_radius_m` are reasonable
- Verify detour factors are ≥ 1.0
- Ensure `n_clusters` ≤ `n_wells`

### Negative or invalid costs
- Verify `w_r`, `w_p`, and `kappa` are positive
- Check scaling factors if using [0,1] range for R and P
- Adjust weights if costs exceed budget constraints

---

## 📝 License

General Public License v3.0 (GPL-3.0)

## 🤝 Contributing

Contributions are welcome! Please submit issues or pull requests on GitHub.

## 📧 Contact

For questions or support, please contact the maintainer at matias.micheletto@uns.edu.ar

---

## 🔍 Example Configuration

Here's a complete working example for a medium-scale oilfield study:

```yaml
general:
  num_instances: 20
  seed: 42
  output_dir: "instances"
  output_prefix: "field_study"
  n_wells: 100
  n_batteries: 3

rounding:
  production: 0
  regime: 0
  risk: 0
  priority: 0
  cost: 0

scaling:
  risk: 100
  priority: 100
  cost: 1
  regime: 100

production:
  gross:
    distribution: lognormal
    mu: 4.0
    sigma: 0.7
  net_ratio:
    distribution: beta
    alpha: 2
    beta: 3
  regime:
    distribution: beta
    alpha: 4
    beta: 2

risk_priority:
  risk:
    rho_g: 0.8
    noise_std: 0.05
  priority:
    rho: 0.7
    noise_std: 0.05
  cost:
    w_r: 10
    w_p: 5
    kappa: 2

limits:
  max_loss: 1000
  max_cost: 2000
  max_quantity: 15

resources:
  crews: 3

spatial:
  output_dir: "distances"
  output_file_name: "distance_matrix.dat"
  area_size_km: 25
  n_clusters: 5
  cluster_radius_m: 500
  road_network:
    extra_edge_ratio: 0.2
    detour_min: 1.2
    detour_max: 1.8
  plot:
    save: true
    show: false
  operations_center:
    location: "random_cluster"
```

This configuration generates 20 instances of a 100-well field with realistic production distributions, correlated risk/priority values, and a spatially realistic road network.
