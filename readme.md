# Oilfield Instance Generator & Packager

A modular Python library for generating synthetic oilfield datasets tailored for MILP (Mixed-Integer Linear Programming) optimization. This tool consolidates statistical well generation, battery requirement modeling, and spatial road network synthesis into a single, automated pipeline.

## 🚀 Features

### 1. Well Generator (`WellGenerator`)
* **Dynamic Distributions:** Configure LogNormal, Beta, or any NumPy distribution via YAML.
* **Correlated Variables:** Realistic Risk and Priority values tied to production levels.
* **Visual Diagnostics:** Automatically generates distribution dashboards (Histograms/Bar charts) to verify statistical integrity.

### 2. Spatial & Road Network (`SpatialGenerator`)
* **Clustered Placement:** Simulates realistic field geography.
* **Graph-Based Roads:** Builds primary well-to-operations-center roads and secondary well-to-well connectors with corridor reuse.
* **Shortest-Path Matrices:** Computes full routing distances between the Operations Center and all wells.

### 3. Battery Logic (`BatteryGenerator`)
* **Smart Assignment:** Evenly distributes wells to production batteries.
* **Noisy Targets:** Calculates total battery requirements ($G_t$) by summing well production and adding Gaussian noise.

### 4. Zimpl Model Generator (`ZPLGenerator`)
* **Solver Ready:** Automatically writes `.zpl` files with relative paths to data files.
* **Integrated Constraints:** Injects YAML limits (max loss, cost, crews) directly into the model.

### 5. LP Model Generator (`LPGenerator`)
* **Zimpl-Free:** Writes standard LP-format files directly from instance data — no `zimpl` binary required.
* **SCIP/CPLEX Compatible:** Generated `.lp` files can be fed directly to SCIP, CPLEX, or any LP-compatible solver.

---

## 📁 Repository Structure
```
oilfield-operations-benchmark/
├── run_pipeline.sh             # End-to-end pipeline script
├── generator/
│   ├── main.py                 # Entry point for instance generation
│   ├── generator_config.yaml   # Default configuration
│   ├── requirements.txt        # Python dependencies
│   ├── instances/              # Generated instance files (output)
│   └── core/
│       ├── battery_generator.py  # Battery assignment and target generation
│       ├── lp_generator.py       # LP model file writer (zimpl-free)
│       ├── spatial_generator.py  # Terrain, well placement, road network, distances
│       ├── well_generator.py     # Well parameter sampling and visualization
│       └── zpl_generator.py      # Zimpl (.zpl) model file writer
├── solver/
│   ├── Makefile
│   ├── solver_config.yaml
│   ├── include/
│   │   ├── loader.h
│   │   ├── models.h
│   │   ├── solver.h
│   │   └── utils.h
│   └── src/
│       ├── loader.cpp
│       ├── solve_main.cpp
│       ├── solver.cpp
│       └── utils.cpp
├── output/
│   ├── compare_solutions.py    # Solution comparison entry point
│   ├── compare/
│   │   ├── parsers.py          # Parsers for CPLEX, SCIP, and greedy output
│   │   ├── matching.py         # Stem-based file matching across solver dirs
│   │   └── reporting.py        # Console table and CSV output
│   ├── cplex/                  # CPLEX .sol output files
│   ├── greedy/                 # Greedy solver .txt output files
│   └── scip/                   # SCIP .txt output files
├── LICENSE
└── readme.md
```

### Setup
Ensure you have the required dependencies:

`pip install numpy pyyaml networkx scipy matplotlib scikit-image`

Or install from [`requirements.txt`](requirements.txt):

`pip install -r requirements.txt`

### 📖 Usage

#### Full Pipeline

Run the entire pipeline (generate instances → solve with SCIP/CPLEX → solve with greedy):

```bash
bash run_pipeline.sh [options]
```

**Options:**

| Flag | Long form | Description | Default |
|------|-----------|-------------|---------|
| `-n N` | `--instances N` | Number of instances to generate | 2 |
| `-w N` | `--wells N` | Number of wells per instance | 10 |
| `-b N` | `--batteries N` | Number of batteries per instance | 2 |
| `-d` | `--dry-run` | Print commands without executing | false |
| `-h` | `--help` | Show help message | — |

**Examples:**

```bash
bash run_pipeline.sh -w 25 -b 2 -n 5
bash run_pipeline.sh --wells 100 --batteries 3 --instances 10
bash run_pipeline.sh -w 10 -n 2 --dry-run
```

**Pipeline steps:**
1. Generate instances (`.dat`, `.zpl`, `.lp` files)
2. Convert ZPL → LP via `zimpl` (skipped if `zimpl` not in PATH — LP files already written in step 1)
3. Solve LP models with CPLEX (skipped if `cplex` not in PATH)
4. Solve LP models with SCIP (skipped if `scip` not in PATH)
5. Solve instances with greedy heuristic

#### Generator Only

To generate instances without running solvers:

```bash
cd generator
python main.py [config.yaml]
```

You can override parameters from the CLI using dot notation:

```bash
python main.py generator_config.yaml \
  --set general.num_instances=5 \
  --set general.n_wells=50 \
  --set spatial.num_peaks=10
```

#### Compare Solutions

After running the pipeline, compare solver outputs:

```bash
cd output
python compare_solutions.py [--cplex DIR] [--greedy DIR] [--scip DIR] [--csv FILE]
```

All directory arguments default to `output/cplex/`, `output/greedy/`, and `output/scip/`. CPLEX and SCIP directories are optional — the table layout adapts automatically to whichever solvers were run.

```bash
# Greedy only
python compare_solutions.py

# Greedy + SCIP with CSV export
python compare_solutions.py --scip scip/ --csv results.csv
```


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
  save_plot: true        # Global flag to save figures.
  show_plot: false       # Global flag to show figures interactively.
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

### Battery Noise

Controls the noise applied to battery production targets:

```yaml
batteries:
  noise_std: 0.1  # Gaussian noise (as a fraction) added to the sum of assigned
                  # well production when computing each battery's target G_t.
                  # 0.0 → deterministic target (exact sum)
                  # 0.1 → ±10% typical deviation
```

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

Controls terrain generation, well placement, road-network growth, and plotting:

```yaml
spatial:
  grid_size: 300
  n_clusters: 3

  # Terrain morphology
  seed_resolution: 20
  smooth_sigma: 2
  num_peaks: 5
  elevation_amplitude: 100
  cost_exponent: 2

  # Secondary roads (well-to-well connectors)
  max_dist_fraction: 0.25
  min_per_well: 2
  max_per_well: 3
  max_path_factor: 1.8
  connector_reuse_penalty: 3.0
```

**Notes:**
- Terrain is generated from low-resolution random fields plus configurable Gaussian peaks (`num_peaks`, `elevation_amplitude`).
- Roads are built with `MCP_Geometric`: primary links connect wells to the operations center, then secondary well-to-well connectors add loops and route alternatives.
- Secondary connectors can be tuned with `max_dist_fraction`, `min_per_well`, `max_per_well`, and `max_path_factor`.
- `connector_reuse_penalty` discourages connector paths from fully collapsing into primary roads.
- The Operations Center is placed randomly near the geometric centre of the grid (within ±`grid_size/8` of centre) to simulate a central depot.

---

---
## 📂 Output Structure

### Instance files (`generator/instances/`)
For each instance from 1 to `num_instances`:
```
instances/
├── parameters_[W]_[B]_[id].dat      # Well parameters
├── batteries_[W]_[B]_[id].dat       # Battery targets
├── distance_[W]_[B]_[id].dat        # Distance matrix
├── model_[W]_[B]_[id].zpl           # Zimpl model (requires zimpl to solve)
├── model_[W]_[B]_[id].lp            # LP model (direct SCIP/CPLEX input)
├── well_bar_[id].png
├── well_hist_[id].png
└── spatial_network_[id].png
```

### Solver output (`output/`)
```
output/
├── cplex/greedy_[W]_[B]_[id].sol    # CPLEX XML solution files
├── greedy/greedy_[W]_[B]_[id].txt   # Greedy heuristic solution files
└── scip/model_[W]_[B]_[id].txt      # SCIP solution files
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
| `ID` | Unique battery identifier (1, 2, ...) |
| `Gpt` | Total gross-production target for the battery (sum of assigned wells + noise)


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
- Use 2–5 spatial clusters (`spatial.n_clusters`)
- Tune connector density with `spatial.max_per_well` and `spatial.max_dist_fraction`

### For Benchmarking
- Generate 20–100 instances per configuration
- Vary `n_wells` systematically (e.g., 20, 50, 100, 200, 500)
- Test both clustered and dispersed spatial configurations
- Include both high and low correlation scenarios

---

### Greedy Solver

The C++ greedy heuristic solver is built with `make` in the `solver/` directory:

```bash
cd solver && make
```

Then run directly:

```bash
solver/bin/solve -c solver_config.yaml -p parameters.dat -b batteries.dat -d distance.dat -o solution.txt
```

The `run_pipeline.sh` script builds and invokes the solver automatically.

### SCIP Solver

SCIP must be installed and available in PATH. The pipeline feeds `.lp` files (generated in step 1) directly to SCIP — no `zimpl` installation required:

```bash
scip -f generator/instances/model_10_2_1.lp
```

For reproducible results matching a reference environment, use SCIP 9.2.4 in optimized mode. The Manjaro/Arch package ships a debug build (10.0.1); install the official prebuilt from https://www.scipopt.org or via conda:

```bash
conda install -c conda-forge scip=9.2
```

---

## 📝 License

General Public License v3.0 (GPL-3.0)

## 🤝 Contributing

Contributions are welcome! Please submit issues or pull requests on GitHub.

## 📧 Contact

For questions or support, please contact the maintainer at matias.micheletto@uns.edu.ar

---