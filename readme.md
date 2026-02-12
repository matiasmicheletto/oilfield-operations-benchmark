# Oilfield Instance Generator

A Python-based utility for generating synthetic oilfield datasets for MILP (Mixed-Integer Linear Programming) optimization and benchmarking. This tool allows for highly flexible data generation by defining probability distributions directly in a YAML configuration file.

## 🚀 Features

* **Dynamic Distributions:** Define any NumPy-supported distribution (Uniform, Beta, LogNormal, etc.) for well properties via YAML.
* **Correlated Variables:** Generate Risk and Priority values that correlate with production levels to simulate realistic field data.
* **Automatic Scaling & Rounding:** Built-in support for scaling values (e.g., 0–1 to 0–100) and rounding to integers for MILP solver compatibility.
* **Reproducibility:** Seed-based random number generation ensures the same instances can be recreated for research consistency.

---

## 🛠 Installation

1. **Requirements:**
   * Python 3.8+
   * `numpy`
   * `PyYAML`

2. **Setup:**
   ```bash
   pip install -r requirements.txt
   ```

## 📖 Usage
To generate instances, run the script and provide the path to your configuration file:

```bash
python generate_instances.py config.yaml
```

The script will create an output directory (as specified in the YAML) containing .dat files with the following header: ```ID    G    N    r    R    P    C    B```

## ⚙️ Configuration Guide

### 1. Production (Dynamic Distributions)
The script uses a dynamic sampling engine. You can define the distribution for Gross production, Net ratio, and Regime.

* Example LogNormal:

```yaml
gross:
  distribution: lognormal
  mu: 3.8
  sigma: 0.6
```

* Example Uniform:

```yaml
gross:
  distribution: uniform
  low: 50
  high: 150
```

### 2. Risk & Priority (Correlation Logic)
If a ```distribution``` key is provided, the variable is sampled independently. If omitted, the script uses a correlation factor (```rho```) to tie the variable to production:
$$Value = \text{clip}(\rho \cdot \text{NormalizedProduction} + \text{GaussianNoise}, 0, 1)$$

### 3. Cost Model
Intervention costs are derived from Risk ($R$) and Priority ($P$):

$$C = \kappa + w_r \cdot R + w_p \cdot (1 - P)$$

### 4. Rounding & Scaling

* Scaling: Multiplies the raw [0, 1] value by a factor (e.g., 100) before rounding.

* Rounding: Set to ```0``` for integers (best for MILP indicators) or ```>0``` for floating point.

## 📂 Output Format
Generated ```.dat``` files are tab-separated. Each row corresponds to a well with the following columns:

| Column | Description |  
| --- | --- |
| ID | Unique well identifier |  
| G | Gross Production |  
| N | Net Production |  
| r | Production Regime |  
| R | Risk Factor |  
| P | Priority Index |  
| C | Intervention Cost |  
| B | Battery Assignment |  
