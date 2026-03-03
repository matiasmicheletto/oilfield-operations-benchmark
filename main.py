import sys
import os
import yaml
import numpy as np
from pathlib import Path

from core.well_generator import WellGenerator
from core.battery_generator import BatteryGenerator
from core.zpl_generator import ZPLGenerator
from core.spatial_generator import SpatialGenerator


def load_config(path: str) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file '{path}' not found.")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_packager(config_path):
    print(f"--- Initializing Packager with: {config_path} ---")
    config = load_config(config_path)

    config = load_config(config_path)
    gen_cfg = config["general"]
    out_dir = Path(gen_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    rng = np.random.default_rng(gen_cfg.get("seed", 42))
    
    well_gen = WellGenerator(rng, config)
    bat_gen = BatteryGenerator(rng, config)
    spatial_gen = SpatialGenerator(rng, config)
    zpl_gen = ZPLGenerator(config)

    for k in range(1, gen_cfg["num_instances"] + 1):

        # File naming
        suffix = f"{gen_cfg['n_wells']}_{gen_cfg['n_batteries']}_{k}.dat"
        p_name = f"{gen_cfg['param_name_prefix']}_{suffix}"
        b_name = f"{gen_cfg['bat_name_prefix']}_{suffix}"
        d_name = f"{gen_cfg['dist_name_prefix']}_{suffix}"
        z_name = f"{gen_cfg['zpl_name_prefix']}_{suffix.replace('.dat', '.zpl')}"
        param_file = out_dir / p_name 
        bat_file = out_dir / b_name
        dist_file = out_dir / d_name
        zpl_file = out_dir / z_name

        # 1. Generate Data
        wells = well_gen.generate()
        bat_ids, bat_targets = bat_gen.generate(wells)
        zpl_gen.generate(zpl_file, p_name, b_name, d_name)

        elev, cost_map = spatial_gen.generate_terrain()
        well_positions = spatial_gen.generate_well_positions()
        ops_center, paths = spatial_gen.build_network(cost_map, well_positions)
        distance_matrix = spatial_gen.compute_distance_matrix(cost_map, well_positions)
        
        # 2. Export Parameters
        with open(param_file, "w") as f:
            f.write("ID\tG\tN\tr\tR\tP\tC\tB\n")
            for i in range(gen_cfg["n_wells"]):
                f.write(f"{i+1}\t{wells['G'][i]}\t{wells['N'][i]}\t{wells['r'][i]}\t"
                        f"{wells['R'][i]}\t{wells['P'][i]}\t{wells['C'][i]}\t{bat_ids[i]}\n")

        # 3. Export Batteries
        with open(bat_file, "w") as f:
            f.write("ID\tGpt\n")
            for i, target in enumerate(bat_targets):
                f.write(f"{i+1}\t{target}\n")

        # 4. Export Distance Matrix
        with open(dist_file, "w") as f:
            f.write("Distance Matrix\n")
            for row in distance_matrix:
                f.write("\t".join(map(lambda x: f"{int(round(x))}", row)) + "\n")

        # 5. Plotting (Optional)
        well_gen.plot_distributions(wells, k)
        well_gen.plot_histograms(wells, k)
        spatial_gen.plot_network(elev, well_positions, paths, ops_center, k)

        print(f"Packaged instance {k}: {param_file.name}, {bat_file.name}, {dist_file.name}, {z_name} generated.")


if __name__ == "__main__":
    # Check if the user provided a specific config file as an argument
    if len(sys.argv) > 1:
        target_config = sys.argv[1]
    else:
        # Fallback to the default filename
        target_config = "default_config.yaml"
        
        # Check if the default actually exists before proceeding
        if not os.path.exists(target_config):
            print(f"Warning: No argument provided and '{target_config}' was not found.")
            print("Usage: python main.py <path_to_config.yaml>")
            sys.exit(1)

    run_packager(target_config)