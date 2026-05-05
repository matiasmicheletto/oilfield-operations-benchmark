"""
Main entry point for the Oilfield Operations Benchmark instance generator.
Usage:
    python generator/main.py --config config.yaml

Overrides:
    python generator/main.py --set general.n_wells=20 general.n_batteries=5

Output:
    For each instance, generates:
    - param_<id>.dat: tabular well parameters (G, N, r, R, P, C, battery_id).
    - bat_<id>.dat: tabular battery parameters (Gpt, Loss, Cost).
    - model_<id>.lp: LP-format file for use with SCIP/CPLEX without zimpl.
    - spatial_<id>.png: visual plot of the terrain and well locations.
    - dist_<id>.dat: tabular distance matrix between wells and operations center.
    - spatial_data_<id>.npz: numpy arrays for elevation, cost map, well positions, etc.
"""

import sys
import os
import numpy as np
from pathlib import Path

from util.config import parse_args, resolve_config_path, load_config, apply_overrides

from core.well_generator import WellGenerator
from core.battery_generator import BatteryGenerator
from core.zpl_generator import ZPLGenerator
from core.lp_generator import LPGenerator
from core.spatial_generator import SpatialGenerator


def run_packager(config):
    print(f"--- Initializing Packager ---")

    gen_cfg = config["general"]
    out_dir = Path(gen_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    num_instances = gen_cfg["num_instances"]
    num_scenarios = gen_cfg.get("num_scenarios", 1)

    rng = np.random.default_rng(gen_cfg.get("seed", 42))

    well_gen = WellGenerator(rng, config)
    bat_gen = BatteryGenerator(rng, config)
    spatial_gen = SpatialGenerator(rng, config)
    zpl_gen = ZPLGenerator(config)
    lp_gen  = LPGenerator(config)

    for k in range(1, num_instances + 1):

        # ---- Per-instance stem and file names ----
        instance_stem = f"{gen_cfg['n_wells']}_{gen_cfg['n_batteries']}_{k}"
        p_name = f"{gen_cfg['param_name_prefix']}_{instance_stem}.dat"
        param_file = out_dir / p_name

        # 1. Generate well data (shared across all scenarios of this instance)
        wells = well_gen.generate()

        # bat_ids is deterministic; re-derive it for this instance's wells
        bat_ids, _, _, _ = bat_gen.generate(wells)

        # 2. Export Parameters (written once per instance)
        with open(param_file, "w") as f:
            f.write("ID\tG\tN\tr\tR\tP\tC\tB\n")
            for i in range(gen_cfg["n_wells"]):
                f.write(f"{i+1}\t{wells['G'][i]}\t{wells['N'][i]}\t{wells['r'][i]}\t"
                        f"{wells['R'][i]}\t{wells['P'][i]}\t{wells['C'][i]}\t{bat_ids[i]}\n")

        # 3. Plot well distributions (once per instance)
        well_gen.plot_distributions(wells, k)
        well_gen.plot_histograms(wells, k)

        for s in range(1, num_scenarios + 1):

            # ---- Per-scenario stem and file names ----
            scenario_stem = f"{instance_stem}_{s}"
            b_name = f"{gen_cfg['bat_name_prefix']}_{scenario_stem}.dat"
            d_name = f"{gen_cfg['dist_name_prefix']}_{scenario_stem}.dat"
            z_name = f"{gen_cfg['zpl_name_prefix']}_{scenario_stem}.zpl"
            l_name = f"model_{scenario_stem}.lp"
            s_name = f"spatial_{scenario_stem}.npz"
            bat_file  = out_dir / b_name
            dist_file = out_dir / d_name
            zpl_file  = out_dir / z_name
            lp_file   = out_dir / l_name

            # 4. Generate scenario-specific data
            _, bat_targets, bat_loses, bat_costs = bat_gen.generate(wells)

            elev, cost_map = spatial_gen.generate_terrain()
            well_positions = spatial_gen.generate_well_positions(n_wells=gen_cfg["n_wells"])
            ops_center, paths = spatial_gen.build_network(cost_map, well_positions)
            distance_matrix = spatial_gen.compute_distance_matrix(cost_map, well_positions)

            # 5. Export Batteries
            with open(bat_file, "w") as f:
                f.write("ID\tGpt\Loss\Cost\n")
                for i in range(len(bat_targets)):
                    f.write(f"{i+1}\t{bat_targets[i]}\t{bat_loses[i]}\t{bat_costs[i]}\n")

            # 6. Export Distance Matrix
            with open(dist_file, "w") as f:
                f.write("Distance Matrix\n")
                for row in distance_matrix:
                    f.write("\t".join(map(lambda x: f"{int(round(x))}", row)) + "\n")

            # 7. Generate ZPL model
            zpl_gen.generate(zpl_file, p_name, b_name, d_name)

            # 8. Generate LP file (used by SCIP/CPLEX without zimpl)
            if gen_cfg.get("generate_lp", True):
                lp_gen.generate(lp_file, scenario_stem, wells, bat_ids, bat_targets, distance_matrix)

            # 9. Plot spatial network
            spatial_gen.plot_network(elev, well_positions, paths, ops_center, scenario_stem, out_dir)

            # 10. Save spatial data for route overlay
            spatial_gen.save_spatial_data(elev, cost_map, well_positions, ops_center, paths, scenario_stem, out_dir)

            print(f"Packaged instance {k}, scenario {s}: "
                  f"{param_file.name}, {bat_file.name}, {dist_file.name}, {z_name}, {l_name}, {s_name} generated.")


if __name__ == "__main__":
    args = parse_args()
    config_path = resolve_config_path(args.config_file)
    config = load_config(str(config_path))
    if args.overrides:
        apply_overrides(config, args.overrides)
    run_packager(config)