"""
Battery Generator
-----------------
This module generates battery-level data based on well-level data. It assigns wells to batteries and computes the target gross production (G_t), loss, and cost for each battery.

Inputs:
- well_data: dict with keys 'G', 'N', 'r', 'C', each an array of length n_wells
Outputs:
- battery_ids: array of length n_wells, where each entry is the 1-based battery ID assigned to the corresponding well
- battery_targets: array of length n_batteries, where each entry is the target gross production (G_t) for that battery
- battery_loss: array of length n_batteries, where each entry is the total loss for that battery
- battery_cost: array of length n_batteries, where each entry is the total cost for that
"""

import numpy as np


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
        
        # 2. Compute G_t (Sum + Noise), Loss, and Cost for each Battery
        battery_targets = []
        battery_loss = []
        battery_cost = []
        for b_id in range(1, n_bats + 1):
            mask = (battery_ids == b_id)
            total_gross = np.sum(well_data["G"][mask])
            current_gross = np.sum((well_data["G"][mask]*well_data["r"][mask])/100)
            total_loss = np.sum(((well_data["G"][mask]-well_data["N"][mask]))) #*well_data["r"][mask])/100)
            total_cost = np.sum(well_data["C"][mask])
            pct = self.rng.uniform(40, 90)
            g_t = total_gross * pct/100
            if rounding == 0:
                battery_targets.append(int(round(g_t)))
            else:
                battery_targets.append(round(g_t, rounding))
            battery_loss.append(total_loss)
            battery_cost.append(total_cost)
                                
        return battery_ids, battery_targets, battery_loss, battery_cost