#pragma once
#ifndef LOADER_H
#define LOADER_H

#include <limits>
#include <string>
#include <vector>

#include "models.h"

namespace loader {

// ---------------------------------------------------------------------------
// Flat configuration struct – mirrors solver_config.yaml.
// Defaults match the hardcoded constants used before config-file support.
// ---------------------------------------------------------------------------
struct SolverConfig {
    // [input]
    std::string param_file;
    std::string battery_file;
    std::string dist_file;
    // [solver]
    int    max_wells    = 1000;  // cyclic-window size K (legacy; now controls sort window)
    double tolerance    = 0.001; // ± tolerance is considered "close enough" for greedy selection
    int    crews        = 3;
    int    max_quantity = 1000;  //std::numeric_limits<int>::max();     // global well-count cap
    double max_cost     = 50000; //std::numeric_limits<double>::max();  // budget cap (sum C[i])
    double max_loss     = 50000; //std::numeric_limits<double>::max();  // net-loss cap (sum (G-N)*regime/100)
    std::string sort_method = "priority_cost"; // per-battery well sort: priority_cost | loss | route

    // [harmony_search]
    int    hs_hms              = 30;     // harmony memory size
    int    hs_iterations       = 1000;   // improvisation iterations
    double hs_hmcr             = 0.92;   // harmony memory consideration rate
    double hs_par              = 0.35;   // pitch adjustment rate
    double hs_bw               = 5.0;    // pitch bandwidth (regime points)
    double hs_change_prob      = 0.25;   // probability a variable departs from current regime
    bool   hs_seed_with_greedy = true;   // seed harmony memory with greedy solution
    double hs_weight_cost      = 1.0;    // weighted objective term: cost
    double hs_weight_loss      = 1.0;    // weighted objective term: loss
    double hs_weight_distance  = 1.0;    // weighted objective term: route distance
    double hs_weight_quantity  = 0.2;    // weighted objective term: selected well count
    double hs_penalty          = 1000.0; // global penalty multiplier for violations
    // [output]
    std::string  output_file;
    PRINT_FORMAT print_format = PRINT_FORMAT::TXT;
    bool         debug = false;
};

// Populate cfg from a YAML file.  Only keys present in the file are applied;
// missing keys keep their current (default) values.
void load_yaml_config(SolverConfig& cfg, const std::string& path);

// Apply a single "section.key=value" override from --set.
void apply_override(SolverConfig& cfg, const std::string& kv);

// Load a complete Instance from the three .dat files produced by main.py.
//   param_file  – parameters_*.dat   (wells: ID G N r R P C B)
//   bat_file    – batteries_*.dat    (targets: ID Gpt)
//   dist_file   – distance_*.dat     ((n+1)×(n+1) matrix, first row is text header)
Instance load(const std::string& param_file,
              const std::string& bat_file,
              const std::string& dist_file);

} // namespace loader

#endif // LOADER_H
