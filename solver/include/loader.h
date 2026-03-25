#pragma once
#ifndef LOADER_H
#define LOADER_H

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
    int    max_wells = 20;    // cyclic-window size K
    double tolerance = 0.05;  // ±5 % deviation from Gpt
    int    crews     = 2;
    // [output]
    std::string output_file;
    bool        debug = false;
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
