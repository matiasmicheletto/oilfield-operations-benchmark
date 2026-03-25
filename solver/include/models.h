#pragma once
#ifndef MODELS_H
#define MODELS_H

#include <vector>
#include <string>

// ---------------------------------------------------------------------------
// Core data structures – mirrors the columns produced by WellGenerator
// and the .dat files written by the Python pipeline:
//   parameters_*.dat  →  ID  G  N  r  R  P  C  B
//   batteries_*.dat   →  ID  Gpt
//   distance_*.dat    →  (n+1)×(n+1) matrix, row/col 0 = depot
// ---------------------------------------------------------------------------

struct Well {
    int    id;              // unique 1-indexed well ID
    double gross_prod;      // G – total gross production capacity
    double net_prod;        // N – net production
    double current_regime;  // r – current production regime (0-100)
    double risk;            // R – risk factor   (0-100)
    double priority;        // P – priority level (0-100)
    double cost;            // C – intervention cost
    int    battery_id;      // B – parent battery ID
};

struct Battery {
    int    id;
    double target_gross;    // Gpt – target gross production for this battery
};

struct Instance {
    std::string name;
    std::vector<Well>                 wells;
    std::vector<Battery>              batteries;
    // (n+1) × (n+1) travel-time matrix.
    // Index 0   = operations-centre depot.
    // Index i   = well whose id == i  (1-indexed, matching the parameters file).
    std::vector<std::vector<double>>  dist_matrix;
};

struct Solution {
    std::vector<int> selected_ids;  // Well IDs chosen for intervention
    std::vector<int> route;         // Visit order using dist_matrix indices (0 = depot)
    double           total_distance;
    bool             feasible;
};

#endif // MODELS_H
