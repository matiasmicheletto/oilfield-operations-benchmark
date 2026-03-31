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
    std::vector<int>    selected_ids;   // Well IDs chosen for intervention
    double              total_distance;
    bool                feasible;

    // new_regimes[i] = assigned production regime (0-100) for well with id==i.
    // Indexed by well ID (1-based); index 0 is unused.
    // Only meaningful for IDs in selected_ids; all others are 0.
    std::vector<double> new_regimes;

    // One route per crew: each route is [0, w1, w2, ..., wk, 0] (depot at both ends).
    // Empty crews are represented as an empty vector (not {0,0}).
    std::vector<std::vector<int>> crew_routes;

    double total_cost;  // sum C[i] over selected wells
    double total_loss;  // sum (G[i]-N[i]) * newregime[i] / 100 over selected wells

    // Legacy single-route view (union of all crew routes, excluding depot repetitions).
    // Kept for backward compatibility; prefer crew_routes for new code.
    std::vector<int> route;
};

#endif // MODELS_H
