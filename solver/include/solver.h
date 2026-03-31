#pragma once
#ifndef SOLVER_H
#define SOLVER_H

#include "loader.h"
#include "models.h"

namespace solver {

// Globally-aware greedy heuristic.
//
// Selection phase:
//   All wells are ranked globally (priority DESC, cost ASC).  They are added
//   to the selected set one by one as long as all three global hard constraints
//   are satisfied (max_quantity, max_cost, max_loss) and per-battery production
//   feasibility is maintained.  After the pass every battery's selected subset
//   must independently cover its Gpt target within cfg.tolerance; if any
//   battery fails, solve() returns false.
//
// Routing phase (single crew, multi-crew partitioning deferred to Step 3):
//   Nearest-neighbour TSP over the selected wells.
//
// Parameters:
//   inst  – loaded instance (wells, batteries, distance matrix)
//   sol   – output solution
//   cfg   – solver configuration (tolerance, max_quantity, max_cost, max_loss,
//            crews, max_wells)
//
// Returns true when a feasible solution is produced.
bool solve(const Instance& inst, Solution& sol,
           const loader::SolverConfig& cfg);

} // namespace solver

#endif // SOLVER_H
