#pragma once
#ifndef SOLVER_H
#define SOLVER_H

#include "models.h"

namespace solver {

// Greedy heuristic with cyclic-swap local search.
//
// For each battery the algorithm:
//   1. Ranks the battery's wells by priority DESC, then cost ASC.
//   2. Slides a window of `max_wells` entries cyclically through the ranked
//      list until sum(G) falls within `tolerance` of the battery target Gpt.
//   3. If a feasible window is found for every battery the selected wells are
//      combined and a nearest-neighbour route is computed.
//
// Parameters:
//   inst       – loaded instance (wells, batteries, distance matrix)
//   sol        – output solution
//   max_wells  – maximum wells to visit per battery (the "K" parameter)
//   tolerance  – allowed fractional deviation from Gpt  (e.g. 0.05 = ±5 %)
//
// Returns true when a feasible solution is produced for ALL batteries.
bool solve(const Instance& inst, Solution& sol,
           int max_wells, double tolerance);

} // namespace solver

#endif // SOLVER_H
