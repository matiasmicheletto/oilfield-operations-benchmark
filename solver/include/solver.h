#pragma once
#ifndef SOLVER_H
#define SOLVER_H

#include "loader.h"
#include "models.h"

namespace solver {

// Regime-adjustment heuristic.
//
// For each battery the wells are sorted according to cfg.sort_method:
//   priority_cost – ascending priority/cost ratio
//   loss          – ascending gross-minus-net production loss (at max regime)
//   route         – nearest-neighbour TSP visit order
//
// The sorted list is then scanned and each well's regime is adjusted toward
// 100 % (when the battery target exceeds current production) or toward 0 %
// (when the target is below current production), stopping as soon as the
// target is met.  The last adjusted well receives the exact partial regime
// required to hit the target precisely.  Wells beyond that point are unchanged.
//
// Global hard constraints (max_quantity, max_cost, max_loss) are enforced
// well-by-well; a well is skipped if accepting it would violate any limit.
//
// The solution is the set of wells whose regimes were changed together with
// their new regimes.  Objective values (total_cost, total_loss) and the
// multi-crew route are computed from that set exactly as before.
//
// Returns true when every battery's target is met within cfg.tolerance.
bool solve(const Instance& inst, Solution& sol,
           const loader::SolverConfig& cfg);

} // namespace solver

#endif // SOLVER_H
