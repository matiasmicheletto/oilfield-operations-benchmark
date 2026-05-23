#pragma once
#ifndef HARMONY_SEARCH_SOLVER_H
#define HARMONY_SEARCH_SOLVER_H

#include "solver_base.h"

namespace solver {

// ---------------------------------------------------------------------------
// Harmony Search solver for constrained regime minimization.
//
// Decision variables:
//   one continuous regime value [0,100] per well ID.
//
// Objective:
//   weighted sum of normalized cost/loss/distance/quantity terms + penalties
//   for target and hard-constraint violations.
//
// The method includes a lightweight repair stage per battery so candidate
// harmonies are nudged toward battery target feasibility before evaluation.
// ---------------------------------------------------------------------------
class HarmonySearchSolver : public SolverBase {
public:
    bool solve(const Instance& inst,
               Solution& sol,
               const loader::SolverConfig& cfg) override;

    std::string name() const override { return "hs"; }
    std::string description() const override {
        return "Harmony Search metaheuristic with penalty-based constrained minimization";
    }
};

} // namespace solver

#endif // HARMONY_SEARCH_SOLVER_H
