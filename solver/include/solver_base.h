#pragma once
#ifndef SOLVER_BASE_H
#define SOLVER_BASE_H

#include <memory>
#include <string>
#include "loader.h"
#include "models.h"

namespace solver {

// ---------------------------------------------------------------------------
// Abstract base class for optimization solvers.
// All solver implementations inherit from this interface.
// ---------------------------------------------------------------------------
class SolverBase {
public:
    virtual ~SolverBase() = default;

    // Solve the instance given the configuration.
    // Returns true if a feasible solution was found, false otherwise.
    // The solution is populated in the `sol` parameter.
    virtual bool solve(const Instance& inst,
                       Solution& sol,
                       const loader::SolverConfig& cfg) = 0;

    // Return a string name/identifier for this solver (e.g., "greedy", "simulated_annealing")
    virtual std::string name() const = 0;

    // Return a brief description of the solver method
    virtual std::string description() const = 0;
};

// Factory function to create a solver by name.
// Supported names: "greedy", "sa" (simulated annealing), etc.
std::unique_ptr<SolverBase> createSolver(const std::string& method);

} // namespace solver

#endif // SOLVER_BASE_H
