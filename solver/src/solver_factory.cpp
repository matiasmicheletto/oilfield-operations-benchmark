#include "../include/solver_base.h"
#include "../include/greedy_solver.h"

#include <stdexcept>

namespace solver {

// ---------------------------------------------------------------------------
// Factory function: create a solver by name
// ---------------------------------------------------------------------------
std::unique_ptr<SolverBase> createSolver(const std::string& method)
{
    if (method == "greedy") {
        return std::make_unique<GreedySolver>();
    }
    // TODO: Add more solvers here as they are implemented:
    // else if (method == "simulated_annealing") {
    //     return std::make_unique<SimulatedAnnealingSolver>();
    // }
    // else if (method == "genetic_algorithm") {
    //     return std::make_unique<GeneticAlgorithmSolver>();
    // }

    throw std::invalid_argument("Unknown solver method: '" + method +
                                "'. Supported methods: greedy");
}

} // namespace solver
