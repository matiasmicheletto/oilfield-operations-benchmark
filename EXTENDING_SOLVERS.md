# Extending the Solver with New Algorithms

This document describes how to add new optimization methods to the oilfield operations solver using the modular solver framework.

## Architecture Overview

The solver uses an **abstract base class pattern** with a **factory function** to support multiple algorithms:

- **`SolverBase`** (`solver/include/solver_base.h`): Abstract interface all solvers implement
- **`GreedySolver`** (`solver/include/greedy_solver.h`, `solver/src/greedy_solver.cpp`): Concrete implementation of the cyclic-window heuristic
- **Factory Function** (`solver/src/solver_factory.cpp`): Creates solver instances by name

## Adding a New Algorithm

### Step 1: Create the Header File

Create a new header file in `solver/include/` that defines your solver class:

```cpp
// solver/include/my_solver.h
#pragma once
#ifndef MY_SOLVER_H
#define MY_SOLVER_H

#include "solver_base.h"
#include "loader.h"
#include "models.h"

namespace solver {

// MySolver: [Brief description of algorithm]
class MySolver : public SolverBase {
public:
    // Implement the abstract interface
    bool solve(const Instance& inst,
               Solution& sol,
               const loader::SolverConfig& cfg) override;

    std::string name() const override { return "my_method"; }
    
    std::string description() const override {
        return "Description of the algorithm: ...";
    }

private:
    // Helper methods for your algorithm
    // ...
};

} // namespace solver

#endif // MY_SOLVER_H
```

### Step 2: Create the Implementation File

Create a new `.cpp` file in `solver/src/` with your algorithm implementation:

```cpp
// solver/src/my_solver.cpp
#include "../include/my_solver.h"
#include "../include/utils.h"
#include <iostream>

namespace solver {

bool MySolver::solve(const Instance& inst,
                     Solution& sol,
                     const loader::SolverConfig& cfg) {
    utils::dbg << "[my_method] Starting solve...\n";
    
    // Implement your algorithm here
    // 1. Select wells for each battery
    // 2. Compute routes for intervention crews
    // 3. Populate the Solution object
    
    // Return true if feasible solution found, false otherwise
    return true;
}

} // namespace solver
```

### Step 3: Register in the Factory

Update `solver/src/solver_factory.cpp` to create instances of your new solver:

```cpp
std::unique_ptr<SolverBase> createSolver(const std::string& method) {
    // ... existing code ...
    
    if (method == "greedy") {
        return std::make_unique<GreedySolver>();
    }
    
    // Add your new solver here:
    if (method == "my_method") {
        return std::make_unique<MySolver>();
    }
    
    // ... rest of factory ...
}
```

Also update the error message in `solver_factory.cpp` to list your new method:

```cpp
throw std::runtime_error("Unknown solver method: '" + method + 
                         "'. Supported methods: greedy, my_method");
```

### Step 4: Update the Manual

Edit `solver/assets/solve_manual.txt` to document your new method in the OPTIONS section:

```
   -m METHOD, --method METHOD
      Select the optimization algorithm. Available methods:
        greedy     Cyclic-window heuristic
        my_method  Description of your algorithm
      Default: greedy
```

### Step 5: Compile and Test

The Makefile automatically picks up new `.cpp` files:

```bash
cd solver
make clean
make
```

Test your solver:

```bash
./bin/solve -p instances/parameters_10_2_1.dat \
            -b instances/batteries_10_2_1_1.dat \
            -d instances/distance_10_2_1_1.dat \
            --method my_method
```

## Data Structures

### Instance

Defined in `solver/include/models.h`:

```cpp
struct Instance {
    std::vector<Well>    wells;     // Well parameters (production, cost, etc)
    std::vector<Battery> batteries; // Battery configurations and targets
    std::vector<std::vector<double>> dist_matrix; // Travel times (0-indexed)
};

struct Well {
    int id;        // 1-indexed
    double G;      // Production under normal regime (bbl/day)
    int N;         // Current regime [0, 100]
    double r, R, P, C;  // Decline/cost parameters
    int battery;   // Battery index (0-indexed)
};

struct Battery {
    int id;
    double Gpt;    // Target production (bbl/day)
    double max_cost, max_loss;  // Per-battery limits
};
```

### Solution

Defined in `solver/include/models.h`:

```cpp
struct Solution {
    std::vector<std::vector<int>> routes;  // crew_id -> [well_id, ...]
    std::vector<int> new_regimes;          // Well index -> new regime [0,100]
    bool feasible = false;
    
    // Print formatted output
    void print(std::ostream& out, const Instance& inst, 
               int num_crews, PRINT_FORMAT fmt);
};
```

## Algorithm Template: Simulated Annealing (Example)

```cpp
// solver/include/sa_solver.h
class SimulatedAnnealingSolver : public SolverBase {
public:
    bool solve(const Instance& inst, Solution& sol,
               const loader::SolverConfig& cfg) override;
    
    std::string name() const override { return "sa"; }
    
    std::string description() const override {
        return "Simulated annealing metaheuristic";
    }

private:
    double temperature = 100.0;
    double cooling_rate = 0.95;
    int iterations = 1000;
};
```

## Template: Genetic Algorithm (Example)

```cpp
// solver/include/ga_solver.h
class GeneticAlgorithmSolver : public SolverBase {
public:
    bool solve(const Instance& inst, Solution& sol,
               const loader::SolverConfig& cfg) override;
    
    std::string name() const override { return "ga"; }
    
    std::string description() const override {
        return "Genetic algorithm metaheuristic";
    }

private:
    int population_size = 50;
    double mutation_rate = 0.1;
    int generations = 100;
};
```

## Key Utilities

Available in `solver/include/utils.h`:

- `nearest_neighbor()` - Greedy TSP construction
- `two_opt()` - 2-opt local search
- `multi_crew_route()` - Crew assignment and routing
- Distance matrix queries via `inst.dist_matrix[i][j]`

## Debug Output

Use the debug stream for logging:

```cpp
utils::dbg << "[my_method] Starting phase 1...\n";
utils::dbg << "[my_method] Selected " << num_wells << " wells\n";
```

Output controlled by `-D/--dbg` flag; enabled via `cfg.debug`.

## Testing

Run the pipeline to generate instances:

```bash
bash run_pipeline.sh
```

Test your solver on all instances:

```bash
for inst in generator/instances/parameters_10_2_*.dat; do
    stem=$(basename "$inst" .dat | sed 's/parameters_//')
    ./solver/bin/solve \
        -p "$inst" \
        -b "generator/instances/batteries_${stem}.dat" \
        -d "generator/instances/distance_${stem}.dat" \
        --method my_method \
        -o "output/my_method_${stem}_solution.txt"
done
```

## Debugging Tips

1. **Enable debug output**: Use `-D` flag to see verbose diagnostics
2. **Check feasibility**: Verify `Solution.feasible == true`
3. **Validate routes**: Ensure all routes start/end at depot (index 0)
4. **Inspect regimes**: Check new regimes are in [0, 100]
5. **Log key decisions**: Add `utils::dbg` statements at algorithm checkpoints

## Future Enhancements

- Constraint-based solvers (CPLEX, SCIP integration)
- Hybrid methods (greedy + local search)
- Parallel algorithms (multi-threaded metaheuristics)
- Parameter tuning framework
