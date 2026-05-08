# Solver Refactoring: Transition from Monolithic to Modular Architecture

## Summary of Changes

The solver has been refactored from a monolithic design to a **modular, extensible architecture** supporting multiple optimization algorithms.

### Files Created

1. **`solver/include/solver_base.h`** - Abstract base interface
   - Defines `SolverBase` class with virtual `solve()`, `name()`, `description()` methods
   - Factory function declaration: `createSolver(method_name)`

2. **`solver/include/greedy_solver.h`** - Concrete GreedySolver class
   - Implements `SolverBase` interface
   - Wraps the cyclic-window heuristic algorithm

3. **`solver/src/greedy_solver.cpp`** - Algorithm implementation
   - ~350 lines of cyclic-window heuristic logic
   - Extracted from original `solver.cpp`
   - Debug prefix: `[greedy]` (was `[solver]`)

4. **`solver/src/solver_factory.cpp`** - Factory pattern implementation
   - Creates solver instances by method name
   - Supports: `"greedy"` (default)
   - Includes placeholder comments for future methods (SA, GA, etc.)
   - Helpful error messages listing available methods

5. **`EXTENDING_SOLVERS.md`** - Extension guide (NEW)
   - Step-by-step instructions for adding new algorithms
   - Data structure documentation
   - Example templates for SA and GA solvers
   - Testing and debugging tips

### Files Modified

1. **`solver/src/solve_main.cpp`**
   - Changed to use factory pattern
   - Added `--method` / `-m` CLI flag (default: "greedy")
   - Uses `solver::createSolver(method_name)` to instantiate solver
   - Proper error handling for unknown methods

2. **`solver/assets/solve_manual.txt`**
   - Added `-m METHOD, --method METHOD` to OPTIONS section
   - Documented available methods and defaults

3. **`Makefile`**
   - No changes needed (automatically includes new `.cpp` files)

### Files Deprecated (Still Present)

1. **`solver/src/solver.cpp`** (DEPRECATED)
   - Original monolithic implementation
   - Logic now duplicated in `greedy_solver.cpp`
   - Should be removed after full migration validation
   - Kept temporarily for reference

2. **`solver/include/solver.h`** (DEPRECATED)
   - Contains old free-function interface: `bool solver::solve(...)`
   - Superseded by `SolverBase` and factory pattern
   - No longer used by `solve_main.cpp`

## Migration Path

### Phase 1: Architecture (COMPLETE ✓)
- [x] Create abstract `SolverBase` class
- [x] Move GreedySolver logic to separate class
- [x] Implement factory pattern
- [x] Update CLI to use factory
- [x] Create extension guide
- [x] Validate compilation and execution

### Phase 2: Cleanup (PENDING)
- [ ] Remove duplicate `solver.cpp` after full validation
- [ ] Mark `solver.h` as deprecated (or remove)
- [ ] Update project documentation if needed

### Phase 3: Future Extensions (READY)
- [ ] Simulated Annealing solver (template provided in EXTENDING_SOLVERS.md)
- [ ] Genetic Algorithm solver (template provided)
- [ ] Other metaheuristics (follow same pattern)

## API Changes

### Old API (DEPRECATED)
```cpp
// In solve_main.cpp or user code
#include "../include/solver.h"

Solution sol;
if (!solver::solve(inst, sol, cfg)) {
    // Handle error
}
```

### New API (CURRENT)
```cpp
// In solve_main.cpp
#include "../include/solver_base.h"

auto solver = solver::createSolver("greedy");  // returns std::unique_ptr<SolverBase>
Solution sol;
if (!solver->solve(inst, sol, cfg)) {
    // Handle error
}
```

## CLI Changes

### Old CLI (BEFORE)
```bash
./solve -p params.dat -b batt.dat -d dist.dat
# Single algorithm, no selection
```

### New CLI (AFTER)
```bash
./solve -p params.dat -b batt.dat -d dist.dat --method greedy
# or use default:
./solve -p params.dat -b batt.dat -d dist.dat
```

## Compilation

Makefile automatically includes all `.cpp` files in `src/`:

```bash
cd solver
make clean
make
```

To add a new solver:
1. Create header: `include/new_solver.h`
2. Create implementation: `src/new_solver.cpp`
3. Register in factory: `src/solver_factory.cpp`
4. Rebuild: `make clean && make`

## Testing

### Single Instance (Greedy)
```bash
./bin/solve -p instances/parameters_10_2_1.dat \
            -b instances/batteries_10_2_1_1.dat \
            -d instances/distance_10_2_1_1.dat \
            --method greedy -D
```

### Invalid Method (Error Handling)
```bash
./bin/solve -p instances/parameters_10_2_1.dat \
            -b instances/batteries_10_2_1_1.dat \
            -d instances/distance_10_2_1_1.dat \
            --method invalid_method
# Output: Error: unknown solver method 'invalid_method'. Supported methods: greedy
```

### Full Pipeline
```bash
bash run_pipeline.sh
```
All instances solve successfully with new architecture ✓

## Verification Checklist

- [x] Code compiles with no errors or warnings
- [x] `greedy` method works as default
- [x] `--method greedy` explicitly works
- [x] Invalid methods produce helpful error messages
- [x] All generated instances solve correctly
- [x] Solution output format unchanged
- [x] CLI help updated
- [x] Extension guide created
- [x] Pipeline validates end-to-end

## Technical Notes

### Namespace
All solver code lives in `solver::` namespace to avoid conflicts.

### Memory Management
Factory returns `std::unique_ptr<SolverBase>` for automatic cleanup.

### Error Handling
Factory throws `std::runtime_error` with helpful message if method unknown.

### Debug Output
Algorithm-specific debug prefix used: `[greedy]`, future `[sa]`, `[ga]`, etc.

### Build System
- Uses `find` to auto-discover source files
- Dependency generation (`.d` files) handled automatically
- Multi-entry-point design via `*_main.cpp` pattern

## Next Steps

1. **Remove deprecated code** (after Phase 2 validation): Delete `solver.cpp` and `solver.h`
2. **Implement first metaheuristic**: Use EXTENDING_SOLVERS.md template
3. **Benchmark comparison**: Compare multiple algorithms on instances
4. **Parameter optimization**: Tune solver parameters per method
5. **Parallel extensions**: Add multi-threaded solver variants

## References

- Extension guide: [EXTENDING_SOLVERS.md](EXTENDING_SOLVERS.md)
- Solver manual: [solver/assets/solve_manual.txt](solver/assets/solve_manual.txt)
- Factory implementation: [solver/src/solver_factory.cpp](solver/src/solver_factory.cpp)
- Base interface: [solver/include/solver_base.h](solver/include/solver_base.h)
- Greedy implementation: [solver/src/greedy_solver.cpp](solver/src/greedy_solver.cpp)
