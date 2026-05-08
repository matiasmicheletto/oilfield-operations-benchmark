#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running full pipeline via modular step scripts"

"$SCRIPT_DIR/01_generate_instances.sh" "$@"
"$SCRIPT_DIR/02_convert_zpl_to_lp.sh" "$@"
"$SCRIPT_DIR/03_solve_cplex.sh" "$@"
"$SCRIPT_DIR/04_run_solvers.sh" "$@"
"$SCRIPT_DIR/05_compare_solutions.sh" "$@"
"$SCRIPT_DIR/06_plot_routes.sh" "$@"

echo "Pipeline finished successfully."
