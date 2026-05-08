#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=pipelines/lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

init_defaults
parse_pipeline_args "$@"
init_paths
ensure_base_dirs

echo "[4/6] Running optimization solvers on all instances"

SOLVER_BIN="$SOLVER_DIR/bin/solve"
SOLVER_CONFIG="$SOLVER_DIR/solver_config.yaml"

build_solver_binary

shopt -s nullglob
param_files=("$INSTANCES_DIR"/parameters_*.dat)
if (( ${#param_files[@]} == 0 )); then
  echo "Warning: no parameters_*.dat files found in '$INSTANCES_DIR'. Skipping step 4."
  exit 0
fi

param_path=""
for param_path in "${param_files[@]}"; do
  param_name="$(basename "$param_path")"
  param_stem="${param_name#parameters_}"
  param_stem="${param_stem%.dat}"

  bat_files=("$INSTANCES_DIR"/batteries_${param_stem}_*.dat)
  if (( ${#bat_files[@]} == 0 )); then
    echo "  Warning: no battery files found for instance '$param_stem', skipping."
    continue
  fi

  bat_path=""
  for bat_path in "${bat_files[@]}"; do
    bat_name="$(basename "$bat_path")"
    scenario_stem="${bat_name#batteries_}"
    scenario_stem="${scenario_stem%.dat}"
    dist_path="$INSTANCES_DIR/distance_${scenario_stem}.dat"

    if [[ ! -f "$dist_path" ]]; then
      echo "  Warning: distance file not found for scenario '$scenario_stem', skipping."
      continue
    fi

    opt_method=""
    for opt_method in "${OPTIMIZATION_METHODS[@]}"; do
      opt_output_dir="${METHOD_OUTPUT_DIRS[$opt_method]}"

      sort_method=""
      for sort_method in "${SORT_METHODS[@]}"; do
        sol_path="$opt_output_dir/${opt_method}_${scenario_stem}_${sort_method}.txt"
        routes_path="$opt_output_dir/routes_${scenario_stem}_${sort_method}.txt"

        echo "  - Solving instance '$param_stem' scenario '$scenario_stem' optimizer='$opt_method' sort_method='$sort_method' -> $(basename "$sol_path")"

        if [[ "$DRY_RUN" == "true" ]]; then
          echo "    $SOLVER_BIN -c $SOLVER_CONFIG -p $param_path -b $bat_path -d $dist_path --method $opt_method --set solver.sort_method=$sort_method -f -o $sol_path > $routes_path"
        else
          "$SOLVER_BIN" \
            -c "$SOLVER_CONFIG" \
            -p "$param_path" \
            -b "$bat_path" \
            -d "$dist_path" \
            --method "$opt_method" \
            --set "solver.sort_method=$sort_method" \
            -f \
            -o "$sol_path" > "$routes_path"
        fi
      done
    done
  done
done

opt_method=""
for opt_method in "${OPTIMIZATION_METHODS[@]}"; do
  echo "Done. $opt_method solutions written to: ${METHOD_OUTPUT_DIRS[$opt_method]}"
done
