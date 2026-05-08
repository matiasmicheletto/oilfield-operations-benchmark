#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=pipelines/lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

init_defaults
parse_pipeline_args "$@"
init_paths
setup_python_environment

echo "[6/6] Plotting crew routes over spatial maps"

PLOT_SCRIPT="$OUTPUT_DIR/plot_routes.py"
if [[ ! -f "$PLOT_SCRIPT" ]]; then
  echo "Warning: plot script not found at '$PLOT_SCRIPT' — skipping step 6."
  exit 0
fi

opt_method=""
for opt_method in "${OPTIMIZATION_METHODS[@]}"; do
  opt_output_dir="${METHOD_OUTPUT_DIRS[$opt_method]}"
  shopt -s nullglob
  routes_files=("$opt_output_dir"/routes_*.txt)

  if (( ${#routes_files[@]} == 0 )); then
    echo "  Note: no routes_*.txt files found in '$opt_output_dir' for method '$opt_method'."
    continue
  fi

  routes_path=""
  for routes_path in "${routes_files[@]}"; do
    routes_name="$(basename "$routes_path")"
    tmp="${routes_name#routes_}"
    sort_method="${tmp##*_}"
    sort_method="${sort_method%.txt}"
    scenario_stem="${tmp%_${sort_method}.txt}"

    spatial_path="$INSTANCES_DIR/spatial_data_${scenario_stem}.npz"
    overlay_path="$opt_output_dir/route_overlay_${scenario_stem}_${sort_method}.png"

    if [[ ! -f "$spatial_path" ]]; then
      echo "  Warning: spatial data not found for '$scenario_stem' — skipping."
      continue
    fi

    echo "  - Plotting routes for '$opt_method'/'$scenario_stem' sort_method='$sort_method' -> $(basename "$overlay_path")"
    if [[ "$DRY_RUN" == "true" ]]; then
      echo "    python $PLOT_SCRIPT --spatial $spatial_path --routes $routes_path --output $overlay_path"
    else
      python "$PLOT_SCRIPT" \
        --spatial "$spatial_path" \
        --routes "$routes_path" \
        --output "$overlay_path"
    fi
  done

  echo "  Done. Route overlays for '$opt_method' written to: $opt_output_dir"
done
