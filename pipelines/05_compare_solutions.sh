#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=pipelines/lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

init_defaults
parse_pipeline_args "$@"
init_paths
setup_python_environment

echo "[5/6] Comparing solutions and writing results"

COMPARE_SCRIPT="$OUTPUT_DIR/compare_solutions.py"
if [[ ! -f "$COMPARE_SCRIPT" ]]; then
  echo "Warning: compare script not found at '$COMPARE_SCRIPT' — skipping step 5."
  exit 0
fi

opt_method=""
for opt_method in "${OPTIMIZATION_METHODS[@]}"; do
  opt_output_dir="${METHOD_OUTPUT_DIRS[$opt_method]}"
  method_csv="$OUTPUT_DIR/benchmark_${opt_method}.csv"
  echo "  - Comparing '$opt_method' solutions and writing to $method_csv"

  if [[ "$DRY_RUN" == "true" ]]; then
    echo "    python $COMPARE_SCRIPT --greedy $opt_output_dir --csv $method_csv"
  else
    python "$COMPARE_SCRIPT" --greedy "$opt_output_dir" --csv "$method_csv"
  fi
done
