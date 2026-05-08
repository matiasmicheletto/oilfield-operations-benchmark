#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=pipelines/lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

init_defaults
parse_pipeline_args "$@"
init_paths
ensure_base_dirs

echo "[3/6] Solving LP models with CPLEX and writing solutions to $CPLEX_OUTPUT_DIR"

if [[ -n "$CPLEX_BIN" ]]; then
  if [[ ! -x "$CPLEX_BIN" ]]; then
    echo "Warning: configured CPLEX_BIN is not executable: '$CPLEX_BIN' — skipping step 3."
    CPLEX_BIN=""
  fi
else
  if command -v cplex >/dev/null 2>&1; then
    CPLEX_BIN="$(command -v cplex)"
  fi
fi

if [[ -z "$CPLEX_BIN" ]]; then
  echo "Warning: 'cplex' is not available in PATH and CPLEX_BIN is not set — skipping step 3."
  echo "         Set CPLEX_BIN or add cplex to PATH to enable ILP solving."
  exit 0
fi

shopt -s nullglob
lp_files=("$INSTANCES_DIR"/*.lp)
if (( ${#lp_files[@]} == 0 )); then
  echo "Warning: no .lp files found in '$INSTANCES_DIR'."
  exit 0
fi

lp_path=""
for lp_path in "${lp_files[@]}"; do
  lp_name="$(basename "$lp_path")"
  stem="${lp_name%.lp}"
  sol_path="$CPLEX_OUTPUT_DIR/${stem}.sol"

  echo "  - Solving $lp_name"
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "    $CPLEX_BIN -c \"read $lp_path\" \"set timelimit $CPLEX_TIME_LIMIT\" \"set mip tolerances mipgap 0.005\" \"set output writelevel 3\" \"optimize\" \"write $sol_path\" \"quit\""
  else
    "$CPLEX_BIN" -c \
      "read $lp_path" \
      "set timelimit $CPLEX_TIME_LIMIT" \
      "set mip tolerances mipgap 0.005" \
      "set output writelevel 3" \
      "optimize" \
      "write $sol_path" \
      "quit"
  fi
done

echo "Done. ILP solutions written to: $CPLEX_OUTPUT_DIR"
