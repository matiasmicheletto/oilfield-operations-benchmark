#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=pipelines/lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

init_defaults
parse_pipeline_args "$@"
init_paths

echo "[2/6] Converting ZPL models to LP in $INSTANCES_DIR"

if ! command -v zimpl >/dev/null 2>&1; then
  echo "Note: 'zimpl' is not in PATH — skipping zimpl conversion (LP files already written by generator)."
  exit 0
fi

shopt -s nullglob
zpl_files=("$INSTANCES_DIR"/*.zpl)
if (( ${#zpl_files[@]} == 0 )); then
  echo "Warning: no .zpl files found in '$INSTANCES_DIR'."
  exit 0
fi

(
  cd "$INSTANCES_DIR"
  local_zpl=""
  for local_zpl in *.zpl; do
    echo "  - zimpl -t lp $local_zpl"
    if [[ "$DRY_RUN" != "true" ]]; then
      lp_out="${local_zpl%.zpl}.lp"
      rm -f "$lp_out"
      if ! zimpl -t lp "$local_zpl"; then
        echo "Error: zimpl failed for '$local_zpl'."
        echo "Hint: inspect '$local_zpl' for model/data consistency issues."
        exit 1
      fi
      if [[ ! -f "$lp_out" ]]; then
        echo "Error: zimpl completed but '$lp_out' was not created."
        exit 1
      fi
    fi
  done
)
