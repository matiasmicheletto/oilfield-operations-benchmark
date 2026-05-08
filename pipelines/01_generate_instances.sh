#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=pipelines/lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

init_defaults
parse_pipeline_args "$@"
init_paths
setup_python_environment
install_python_requirements
ensure_base_dirs
clear_previous_outputs

if ! command -v python >/dev/null 2>&1; then
  echo "Error: 'python' is not available in PATH."
  exit 1
fi

echo "[1/6] Generating instances with main.py"
gen_cmd=(
  python "$GENERATOR_DIR/main.py" "$GENERATOR_DIR/$CONFIG_FILE"
  --set "general.num_instances=${NUM_INSTANCES}"
  --set "general.num_scenarios=${NUM_SCENARIOS}"
  --set "general.n_wells=${N_WELLS}"
  --set "general.n_batteries=${N_BATTERIES}"
  --set "general.output_dir=${INSTANCES_DIR}"
)

echo "  - ${gen_cmd[*]}"
if [[ "$DRY_RUN" != "true" ]]; then
  "${gen_cmd[@]}"
fi
