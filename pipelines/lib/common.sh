#!/usr/bin/env bash

# Shared helpers and configuration for pipeline step scripts.
# This file is intended to be sourced by scripts under pipelines/.

set -euo pipefail

pipeline_usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  -n, --instances N    Number of instances to generate (default: ${NUM_INSTANCES:-5})
  -s, --scenarios N    Number of scenarios per instance (default: ${NUM_SCENARIOS:-1})
  -w, --wells N        Number of wells per instance (default: ${N_WELLS:-10})
  -b, --batteries N    Number of batteries per instance (default: ${N_BATTERIES:-2})
  -d, --dry-run        Print commands without executing them
      --keep           Keep existing files whose names do not match the current
                       wells/batteries configuration (default: delete all previous files)
  -h, --help           Show this help message
EOF
}

init_defaults() {
  CONFIG_FILE="generator_config.yaml"
  NUM_INSTANCES=5
  NUM_SCENARIOS=1
  DELETE_PREVIOUS=true
  N_WELLS=10
  N_BATTERIES=2
  DRY_RUN=false
  CPLEX_BIN=""
  CPLEX_TIME_LIMIT=1200
  SORT_METHODS=(priority_cost loss route)
  OPTIMIZATION_METHODS=(greedy hs)
}

parse_pipeline_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -n|--instances)  NUM_INSTANCES="$2";  shift 2 ;;
      -s|--scenarios)  NUM_SCENARIOS="$2"; shift 2 ;;
      -w|--wells)      N_WELLS="$2";        shift 2 ;;
      -b|--batteries)  N_BATTERIES="$2";   shift 2 ;;
      -d|--dry-run)    DRY_RUN=true;         shift   ;;
      --keep)          DELETE_PREVIOUS=false; shift   ;;
      -h|--help)       pipeline_usage; exit 0 ;;
      *) echo "Unknown option: $1" >&2; pipeline_usage; exit 1 ;;
    esac
  done
}

init_paths() {
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  GENERATOR_DIR="$ROOT_DIR/generator"
  SOLVER_DIR="$ROOT_DIR/solver"
  INSTANCES_DIR="$GENERATOR_DIR/instances"
  OUTPUT_DIR="$ROOT_DIR/output"
  CPLEX_OUTPUT_DIR="$OUTPUT_DIR/cplex"
  SCIP_OUTPUT_DIR="$OUTPUT_DIR/scip"

  declare -gA METHOD_OUTPUT_DIRS
  METHOD_OUTPUT_DIRS[greedy]="$OUTPUT_DIR/greedy"
  METHOD_OUTPUT_DIRS[hs]="$OUTPUT_DIR/hs"
}

setup_python_environment() {
  if [[ -f "$GENERATOR_DIR/venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$GENERATOR_DIR/venv/bin/activate"
  elif [[ -f "$GENERATOR_DIR/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$GENERATOR_DIR/.venv/bin/activate"
  elif [[ -f "$ROOT_DIR/venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$ROOT_DIR/venv/bin/activate"
  elif [[ -f "$ROOT_DIR/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$ROOT_DIR/.venv/bin/activate"
  else
    echo "Error: no virtual environment found under '$GENERATOR_DIR' or '$ROOT_DIR'."
    echo "Create one first (for example: python3 -m venv generator/venv)."
    exit 1
  fi
}

install_python_requirements() {
  local req_file="$GENERATOR_DIR/requirements.txt"
  if [[ ! -f "$req_file" ]]; then
    echo "Error: requirements file not found at '$req_file'."
    exit 1
  fi
  echo "Installing Python dependencies from requirements.txt"
  python -m pip install -r "$req_file"
}

ensure_base_dirs() {
  mkdir -p "$INSTANCES_DIR" "$CPLEX_OUTPUT_DIR" "$SCIP_OUTPUT_DIR"
  local opt_method
  for opt_method in "${OPTIMIZATION_METHODS[@]}"; do
    if [[ -z "${METHOD_OUTPUT_DIRS[$opt_method]+x}" ]]; then
      echo "Error: output directory mapping is not defined for optimization method '$opt_method'."
      echo "Add it to METHOD_OUTPUT_DIRS in pipelines/lib/common.sh."
      exit 1
    fi
    mkdir -p "${METHOD_OUTPUT_DIRS[$opt_method]}"
  done
}

clear_previous_outputs() {
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "DRY_RUN is enabled. Commands will be printed but not executed."
    return
  fi

  if [[ "$DELETE_PREVIOUS" == "true" ]]; then
    # Delete all generated files regardless of configuration.
    find "$INSTANCES_DIR"    -maxdepth 1 -name "*.dat" -delete
    find "$INSTANCES_DIR"    -maxdepth 1 -name "*.zpl" -delete
    find "$INSTANCES_DIR"    -maxdepth 1 -name "*.lp"  -delete
    find "$CPLEX_OUTPUT_DIR" -maxdepth 1 -name "*.sol" -delete
    find "$SCIP_OUTPUT_DIR"  -maxdepth 1 -name "*.txt" -delete

    local opt_method
    for opt_method in "${OPTIMIZATION_METHODS[@]}"; do
      find "${METHOD_OUTPUT_DIRS[$opt_method]}" -maxdepth 1 -name "*.txt" -delete
      find "${METHOD_OUTPUT_DIRS[$opt_method]}" -maxdepth 1 -name "*.png" -delete
    done
  else
    # Selective deletion: only remove files whose names embed the current
    # wells/batteries configuration stem, leaving other configurations intact.
    local stem="${N_WELLS}_${N_BATTERIES}"
    find "$INSTANCES_DIR"    -maxdepth 1 -name "*_${stem}_*.dat" -delete
    find "$INSTANCES_DIR"    -maxdepth 1 -name "*_${stem}_*.zpl" -delete
    find "$INSTANCES_DIR"    -maxdepth 1 -name "*_${stem}_*.lp"  -delete
    find "$CPLEX_OUTPUT_DIR" -maxdepth 1 -name "*_${stem}_*.sol" -delete
    find "$SCIP_OUTPUT_DIR"  -maxdepth 1 -name "*_${stem}_*.txt" -delete

    local opt_method
    for opt_method in "${OPTIMIZATION_METHODS[@]}"; do
      find "${METHOD_OUTPUT_DIRS[$opt_method]}" -maxdepth 1 -name "*_${stem}_*.txt" -delete
      find "${METHOD_OUTPUT_DIRS[$opt_method]}" -maxdepth 1 -name "*_${stem}_*.png" -delete
    done
  fi
}

build_solver_binary() {
  local solver_bin="$SOLVER_DIR/bin/solve"

  if [[ ! -f "$SOLVER_DIR/Makefile" ]]; then
    echo "Error: Makefile not found in solver directory at '$SOLVER_DIR/Makefile'."
    echo "Create a Makefile in the solver directory to build the solver binary."
    exit 1
  fi

  if [[ "$DRY_RUN" == "true" ]]; then
    echo "  - (dry run) cd $SOLVER_DIR && make"
    return
  fi

  echo "  - Building solver binary with 'make' in $SOLVER_DIR"
  (cd "$SOLVER_DIR" && make)

  if [[ ! -x "$solver_bin" ]]; then
    echo "Error: build succeeded but binary not found at '$solver_bin'."
    echo "Check that your Makefile outputs the binary to bin/solve."
    exit 1
  fi

  echo "  - Built: $solver_bin"
}
