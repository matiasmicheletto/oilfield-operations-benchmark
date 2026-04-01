#!/usr/bin/env bash
set -euo pipefail

# -------------------------------
# User-configurable parameters
# -------------------------------
CONFIG_FILE="generator_config.yaml"
NUM_INSTANCES=2
N_WELLS=10
N_BATTERIES=2
DRY_RUN=false # Set to true to print commands without executing them
CPLEX_BIN="" # Optional absolute path to CPLEX executable if not in PATH

# Optional: override directories if needed
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERATOR_DIR="$ROOT_DIR/generator"
SOLVER_DIR="$ROOT_DIR/solver"
INSTANCES_DIR="$GENERATOR_DIR/instances"
OUTPUT_DIR="$ROOT_DIR/output"
CPLEX_OUTPUT_DIR="$OUTPUT_DIR/cplex"
GREEDY_OUTPUT_DIR="$OUTPUT_DIR/greedy"
SCIP_OUTPUT_DIR="$OUTPUT_DIR/scip"

# -------------------------------
# Python environment activation
# -------------------------------
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

# -------------------------------
# Python dependencies
# -------------------------------
REQ_FILE="$GENERATOR_DIR/requirements.txt"
if [[ ! -f "$REQ_FILE" ]]; then
  echo "Error: requirements file not found at '$REQ_FILE'."
  exit 1
fi

echo "Installing Python dependencies from requirements.txt"
python -m pip install -r "$REQ_FILE"

# -------------------------------
# Dependency checks
# -------------------------------
if ! command -v python >/dev/null 2>&1; then
  echo "Error: 'python' is not available in PATH."
  exit 1
fi

mkdir -p "$INSTANCES_DIR" "$CPLEX_OUTPUT_DIR" "$GREEDY_OUTPUT_DIR" "$SCIP_OUTPUT_DIR"

# Clear previous instances and optimizer output so stale results don't mix with new ones
if [[ "$DRY_RUN" != "true" ]]; then
  find "$INSTANCES_DIR"     -maxdepth 1 -name "*.dat" -delete
  find "$INSTANCES_DIR"     -maxdepth 1 -name "*.zpl" -delete
  find "$INSTANCES_DIR"     -maxdepth 1 -name "*.lp"  -delete
  find "$CPLEX_OUTPUT_DIR"  -maxdepth 1 -name "*.sol" -delete
  find "$GREEDY_OUTPUT_DIR" -maxdepth 1 -name "*.txt" -delete
  find "$SCIP_OUTPUT_DIR"   -maxdepth 1 -name "*.txt" -delete
fi

if [[ "$DRY_RUN" == "true" ]]; then
  echo "DRY_RUN is enabled. Commands will be printed but not executed."
fi

echo "[1/5] Generating instances with main.py"
gen_cmd=(
  python "$GENERATOR_DIR/main.py" "$GENERATOR_DIR/$CONFIG_FILE"
  --set "general.num_instances=${NUM_INSTANCES}"
  --set "general.n_wells=${N_WELLS}"
  --set "general.n_batteries=${N_BATTERIES}"
  --set "general.output_dir=${INSTANCES_DIR}"
)
echo "  - ${gen_cmd[*]}"
if [[ "$DRY_RUN" != "true" ]]; then
  "${gen_cmd[@]}"
fi

echo "[2/5] Converting ZPL models to LP in $INSTANCES_DIR"
# LP files are generated directly by main.py (step 1) via lp_generator.py,
# so zimpl is only needed if you want to re-convert ZPL files independently.
if ! command -v zimpl >/dev/null 2>&1; then
  echo "Note: 'zimpl' is not in PATH — skipping zimpl conversion (LP files already written by generator)."
else
  shopt -s nullglob
  zpl_files=("$INSTANCES_DIR"/*.zpl)
  if (( ${#zpl_files[@]} == 0 )); then
    echo "Warning: no .zpl files found in '$INSTANCES_DIR'."
  else
    (
      cd "$INSTANCES_DIR"
      for zpl in *.zpl; do
        echo "  - zimpl -t lp $zpl"
        if [[ "$DRY_RUN" != "true" ]]; then
          lp_out="${zpl%.zpl}.lp"
          rm -f "$lp_out"
          if ! zimpl -t lp "$zpl"; then
            echo "Error: zimpl failed for '$zpl'."
            echo "Hint: inspect '$zpl' for model/data consistency issues."
            exit 1
          fi
          if [[ ! -f "$lp_out" ]]; then
            echo "Error: zimpl completed but '$lp_out' was not created."
            exit 1
          fi
        fi
      done
    )
  fi
fi

echo "[3/5] Solving LP models with CPLEX and writing solutions to $CPLEX_OUTPUT_DIR"
# Resolve CPLEX executable. In DRY_RUN mode this is optional.
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
else
  shopt -s nullglob
  lp_files=("$INSTANCES_DIR"/*.lp)
  if (( ${#lp_files[@]} == 0 )); then
    echo "Warning: no .lp files found in '$INSTANCES_DIR'."
  else
    for lp_path in "${lp_files[@]}"; do
      lp_name="$(basename "$lp_path")"
      stem="${lp_name%.lp}"
      sol_path="$CPLEX_OUTPUT_DIR/${stem}.sol"

      echo "  - Solving $lp_name"
      if [[ "$DRY_RUN" == "true" ]]; then
        echo "    $CPLEX_BIN -c \"read $lp_path\" \"optimize\" \"write $sol_path\" \"quit\""
      else
        "$CPLEX_BIN" -c \
          "read $lp_path" \
          "optimize" \
          "write $sol_path" \
          "quit"
      fi
    done
  fi
  echo "Done. ILP solutions written to: $CPLEX_OUTPUT_DIR"
fi

# -----------------------------------------------------------------------
# Step 4: SCIP solver
# -----------------------------------------------------------------------
echo "[4/5] Solving ZPL models with SCIP and writing solutions to $SCIP_OUTPUT_DIR"

if ! command -v scip >/dev/null 2>&1; then
  echo "Warning: 'scip' is not available in PATH — skipping step 5."
  echo "         Install SCIP or add it to PATH to enable ZPL solving."
else
  shopt -s nullglob
  lp_files_scip=("$INSTANCES_DIR"/*.lp)
  if (( ${#lp_files_scip[@]} == 0 )); then
    echo "Warning: no .lp files found in '$INSTANCES_DIR' — run step 2 (zimpl) first."
  else
    for lp_path in "${lp_files_scip[@]}"; do
      lp_name="$(basename "$lp_path")"
      stem="${lp_name%.lp}"
      sol_path="$SCIP_OUTPUT_DIR/${stem}.txt"

      echo "  - Solving $lp_name -> $(basename "$sol_path")"
      if [[ "$DRY_RUN" == "true" ]]; then
        echo "    scip -f $lp_path | tee $sol_path"
      else
        scip -f "$lp_path" | tee "$sol_path"
      fi
    done
  fi
  echo "Done. SCIP solutions written to: $SCIP_OUTPUT_DIR"
fi

# -----------------------------------------------------------------------
# Step 5: Greedy heuristic solver
# -----------------------------------------------------------------------
echo "[5/5] Running greedy heuristic solver on all instances"

SOLVER_BIN="$SOLVER_DIR/bin/solve"
SOLVER_CONFIG="$SOLVER_DIR/solver_config.yaml"


# Check if makefile exists before checking for binary, to give a more helpful error message
if [[ ! -f "$SOLVER_DIR/Makefile" ]]; then
  echo "Error: Makefile not found in solver directory at '$SOLVER_DIR/Makefile'."
  echo "Create a Makefile in the solver directory to build the solver binary."
  exit 1
fi

# Check if dry run and binary existence separately to allow dry run without a built solver
if [[ "$DRY_RUN" == "true" ]]; then
  echo "  - (dry run) cd $SOLVER_DIR && make"
else
  echo "  - Building solver binary with 'make' in $SOLVER_DIR"
  (cd "$SOLVER_DIR" && make)
  if [[ ! -x "$SOLVER_BIN" ]]; then
    echo "Error: build succeeded but binary not found at '$SOLVER_BIN'."
    echo "Check that your Makefile outputs the binary to bin/solve."
    exit 1
  fi

  echo "  - Built: $SOLVER_BIN"
fi  


param_files=("$INSTANCES_DIR"/parameters_*.dat)
if (( ${#param_files[@]} == 0 )); then
  echo "Warning: no parameters_*.dat files found in '$INSTANCES_DIR'. Skipping step 4."
else
  for param_path in "${param_files[@]}"; do
    param_name="$(basename "$param_path")"
    # Derive stem: parameters_<stem>.dat -> <stem>
    stem="${param_name#parameters_}"
    stem="${stem%.dat}"

    bat_path="$INSTANCES_DIR/batteries_${stem}.dat"
    dist_path="$INSTANCES_DIR/distance_${stem}.dat"

    if [[ ! -f "$bat_path" ]]; then
      echo "  Warning: battery file not found for instance '$stem', skipping."
      continue
    fi
    if [[ ! -f "$dist_path" ]]; then
      echo "  Warning: distance file not found for instance '$stem', skipping."
      continue
    fi

    sol_path="$GREEDY_OUTPUT_DIR/greedy_${stem}.txt"
    echo "  - Solving instance '$stem' -> $(basename "$sol_path")"
    if [[ "$DRY_RUN" == "true" ]]; then
      echo "    $SOLVER_BIN -c $SOLVER_CONFIG -p $param_path -b $bat_path -d $dist_path -o $sol_path"
    else
      "$SOLVER_BIN" \
        -c "$SOLVER_CONFIG" \
        -p "$param_path" \
        -b "$bat_path" \
        -d "$dist_path" \
        -o "$sol_path"
    fi
  done
fi

echo "Done. Greedy solutions written to: $GREEDY_OUTPUT_DIR"
