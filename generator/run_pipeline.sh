#!/usr/bin/env bash
set -euo pipefail

# -------------------------------
# User-configurable parameters
# -------------------------------
CONFIG_FILE="generator_config.yaml"
NUM_INSTANCES=5
N_WELLS=100
N_BATTERIES=2
DRY_RUN=false # Set to true to print commands without executing them
CPLEX_BIN="" # Optional absolute path to CPLEX executable if not in PATH

# Optional: override directories if needed
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTANCES_DIR="$ROOT_DIR/instances"
OUTPUT_DIR="$ROOT_DIR/output"

# -------------------------------
# Python environment activation
# -------------------------------
if [[ -f "$ROOT_DIR/venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/venv/bin/activate"
elif [[ -f "$ROOT_DIR/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.venv/bin/activate"
else
  echo "Error: no virtual environment found at '$ROOT_DIR/venv' or '$ROOT_DIR/.venv'."
  echo "Create one first (for example: python3 -m venv venv)."
  exit 1
fi

# -------------------------------
# Python dependencies
# -------------------------------
REQ_FILE="$ROOT_DIR/requirements.txt"
if [[ ! -f "$REQ_FILE" ]]; then
  echo "Error: requirements file not found at '$REQ_FILE'."
  exit 1
fi

echo "Installing Python dependencies from requirements.txt"
python -m pip install -r "$REQ_FILE"

# -------------------------------
# Dependency checks
# -------------------------------
for cmd in python zimpl; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Error: '$cmd' is not available in PATH."
    exit 1
  fi
done

# Resolve CPLEX executable. In DRY_RUN mode this is optional.
if [[ -n "$CPLEX_BIN" ]]; then
  if [[ ! -x "$CPLEX_BIN" ]]; then
    echo "Error: configured CPLEX_BIN is not executable: '$CPLEX_BIN'."
    exit 1
  fi
else
  if command -v cplex >/dev/null 2>&1; then
    CPLEX_BIN="$(command -v cplex)"
  fi
fi

if [[ "$DRY_RUN" != "true" && -z "$CPLEX_BIN" ]]; then
  echo "Error: CPLEX executable not found."
  echo "Set CPLEX_BIN to the full path of the cplex binary, or add cplex to PATH."
  exit 1
fi

if [[ "$DRY_RUN" == "true" && -z "$CPLEX_BIN" ]]; then
  echo "Warning: CPLEX executable not found, but DRY_RUN=true so continuing."
fi

mkdir -p "$INSTANCES_DIR"
mkdir -p "$OUTPUT_DIR"

if [[ "$DRY_RUN" == "true" ]]; then
  echo "DRY_RUN is enabled. Commands will be printed but not executed."
fi

echo "[1/3] Generating instances with main.py"
gen_cmd=(
  python "$ROOT_DIR/main.py" "$ROOT_DIR/$CONFIG_FILE"
  --set "general.num_instances=${NUM_INSTANCES}"
  --set "general.n_wells=${N_WELLS}"
  --set "general.n_batteries=${N_BATTERIES}"
)
echo "  - ${gen_cmd[*]}"
if [[ "$DRY_RUN" != "true" ]]; then
  "${gen_cmd[@]}"
fi

echo "[2/3] Converting ZPL models to LP in $INSTANCES_DIR"
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

echo "[3/3] Solving LP models with CPLEX and writing solutions to $OUTPUT_DIR"
lp_files=("$INSTANCES_DIR"/*.lp)
if (( ${#lp_files[@]} == 0 )); then
  echo "Warning: no .lp files found in '$INSTANCES_DIR'."
  exit 0
fi

for lp_path in "${lp_files[@]}"; do
  lp_name="$(basename "$lp_path")"
  stem="${lp_name%.lp}"
  sol_path="$OUTPUT_DIR/${stem}.sol"

  echo "  - Solving $lp_name"
  if [[ "$DRY_RUN" == "true" ]]; then
    cplex_preview="${CPLEX_BIN:-cplex}"
    echo "    $cplex_preview -c \"read $lp_path\" \"optimize\" \"write $sol_path\" \"quit\""
  else
    "$CPLEX_BIN" -c \
      "read $lp_path" \
      "optimize" \
      "write $sol_path" \
      "quit"
  fi
done

echo "Done. Solutions written to: $OUTPUT_DIR"
