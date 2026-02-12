#!/usr/bin/env bash

set -e  # Stop on first error

VENV_DIR="venv"
REQUIREMENTS_FILE="requirements.txt"
DEFAULT_CONFIG="default_config.yaml"

echo "----------------------------------------"
echo "Oilfield Operations Benchmark Runner"
echo "----------------------------------------"

# 1️⃣ Resolve config file
if [ $# -eq 0 ]; then
    CONFIG_FILE="$DEFAULT_CONFIG"
    echo "No config provided. Using default: $CONFIG_FILE"
else
    CONFIG_FILE="$1"
fi

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found."
    exit 1
fi

# 2️⃣ Create virtual environment if missing
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists."
fi

# 3️⃣ Activate venv
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# 4️⃣ Install dependencies
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Installing dependencies..."
    pip install --upgrade pip > /dev/null
    pip install -r "$REQUIREMENTS_FILE"
else
    echo "Warning: requirements.txt not found."
fi

# 5️⃣ Run scripts
echo "Running instance generator..."
python3 generate_instances.py "$CONFIG_FILE"

echo "Running distance matrix generator..."
python3 generate_distance_matrix.py "$CONFIG_FILE"

echo "----------------------------------------"
echo "Done."
echo "----------------------------------------"
