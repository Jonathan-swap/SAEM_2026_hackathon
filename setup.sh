#!/bin/bash
# Creates venv, installs requirements, and creates the out folders needed for task 1/2

set -e   # exit on first error

# ---- Python virtual environment ----
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

# ---- Install Python requirements ----
echo "Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# ---- Create output folders ----
echo "Creating output folders..."
mkdir -p task1_drug_identifier/out
mkdir -p task2_disposition/out

# ---- R packages ----
echo "Setup complete."
