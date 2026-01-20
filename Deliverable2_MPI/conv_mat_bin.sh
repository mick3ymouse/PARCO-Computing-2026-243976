#!/bin/bash

echo "=========================================="
echo "  SparseMatrix: LAUNCH CONVERSION SCRIPT"
echo "=========================================="

# ======================================
# LOAD MODULES
# ======================================

echo ">>> Loading required module..."

if ! module list 2>&1 | grep -q "Python/3.11.3-GCCcore-12.3.0"; then
        echo "    Loading Python/3.11.3-GCCcore-12.3.0..."
        module load "Python/3.11.3-GCCcore-12.3.0"
        
        if [ $? -ne 0 ]; then
            echo "    ERROR: Failed to load Python/3.11.3-GCCcore-12.3.0."
            exit 1
        fi
else
    echo "    Python/3.11.3-GCCcore-12.3.0 is already loaded."
fi

echo ">>> Module loaded successfully."

# ======================================
# MAIN SCRIPT
# ======================================

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo ">>> Creating virtual environment (.venv)..."
    python3 -m venv .venv

    # Activate virtual environment and install dependencies
    echo ">>> Activating .venv and installing dependencies..."
    source .venv/bin/activate
    pip install --upgrade pip --quiet

    if [ -f "requirements/req_convert_bin.txt" ]; then
        pip install -r "requirements/req_convert_bin.txt"
    else
        echo "   File requirements/req_convert_bin.txt not found!"
        exit 1
    fi
    deactivate
    echo ">>> Virtual environment setup complete."
else
    echo ">>> Virtual environment .venv found, skipping creation."
fi

# Create logs directory if it doesn't exist
mkdir -p logs/conversion

# Submit the conversion job to the scheduler
echo ">>> Submitting conversion job to scheduler..."

PBS_SCRIPT="scripts/convert_bin.pbs"
if [ -f "$PBS_SCRIPT" ]; then
    qsub "$PBS_SCRIPT"
else
    echo "   PBS script $PBS_SCRIPT not found."
    exit 1
fi

echo ">>> Conversion job submitted."