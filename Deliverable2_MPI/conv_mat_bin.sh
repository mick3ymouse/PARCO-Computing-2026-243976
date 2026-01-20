#!/bin/bash

echo "=========================================="
echo "  SparseMatrix: LAUNCH CONVERSION SCRIPT"
echo "=========================================="

# ======================================
# LOAD MODULES
# ======================================

echo ">>> Loading required module..."

if ! module list 2>&1 | grep -q "python-3.8.13"; then
        echo "    Loading python-3.8.13..."
        module load "python-3.8.13"
        
        if [ $? -ne 0 ]; then
            echo "    ERROR: Failed to load python-3.8.1."
            exit 1
        fi
else
    echo "    python-3.8.13 is already loaded."
fi

echo ">>> Module loaded successfully."

# ======================================
# MAIN SCRIPT
# ======================================

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo ">>> Creating virtual environment (.venv)..."
    python3 -m venv .venv
else
    echo ">>> Virtual environment .venv found."
fi

# Activate virtual environment
echo ">>> Activating .venv and installing dependencies..."
source .venv/bin/activate

REQ_FILE="requirements/req_convert_bin.txt"
if [ -f "$REQ_FILE" ]; then
    pip install -r "$REQ_FILE" --quiet
    if [ $? -ne 0 ]; then
        echo "   Dependency installation failed."
        deactivate
        exit 1
    fi
    echo "   Dependencies installed successfully."
else
    echo "   File $REQ_FILE not found!"
    deactivate
    exit 1
fi

deactivate

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