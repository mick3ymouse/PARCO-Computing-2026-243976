#!/bin/bash

# ======================================
# LOAD MODULES
# ======================================

echo ">>> Loading required modules..."

MODULES_TO_LOAD=("gompi/2023a" "CMake/3.26.3-GCCcore-12.3.0" "make/4.4.1-GCCcore-12.3.0")

# Check and load each module
for module_name in "${MODULES_TO_LOAD[@]}"; do
    if ! module list 2>&1 | grep -q "$module_name"; then
        echo "    Loading $module_name..."
        module load "$module_name"
        
        if [ $? -ne 0 ]; then
            echo "    ERROR: Failed to load $module_name."
            exit 1
        fi
    else
        echo "    $module_name is already loaded."
    fi
done

echo ">>> Modules loaded successfully."

# Start compilation
echo ""
echo ">>> Starting Compilation..."

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Run CMake and Make
# We build in Release mode for max performance (-O3)
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4

# Check if compilation was successful
if [ ! -f ./spmv_strong_exec ]; then
    echo "   Compilation failed! Executable not found."
    exit 1
fi

echo ""
echo ">>> Compilation Successful."
echo ""

cd .. # Return to project root

# Create logs directory if it doesn't exist
mkdir -p logs/strong_scaling
mkdir -p logs/weak_scaling

# ======================================
# MAIN SCRIPT
# ======================================

# Submit the strong scaling job
echo ">>> Submitting strong scaling job..."
qsub scripts/run_strong_scaling.pbs