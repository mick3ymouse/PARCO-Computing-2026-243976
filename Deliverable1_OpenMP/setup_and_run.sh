#!/bin/bash
#
# Script to compile and launch the SpMV benchmarks.
#
# Usage:
#   ./setup_and_run.sh [option]
#
# Options:
#   timings : Submit only the timings benchmark job
#   cache   : Submit only the cache benchmark job
#   all     : Submit both jobs (default)
#

echo "=========================================="
echo "  SpmvBenchmark SETUP & RUN SCRIPT"
echo "=========================================="

# ======================================
# SET TARGET
# ======================================
# Read the first command-line argument, default to "all"
TARGET=${1:-all}
echo ">>> Selected target: $TARGET"

# ======================================
# LOAD MODULES
# ======================================
echo ">>> Loading required modules..."

# List of required modules
MODULES_TO_LOAD=("gcc91" "cmake-3.15.4" "make-4.3")

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

echo "    Modules loaded successfully."

# Set compiler environment variable
export CXX="g++-9.1.0"

# ======================================
# COMPILE PROJECT
# ======================================
echo ">>> Compiling project with CMake/Make..."

# Create build directory
mkdir -p build

# Configure and compile
cd build
cmake ..
make

# Check compilation status
if [ $? -ne 0 ]; then
    echo "    ERROR: Compilation failed."
    exit 1
fi

# Return to project root
cd ..
echo "    Compilation complete. Executable is at ./build/spmv_benchmark"

# ======================================
# SUBMIT PBS JOBS
# ======================================
echo ">>> Submitting PBS job(s)..."

# Ensure directories exist
cd "$(dirname "$0")" || exit 1
mkdir -p logs
mkdir -p results

# Array to store job IDs
declare -a JOB_IDS=()

# Submit timings job
if [ "$TARGET" == "timings" ] || [ "$TARGET" == "all" ]; then
    echo "    Submitting job: scripts/bench_timings.pbs"
    JOB_OUTPUT=$(qsub scripts/bench_timings.pbs)
    
    if [ $? -eq 0 ]; then
        JOB_IDS+=("$JOB_OUTPUT")
        echo "    Job submitted: $JOB_OUTPUT"
    else
        echo "    ERROR: qsub submission failed for timings."
    fi
fi

# Submit cache job
if [ "$TARGET" == "cache" ] || [ "$TARGET" == "all" ]; then
    echo "    Submitting job: scripts/bench_cache.pbs"
    JOB_OUTPUT=$(qsub scripts/bench_cache.pbs)
    
    if [ $? -eq 0 ]; then
        JOB_IDS+=("$JOB_OUTPUT")
        echo "    Job submitted: $JOB_OUTPUT"
    else
        echo "    ERROR: qsub submission failed for cache."
    fi
fi

echo ""
echo "=========================================="
echo "  SCRIPT COMPLETE!"
echo "  Job(s) submitted. Check status with 'qstat'."
echo "=========================================="

