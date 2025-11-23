# PARCO-Computing-2026-243976
# Parallel Computing Project: SpMV with OpenMP
This repository contains an optimized implementation of the **Sparse Matrix-Vector Multiplication (SpMV)** kernel using the **Compressed Sparse Row (CSR)** format. The project focuses on performance analysis and scalability on shared-memory multicore architectures using **OpenMP**.

## Prerequisites
The project is designed to be executed in a **Linux environment** (e.g., HPC Cluster).

* **Compiler:** GCC with OpenMP support (`g++` >= 9.1.0 recommended).
* **Python:** Version 3.6+ for analysis and plotting scripts.
* **Python Libraries:** `pandas`, `matplotlib`.
The scripts are configured to use the "Agg" backend for matplotlib, allowing execution on headless clusters (no GUI required).

## Initial Setup

### 1. Dataset Preparation
Due to storage constraints, input matrices are not included in the repository.
1. Navigate to the `matrices/` directory.
2. Read the `matrices/README.txt` file, which contains links to the datasets (Matrix Market format).
3. Download the matrices and place them inside the `matrices/` folder

**Note:** Failing to complete this step will cause the simulations to abort as the input `.mtx` files will be missing.

### 2. Python Environment Setup
It is strictly recommended to use a virtual environment located INSIDE the project root directory to manage dependencies and execution paths correctly.

1.  Ensure you are in the project root folder:

    cmd: `cd /path/to/project_root/`

2.  Create and activate the virtual environment:
    
    cmd: `python3 -m venv .venv`
    
    cmd: `source .venv/bin/activate`

3.  Install required dependencies:

    cmd: `pip install -r requirements.txt`

### NOTE: You must activate this environment ("source .venv/bin/activate") every time you open a new terminal session

## Execution Toolchain 

### (NOTE: run this commands from the project’s root directory)
The analysis pipeline consists of 4 sequential steps: 

**STEP 1: TIMING BENCHMARKS**

Description: Compiles the source code and submits a PBS job to the cluster queue to run raw performance benchmarks. It varies the number of threads and scheduling policies (static, dynamic, guided).

cmd: `bash setup_and_run.sh timings`

Output: 
- Submits a job (e.g., "spmv_timings").
- Raw logs are written to "logs/".
- Generates preliminary CSV timing data once the job completes in "results/".

**STEP 2: TUNING (OPTIONAL for default dataset)**

Description: Analyzes the timing data from Step 3.1 to identify the optimal chunk_size for each matrix.
NOTE: Since "best_configs.csv" is already provided, run this only if you want to re-calibrate the optimal parameters for your specific hardware

cmd: `python3 tuning.py`

Output: 
-Updates the "best_configs.csv" file in "results/".

**STEP 3: HARWARE PROFILING**

Description: Submits a specific PBS job to re-run the kernel using ONLY the optimal configurations found in "best_configs.csv". It uses `perf` tool to capture IPC, L1 Miss Rates, and LLC Miss Rates.

cmd: `bash setup_and_run.sh cache`

Output: 
-Submits a job (e.g., "spmv_cache").
-Generates CSV file containing cache metrics in "results/".

**STEP 4: PLOT GENERATION**

Description: Processes all collected data and generates the final performance figures (Speedup, Efficiency, Cache Analysis).

cmd: `python3 plot_results.py`

Output: PNG images saved in the "plots/" directory.

# NOTE for the Cluster
Since `setup_and_run.sh` handles PBS submission internally:

1. EXECUTION LOCATION: You can safely run `setup_and_run.sh` from the 
   LOGIN NODE. It will not run the heavy computation there; it will simply 
   dispatch the job to the scheduler.

2. QUEUE CONFIGURATION: If you need to change the queue name, walltime, or 
   resource allocation (default: select=1:ncpus=64:mem=8gb), you must edit the 
   header section inside the `.pbs` scripts before running it.

3. EMAIL NOTIFICATIONS: By default, email alerts are disabled to avoid spam. 
   To receive notifications for job start/abort/end:
   - Open the `.pbs` script.
   - Uncomment the lines `# #PBS -m abe` and `# #PBS -M ...`.
   - Replace `your_email@domain.com` with your actual address.

4. JOB MONITORING: Use standard cluster commands to check status:
   cmd: `qstat -u <username>`
