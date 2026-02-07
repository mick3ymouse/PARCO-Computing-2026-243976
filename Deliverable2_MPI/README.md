# Parallel Computing Project: SpMV with MPI

This repository contains a scalable implementation of the **Sparse Matrix-Vector Multiplication (SpMV)** kernel designed for distributed-memory architectures using **MPI** (Message Passing Interface).

The project implements a **1D Row Partitioning** strategy with **Halo Exchange** to handle dependencies across nodes. 
It focuses on Strong and Weak scaling analysis, alongside different metrics for load balancing (NNZ distribution and Communication Volume).

## Prerequisites

The project is designed to be executed in a **Linux environment** (specifically an HPC Cluster with PBS).

* **MPI Implementation:** `gompi` toolchain (GCC + OpenMPI).
* **Build System:** `CMake` and `Make`.
* **Python:** For matrix conversion and plotting.
* **Python Libraries:** `numpy`, `scipy`, `pandas`, `matplotlib`.

## Initial Setup

### 1. Dataset Preparation
Input matrices are not included in the repository due to size constraints.
1.  Navigate to the `matrices/` directory.
2.  Read the `matrices/README.md` file for download links (Matrix Market format `.mtx`).
3.  Download the matrices and place them inside the `matrices/` folder.

> **Note:** Do not worry about converting them to binary manually. **Step 1** of the execution toolchain will handle the conversion from `.mtx` to the custom binary format required for MPI I/O.

### 2. Python Environment
The bash scripts provided (`conv_mat_bin.sh`, `run_plot.sh`) are designed to **automatically** create a virtual environment (`.venv`) and install dependencies from `requirements.txt` if missing.
However, you can set it up manually to ensure everything is correct:

```bash
cd /path/to/Deliverable2_MPI/
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Execution Toolchain

> **NOTE:** Run these commands from the project’s **root directory** (`Deliverable2_MPI/`).

The workflow is divided into 3 automated steps using Bash scripts that handle module loading and PBS job submission.

### 1. MATRIX CONVERSION
Converts the downloaded Matrix Market (`.mtx`) files into a binary CSR format optimized for Parallel MPI I/O. This step is required before running simulations.
* **Script:** `conv_mat_bin.sh`
* **Action:** Loads Python module, sets up the environment, and submits `scripts/convert_bin.pbs`.
* **Command:**
  
    ```bash
    bash conv_mat_bin.sh
    ```
    
* **Output:** Generates binary files (`row_ptr.bin`, `col_ind.bin`, `val.bin`, `meta.bin`) inside each matrix folder.

### 2. COMPILATION & BENCHMARKS
Loads the MPI, CMake and Make modules, compiles the C++ source code in `Release` mode (with `-O3` optimizations), and submits both **Strong Scaling** and **Weak Scaling** jobs to the cluster.
* **Script:** `run_simulations.sh`
* **Action:**
    * Loads modules: `gompi/2023a`, `CMake`, `make`.
    * Compiles code into the `build/` directory.
    * Submits `scripts/run_strong_scaling.pbs` and `scripts/run_weak_scaling.pbs`.
* **Command:**
  
    ```bash
    bash run_simulations.sh
    ```
    
* **Output:**
    * Executable: `build/spmv_mpi`
    * Raw logs: `logs/strong_scaling/` and `logs/weak_scaling/`
    * CSV Results: `results/strong_scaling.csv`, `results/weak_scaling.csv`, `results/load_balance_strong.csv` and `results/load_balance_weak.csv`.

### 3. PLOT GENERATION
Processes the CSV results generated in Step 2 to produce performance graphs (Speedup, Efficiency, GFLOPS) and Load Balancing visualization.
* **Script:** `run_plot.sh`
* **Action:** Loads Python module and submits `scripts/plot.pbs`.
* **Command:**
  
    ```bash
    bash run_plot.sh
    ```
    
* **Output:** PNG images saved in the `plots/` directory.

---

## NOTE for the Cluster

Since the bash scripts handle module loading and PBS submission internally:

### 1. EXECUTION LOCATION
You can safely run the 3 bash scripts from the **LOGIN NODE**. They do not perform heavy computations directly; they compile the code (lightweight) and dispatch the heavy work (Conversion, Simulation, Plotting) to the Compute Nodes via `qsub`.

### 2. MODULES
The scripts are hardcoded to load the following modules available on the cluster:
* `Python/3.11.3-GCCcore-12.3.0` (for conversion and plotting)
* `gompi/2023a` (GCC + OpenMPI for C++ execution)
* `CMake/3.26.3` & `make/4.4.1` (for build)
> **Note:** Ensure that the module names defined in the `.sh` and `.pbs` files correspond to those available on your system before execution.

### 3. QUEUE CONFIGURATION
If you need to change the queue name, walltime, or resource allocation (e.g., number of nodes/cores), you must edit the **header section** inside the `.pbs` files located in the `scripts/` folder **before** running the bash wrappers.

### 4. JOB MONITORING
Use standard cluster commands to check the status of your jobs:
```bash
qstat -u <your_username>
```

