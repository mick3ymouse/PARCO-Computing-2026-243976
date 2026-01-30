# PARCO-Computing-2026-243976

---

# Parallel Sparse Matrix-Vector Multiplication (SpMV)

![C++](https://img.shields.io/badge/Language-C++-00599C?style=flat-square&logo=c%2B%2B)
![Python](https://img.shields.io/badge/Language-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Bash](https://img.shields.io/badge/Script-Bash-4EAA25?style=flat-square&logo=gnu-bash&logoColor=white)
![OpenMP](https://img.shields.io/badge/API-OpenMP-blue?style=flat-square)
![MPI](https://img.shields.io/badge/API-OpenMPI-green?style=flat-square)

## Project Overview
This repository hosts a comprehensive performance analysis of the **Sparse Matrix-Vector Multiplication (SpMV)** kernel, a fundamental operation in scientific computing and engineering.

The project explores parallelization strategies across two distinct memory architectures:
1.  **Shared Memory:** Leveraging **OpenMP** for multi-threaded optimization.
2.  **Distributed Memory:** Leveraging **MPI** for scalability across multiple nodes.

The goal is to evaluate scalability, computational efficiency, and the impact of load balancing on real-world non-symmetric matrices (both structured and unstructured).

---

## Repository Structure

```text
.
├── Deliverable1_OpenMP/    # Shared Memory Implementation (OpenMP)
├── Deliverable2_MPI/       # Distributed Memory Implementation (MPI)
└── README.md
```

## Deliverable 1: Shared Memory (OpenMP)
**Focus: Scheduling Strategies & Strong Scaling**

In this first deliverable, the impact of OpenMP scheduling on SpMV performance was analyzed. Various configurations were tested using real-world non-symmetric matrices, covering both structured and unstructured sparsity patterns.
The study focused exclusively on **strong scaling**, identifying the best scheduling strategy and the ideal **chunk size** for each specific matrix topology.

### Key Features:
* **Scheduling Analysis:** Comparison of `static`, `dynamic`, and `guided` schedules.
* **Tuning:** Determination of the optimal chunk size to minimize overhead and maximize cache locality.
* **Benchmarking:** Detailed performance analysis on real-world datasets.

---

## Deliverable 2: Distributed Memory (MPI)
**Focus: Scalability, Partitioning & Communication**

In the second deliverable, the SpMV analysis was extended to distributed memory architectures using MPI. Both **strong and weak scaling** benchmarks were conducted on the same set of matrices used in the previous deliverable.

Additionally, specific **load balancing metrics** were derived to assess partition quality. The implementation features a fully distributed approach:

* **Parallel I/O:** Reading binary matrix files directly into distributed memory.
* **Distributed Generation:** Generating synthetic vectors and matrices locally on each rank.
* **1D Row Partitioning:** Implementing a row-wise decomposition with a Halo Exchange mechanism to efficiently handle ghost entries.

### Key Features:
* **Scalability:** Strong and Weak scaling analysis.
* **Halo Exchange:** Efficient management of ghost nodes communication.
* **Load Balancing:** Automated collection of NNZ distribution and communication volume statistics.
* **Parallel I/O:** Scalable file reading and synthetic data generation.

---

## Benchmark Approach
Both projects utilize a rigorous benchmarking methodology:

* **Datasets:** Real-world non-symmetric sparse matrices (from the SuiteSparse Matrix Collection).
* **Metrics:** Execution time (90th percentile), Speedup, Efficiency, and GFLOPS.
* **Environment:** Validated on high-performance cluster nodes.
