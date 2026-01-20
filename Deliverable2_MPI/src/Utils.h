#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>
#include <mpi.h>
#include "SparseMatrix.h"

using namespace std;

/**
 * @brief Generates a random vector distributed across processes using rand().
 * * Strategy:
 * 1. Calculates how many elements each process must generate (Load Balancing).
 * 2. Uses rand() seeded with time + rank to generate unique sequences.
 * 3. Assembles the full vector using MPI_Allgatherv.
 * * @param global_dimension Total size of the vector.
 * @param rank MPI Rank.
 * @param size MPI Size.
 * @param full_vec Output vector (automatically resized).
 */
void generate_distributed_vector(int global_dimension, int rank, int size, vector<double>& full_vec);

/**
 * @brief Appends performance metrics to the specified CSV file.
 * @param size Number of MPI processes.
 * @param time_p90 The 90th percentile of execution time.
 * @param gflops The calculated GFLOPS based on P90 time.
 */
void log_metrics_csv(const string& output_path, const string& matrix_name, int size, double time_p90, double gflops);

/**
 * @brief Run the SpMV benchmark.
 * 1. Uses the provided x_vector for computations.
 * 2. Runs a Warmup.
 * 3. Executes the loop 10 times.
 * 4. Collects times and computes the 90th percentile.
 * 5. Calls log_metrics_csv() to report results.
 * * @param mat Local CSR Matrix partition.
 * @param x_vector The input vector (Already generated and distributed).
 * @param matrix_name Name of the matrix (for logging).
 * @param output_path Path to the CSV output file.
 * @param total_cols Global width of the matrix (size of vector x).
 * @param total_nnz Total non-zeros (for GFLOPS calculation).
 * @param rank MPI Rank.
 * @param size MPI Size.
 */
void run_spmv_benchmark(const CSRMatrix& mat, const vector<double>& x_vector, 
                        const string& matrix_name, const string& output_path, 
                        int total_nnz, int rank, int size);

#endif // UTILS_H