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


// Functions for strong scaling:

/**
 * @brief Logs strong scaling benchmark results to a CSV file.
 * @param output_path Path to the CSV output file.
 * @param matrix_name Name of the matrix (for logging).
 * @param size Number of MPI processes.
 * @param time_p90 The 90th percentile of execution time.
 * @param gflops The calculated GFLOPS based on P90 time.
 */
void log_strong_scaling_csv(const string& output_path, const string& matrix_name, int size, double time_p90, double gflops);

/**
 * @brief Run the Strong Scaling benchmark.
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
void run_strong_scaling(const CSRMatrix& mat, const vector<double>& x_vector, 
                        const string& matrix_name, const string& output_path, 
                        int total_nnz, int rank, int size);


// Functions for weak scaling:

/**
 * @brief Generates a synthetic CSR matrix for weak scaling tests.
 * Each process generates a local portion of the matrix with a fixed number of rows
 * and non-zeros per row.
 * @param local_rows Number of rows for this process.
 * @param global_cols Total number of columns in the global matrix.
 * @param nnz_per_row Number of non-zero elements per row.
 * @param rank MPI Rank.
 * @return CSRMatrix The generated local CSR matrix partition.
 */
CSRMatrix generate_synthetic_matrix(int local_rows, int global_cols, int nnz_per_row, int rank);

/**
 * @brief Logs weak scaling benchmark results to a CSV file.
 * @param output_path Path to the CSV output file.
 * @param size Number of MPI processes.
 * @param global_rows Total number of rows in the global matrix.
 * @param global_nnz Total number of non-zero elements in the global matrix.
 * @param time_p90 The 90th percentile of execution time.
 * @param gflops The calculated GFLOPS based on P90 time.
 */
void log_weak_scaling_csv(const string& output_path, int size, int global_rows, int global_nnz, double time_p90, double gflops);

/**
 * @brief Run the Weak Scaling benchmark.
 * 1. Uses the provided x_vec for computations.
 * 2. Runs a Warmup.
 * 3. Executes the loop 10 times.
 * 4. Collects times and computes the 90th percentile.
 * 5. Calls log_weak_scaling_csv() to report results.
 * * @param mat Local CSR Matrix partition.
 * @param x_vec The input vector (Already generated and distributed).
 * @param output_path Path to the CSV output file.
 * @param rows_per_proc Number of rows assigned to each process.
 * @param global_nnz Total non-zeros in the global matrix (for GFLOPS calculation).
 * @param rank MPI Rank.
 * @param size MPI Size.
 */
void run_weak_scaling(const CSRMatrix& mat, const vector<double>& x_vec, 
                      const string& output_path, int rows_per_proc, 
                      long long global_nnz, int rank, int size);

#endif // UTILS_H