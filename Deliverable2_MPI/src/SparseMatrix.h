#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

#include <vector>
#include <string>
#include <mpi.h>

using namespace std;

/**
 * @brief Struct to hold the LOCAL portion of the CSR Matrix.
 */
struct CSRMatrix {
    int local_rows;         // Number of rows managed by this rank
    int global_start_row;   // Global index of the first row (for reference)
    int local_nnz;          // Number of non-zero elements in this partition

    // CSR Vectors (Local slice)
    vector<int> row_ptr;     // Size: local_rows + 1
    vector<int> col_ind;     // Contiene SOLO indici locali: [0 ... local_rows + n_ghosts - 1]
    vector<double> values;   // Size: local_nnz

    // Mappatura per i Ghost
    // L'elemento locale (local_rows + i) corrisponde all'ID Globale ghost_ids[i]
    vector<int> ghost_ids; 

    /**
     * @brief Performs the Matrix-Vector multiplication: y_local = A_local * x_global
     * @param x The dense input vector (must be fully replicated on this process).
     * @return vector<double> The local portion of the result vector y.
     */
    vector<double> multiply(const vector<double>& x_local) const;
};

/**
 * @brief Helper to read global dimensions from meta.bin
 * Format: [total_rows, total_cols, total_nnz]
 */
void read_global_metadata(const string& filename, int& rows, int& cols, int& nnz);

/**
 * @brief Reads the binary matrix files in parallel using MPI I/O.
 * Each rank independently calculates its offset and reads its portion.
 * @param folder_path Path containing row_ptr.bin, col_ind.bin, val.bin, meta.bin
 * @param rank MPI Rank
 * @param size MPI Size
 * @param mat Output structure to fill
 */
void load_matrix_parallel(const string& folder_path, int rank, int size, CSRMatrix& mat);

/**
 * @brief Converts global column indices to local indices, handling ghost nodes.
 * Updates the CSRMatrix in place.
 * @param mat The CSRMatrix to convert.
 * @param rank MPI Rank
 * @param size MPI Size
 */
void convert_matrix_to_local(CSRMatrix& mat);

#endif // SPARSE_MATRIX_H