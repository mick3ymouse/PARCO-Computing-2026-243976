#include "SparseMatrix.h"
#include <iostream>
#include <vector>

using namespace std;

void read_global_metadata(const string& filename, int& rows, int& cols, int& nnz) {
    MPI_File fh;
    if (MPI_File_open(MPI_COMM_WORLD, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh) != MPI_SUCCESS) {
        cerr << "Error: Could not open " << filename << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Read 3 integers from the beginning of the file
    int meta[3];
    MPI_File_read_at(fh, 0, meta, 3, MPI_INT, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);

    rows = meta[0];
    cols = meta[1];
    nnz  = meta[2];
}

void load_matrix_parallel(const string& folder_path, int rank, int size, CSRMatrix& mat) {
    
    // --- STEP 1: Get Global Dimensions ---
    int total_rows, total_cols, total_nnz;
    read_global_metadata(folder_path + "/meta.bin", total_rows, total_cols, total_nnz);

    // --- STEP 2: Calculate 1D Row Partitioning ---
    // We distribute rows as evenly as possible. 
    // If total_rows % size != 0, the first 'remainder' ranks get 1 extra row.
    
    int rows_per_proc = total_rows / size;
    int remainder = total_rows % size;

    if (rank < remainder) {
        mat.local_rows = rows_per_proc + 1;
        mat.global_start_row = rank * (rows_per_proc + 1);
    } else {
        mat.local_rows = rows_per_proc;
        // Offset = (Rows of "heavy" ranks) + (Rows of "standard" ranks before me)
        mat.global_start_row = (remainder * (rows_per_proc + 1)) + 
                               ((rank - remainder) * rows_per_proc);
    }

    // --- STEP 3: Read 'row_ptr' (Parallel Read) ---
    // We need to read (local_rows + 1) integers.
    // Offset is calculated based on the global start row index.
    
    mat.row_ptr.resize(mat.local_rows + 1);
    
    MPI_File fh;
    string ptr_file = folder_path + "/row_ptr.bin";
    
    MPI_File_open(MPI_COMM_WORLD, ptr_file.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    
    MPI_Offset ptr_offset = (MPI_Offset)mat.global_start_row * sizeof(int);
    MPI_File_read_at(fh, ptr_offset, mat.row_ptr.data(), mat.local_rows + 1, MPI_INT, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);

    // --- STEP 4: Determine NNZ range ---
    // The values read from row_ptr are global indices into the values/col_ind arrays.
    int global_nnz_start = mat.row_ptr[0];
    int global_nnz_end   = mat.row_ptr[mat.local_rows];
    
    mat.local_nnz = global_nnz_end - global_nnz_start;

    // Normalize row_ptr: local indices must start from 0 for SpMV
    for (int& val : mat.row_ptr) {
        val -= global_nnz_start;
    }

    // --- STEP 5: Read 'col_ind' and 'values' (Parallel Read) ---
    // Each rank jumps to 'global_nnz_start' in these files and reads 'local_nnz' items.

    mat.col_ind.resize(mat.local_nnz);
    mat.values.resize(mat.local_nnz);

    // Offsets in bytes
    MPI_Offset col_offset = (MPI_Offset)global_nnz_start * sizeof(int);
    MPI_Offset val_offset = (MPI_Offset)global_nnz_start * sizeof(double);

    // Read Column Indices
    MPI_File_open(MPI_COMM_WORLD, (folder_path + "/col_ind.bin").c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_File_read_at(fh, col_offset, mat.col_ind.data(), mat.local_nnz, MPI_INT, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);

    // Read Values
    MPI_File_open(MPI_COMM_WORLD, (folder_path + "/val.bin").c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_File_read_at(fh, val_offset, mat.values.data(), mat.local_nnz, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
}


// --- SpMV Implementation ---

vector<double> CSRMatrix::multiply(const vector<double>& x) const {
    // 1. Prepare the local result vector
    // It will have the same number of rows as the local matrix partition
    vector<double> y_local(local_rows, 0.0);

    // 2. Check for Consistency
    // If the input vector is empty, something went wrong in the logic.
    if (x.empty()) {
        cerr << "[Error] Input vector x is empty! Aborting execution." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // 3. CSR Multiplication Kernel
    // NOTE: In 'load_matrix_parallel', we already normalized 'row_ptr' 
    // to start from 0 relative to the local 'values' array.
    // So we can use row_ptr[i] directly as the index.

    for (int i = 0; i < local_rows; ++i) {
        double dot_product = 0.0;

        // Get the range of non-zero elements for the current row
        int start_index = row_ptr[i];
        int end_index   = row_ptr[i+1];

        // Safety Check (Optional - useful for debugging)
        if (start_index < 0 || end_index > local_nnz) {
             cerr << "[Error] Row pointer indices out of bounds. "
                       << "Row: " << i << ", Start: " << start_index << ", End: " << end_index 
                       << ", Local NNZ: " << local_nnz << endl;
             MPI_Abort(MPI_COMM_WORLD, 2);
        }

        // Inner loop: Dot product of the row and vector x
        for (int k = start_index; k < end_index; ++k) {
            
            // col_ind contains global column indices. 
            // x is the global vector, so we access it directly.
            int col_idx = col_ind[k];

            // Segfault prevention check
            if (col_idx >= (int)x.size()) {
                cerr << "[Error] Matrix column index exceeds vector size. "
                          << "Col Index: " << col_idx << ", Vector Size: " << x.size() << endl;
                MPI_Abort(MPI_COMM_WORLD, 3);
            }

            // Calculation: A[i][j] * x[j]
            dot_product += values[k] * x[col_idx];
        }

        // Store result
        y_local[i] = dot_product;
    }

    return y_local;
}