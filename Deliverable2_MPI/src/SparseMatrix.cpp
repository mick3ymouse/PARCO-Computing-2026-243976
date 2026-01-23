#include "SparseMatrix.h"
#include <iostream>
#include <vector>
#include <map>

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

    // --- STEP 6: Convert Global Col Indices to Local Indices ---
    convert_matrix_to_local(mat);
}

void convert_matrix_to_local(CSRMatrix& mat) {
    int my_start = mat.global_start_row;
    int my_end   = mat.global_start_row + mat.local_rows;
    
    // Mappa: ID Globale -> ID Locale Nuovo
    map<int, int> global_to_local;
    vector<int> ghosts_found;
    int next_ghost_local_id = mat.local_rows; // I ghost iniziano dopo le mie righe

    // Scansioniamo tutta la matrice
    for (int& col : mat.col_ind) {
        int global_id = col;

        // Se è una colonna MIA
        if (global_id >= my_start && global_id < my_end) {
            col = global_id - my_start; // Diventa 0, 1, ... local_rows-1
        } 
        // Se è un GHOST
        else {
            // L'abbiamo già visto?
            if (global_to_local.find(global_id) == global_to_local.end()) {
                // No, è nuovo. Creiamo una nuova entry.
                global_to_local[global_id] = next_ghost_local_id;
                ghosts_found.push_back(global_id);
                next_ghost_local_id++;
            }
            // Sostituiamo l'indice globale con quello locale
            col = global_to_local[global_id];
        }
    }

    // Salviamo la lista ordinata dei ghost (ci serve per MPI)
    mat.ghost_ids = ghosts_found;
}


// --- SpMV Implementation ---

vector<double> CSRMatrix::multiply(const vector<double>& x_local) const {
    // x_local contiene: [Miei Dati (0..local_rows-1) | Ghost (local_rows...)]
    vector<double> y(local_rows);
    for (int i = 0; i < local_rows; ++i) {
        double sum = 0.0;
        for (int k = row_ptr[i]; k < row_ptr[i+1]; ++k) {
            // Accesso diretto! Nessun controllo, nessuna mappa.
            sum += values[k] * x_local[col_ind[k]];
        }
        y[i] = sum;
    }
    return y;
}