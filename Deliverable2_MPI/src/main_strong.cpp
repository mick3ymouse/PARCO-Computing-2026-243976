#include <iostream>
#include <mpi.h>
#include "SparseMatrix.h"
#include "Utils.h"

using namespace std;

int main(int argc, char** argv) {
    // --- 1. MPI Initialization ---
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Who am I?
    MPI_Comm_size(MPI_COMM_WORLD, &size); // How many processes?

    // --- 2. Argument Validation ---
    // Usage: ./main_strong <matrix_folder_path> <output_csv_path>
    if (argc < 3) {
        if (rank == 0) {
            cerr << "Usage: " << argv[0] << " <matrix_folder_path> <output_csv_path>" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    string matrix_folder = argv[1];
    string output_csv_path = argv[2];

    // --- 3. Parallel I/O Phase ---
    CSRMatrix my_matrix;

    // Each rank reads its own portion of the matrix from the files
    load_matrix_parallel(matrix_folder, rank, size, my_matrix);

    // --- 4. Input Vector Generation ---
    // We need global dimensions (cols) to generate the correct vector size.
    int total_rows, total_cols, total_nnz;
    read_global_metadata(matrix_folder + "/meta.bin", total_rows, total_cols, total_nnz);

    vector<double> x_local;
    
    // Generate random vector in parallel (distributed)
    generate_distributed_vector(total_rows, rank, size, x_local);

    // Get matrix name from folder path for logging
    string matrix_name = matrix_folder;
    size_t last_slash = matrix_folder.find_last_of("/\\");
    if (last_slash != string::npos) matrix_name = matrix_folder.substr(last_slash + 1);

    // --- 5. Computation Phase (SpMV) ---
    // The core multiplication: y = A * x
    run_strong_scaling(my_matrix, x_local, matrix_name, output_csv_path, rank, size, (long long) total_nnz);

    // --- 6. Finalization ---
    MPI_Finalize();
    return 0;
}