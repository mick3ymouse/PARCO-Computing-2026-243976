#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
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
    // Usage: ./main_weak <rows_per_proc> <nnz_per_row>
    if (argc < 4) {
        if (rank == 0) {
            cerr << "Usage: " << argv[0] << " <rows_per_proc> <nnz_per_row>" << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rows_per_proc = atoi(argv[1]);  // BASE dimension (Constant per process)
    int nnz_per_row   = atoi(argv[2]);  // Fixed Density
    string output_csv_path = argv[3];   // Output CSV file path

    // --- 3. Calculate Global Dimensions ---
    // In weak scaling: Total Dimension = Local Dimension * Number of Processes
    int total_rows = rows_per_proc * size;
    int total_cols = total_rows;
    long long total_nnz = static_cast<long long> (total_rows) * nnz_per_row;

    if (rank == 0) {
        cout << "=== STARTING WEAK SCALING TEST ===" << endl;
        cout << "Processes:      " << size << endl;
        cout << "Rows per Proc:  " << rows_per_proc << endl;
        cout << "Global Size:    " << total_rows << " x " << total_cols << endl;
        cout << "Total NNZ: " << total_nnz << endl << endl;
    }

    // --- 4. Matrix Generation (Synthetic) ---
    CSRMatrix mat = generate_synthetic_matrix(rows_per_proc, total_cols, nnz_per_row, rank);

    // --- 5. Input Vector Generation (Distributed) ---
    vector<double> x_local;
    generate_distributed_vector(total_cols, rank, size, x_local);

    // --- 6. Benchmark Execution ---    
    run_weak_scaling(mat, x_local, output_csv_path, rows_per_proc, total_nnz, rank, size);

    MPI_Finalize();
    return 0;
}