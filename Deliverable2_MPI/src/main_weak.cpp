#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include "SparseMatrix.h"
#include "Utils.h"

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 1. Check Argomenti
    // Usage: ./main_weak <rows_per_proc> <nnz_per_row>
    if (argc < 3) {
        if (rank == 0) {
            cerr << "Usage: " << argv[0] << " <rows_per_proc> <nnz_per_row>" << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rows_per_proc = atoi(argv[1]); // Dimensione BASE (Costante per processo)
    int nnz_per_row   = atoi(argv[2]); // Densità

    // 2. Calcolo Dimensioni Globali
    // Nel weak scaling: Dimensione Totale = Dimensione Locale * N_Processi
    int global_rows = rows_per_proc * size;
    int global_cols = global_rows; // Matrice Quadrata
    long long global_nnz = (long long)global_rows * nnz_per_row;

    if (rank == 0) {
        cout << "=== STARTING WEAK SCALING TEST ===" << endl;
        cout << "Processes:      " << size << endl;
        cout << "Rows per Proc:  " << rows_per_proc << endl;
        cout << "Global Size:    " << global_rows << " x " << global_cols << endl;
        cout << "Total NNZ: " << global_nnz << endl << endl;
    }

    // 3. Generazione Matrice Sintetica (Locale)
    CSRMatrix mat = generate_synthetic_matrix(rows_per_proc, global_cols, nnz_per_row, rank);

    // 4. Generazione Vettore x (Distribuito)
    vector<double> x_local;
    generate_distributed_vector(global_cols, rank, size, x_local);

    // 5. Esecuzione Benchmark
    // Passiamo tutti i dati alla funzione dedicata in Utils
    string output_file = "results/weak_scaling.csv";
    
    run_weak_scaling(mat, x_local, output_file, rows_per_proc, global_nnz, rank, size);

    MPI_Finalize();
    return 0;
}