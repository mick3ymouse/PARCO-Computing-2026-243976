#include "Utils.h"
#include <mpi.h>
#include <cstdlib> // For rand(), srand()
#include <ctime>   // For time()
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <vector>

using namespace std;

void generate_distributed_vector(int global_dimension, int rank, int size, vector<double>& full_vec) {
    
    // --- 1. Calculate Local Portion Size (Partitioning) ---
    // We determine how many elements this specific rank needs to generate.
    
    int base_count = global_dimension / size;
    int remainder = global_dimension % size;
    
    // Arrays required by MPI_Allgatherv to know everyone's portion size
    vector<int> all_counts(size); // Maps how many elements EACH process contributes to the global vector. Index 'i' corresponds to the count for Rank 'i'
    vector<int> all_displs(size); // Maps the starting index (offset) in the global vector for EACH process's data.  Index 'i' corresponds to the displacement for Rank 'i'
    
    int current_disp = 0;
    for (int r = 0; r < size; r++) {
        // Distribute the remainder evenly among the first ranks
        int count = base_count + (r < remainder ? 1 : 0);
        
        all_counts[r] = count;
        all_displs[r] = current_disp;
        
        current_disp += count;
    }
    
    // My specific workload
    int my_count = all_counts[rank];

    // --- 2. Simple Random Generation (Your Approach) ---
    vector<double> local_vec(my_count);

    // CRITICAL: We add 'rank' to the seed.
    // Since MPI processes start simultaneously, time(nullptr) would be identical for all.
    // Adding 'rank' ensures each process generates a different sequence.
    srand(static_cast<unsigned int>(time(nullptr)) + rank);

    // Fill with random values in [-1, 1] using standard rand()
    for (int i = 0; i < my_count; ++i) {
        local_vec[i] = (static_cast<double>(rand()) / RAND_MAX) * 2.0 - 1.0;
    }

    // --- 3. Global Assembly (Allgather) ---
    // Resize the output vector to hold the entire global vector
    full_vec.resize(global_dimension);

    // Combine all local parts into the full vector on every process
    MPI_Allgatherv(local_vec.data(), my_count, MPI_DOUBLE,          // Send my part
                   full_vec.data(), all_counts.data(), all_displs.data(), MPI_DOUBLE, // Receive everyone's parts
                   MPI_COMM_WORLD);
}

void log_metrics_csv(const string& output_path, const string& matrix_name, int size, double time_p90, double gflops) {    
    // 1. Open file in Append Mode
    ofstream file(output_path, ios::app);
    
    // 2. Safety Check
    if (!file.is_open()) {
        cerr << "[Error] Could not open CSV file: " << output_path 
             << ". Ensure the 'results' directory exists!" << endl;
        return; // Esce subito se c'è un errore
    }

    // 3. Write Header if file is empty
    file.seekp(0, ios_base::end);
    if (file.tellp() == 0) {
        file << "MatrixName,MPI_Procs,Time_P90_ms,GFLOPS" << endl;
    }

    // 4. Write Data Row
    file << matrix_name << "," << size << "," << time_p90 << "," << gflops << endl;
    
    // 5. Close File
    file.close();
}


void run_spmv_benchmark(const CSRMatrix& mat, const vector<double>& x_vector, 
                        const string& matrix_name, const string& output_path, 
                        int total_nnz, int rank, int size) {
    const int N_RUNS = 10; // Number of benchmark iterations

    // --- A. Setup Output Buffer (Y) ---
    // Dobbiamo sapere quanti elementi produce ogni rank (local_rows) per raccoglierli.
    vector<int> y_counts(size); // Quanti elementi produce ogni rank
    vector<int> y_displs(size); // Dove inizia il blocco di ogni rank nel vettore globale
    
    // 1. Ogni processo dice a tutti quante righe possiede
    MPI_Allgather(&mat.local_rows, 1, MPI_INT, y_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // 2. Calcolo degli offset (displacements)
    y_displs[0] = 0;
    for (int i = 1; i < size; i++) {
        y_displs[i] = y_displs[i-1] + y_counts[i-1];
    }

    // 3. Allocazione vettore risultato globale
    int total_rows_global = y_displs[size-1] + y_counts[size-1];
    vector<double> y_global(total_rows_global);

    // --- B. Warmup (1 Run - Senza Tempo) ---
    vector<double> y_warmup = mat.multiply(x_vector);
    // Facciamo anche la gather per scaldare la rete
    MPI_Allgatherv(y_warmup.data(), mat.local_rows, MPI_DOUBLE, 
                   y_global.data(), y_counts.data(), y_displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD); // Sincronizziamo prima del benchmark

    // --- C. Loop di Misurazione (10 Volte) ---
    vector<double> run_times; // Store times for each run
    if (rank == 0) run_times.reserve(N_RUNS);

    for (int i = 0; i < N_RUNS; i++) {
        MPI_Barrier(MPI_COMM_WORLD); // Start sincronizzato
        
        double t_start = MPI_Wtime();

        // 1. CALCOLO (CPU Phase)
        // Ogni rank produce il suo pezzo locale
        vector<double> y_local = mat.multiply(x_vector);

        // 2. RACCOLTA RISULTATI (Network Phase)
        // Mettiamo insieme i pezzi. Qui paghiamo la latenza di rete.
        // Questo simula la necessità di avere il risultato disponibile per l'iterazione successiva.
        MPI_Allgatherv(y_local.data(), mat.local_rows, MPI_DOUBLE, 
                       y_global.data(), y_counts.data(), y_displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

        double t_end = MPI_Wtime();
        
        // Il tempo include: Calcolo + Attesa Rete + Trasferimento Dati
        double local_time = t_end - t_start;

        // RIDUZIONE: Il tempo del sistema è il tempo del processo più lento
        double global_time = 0.0;
        MPI_Reduce(&local_time, &global_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            run_times.push_back(global_time);
        }
    }

    // --- D. Statistiche e Log (Solo Rank 0) ---
    if (rank == 0) {
        sort(run_times.begin(), run_times.end());

        // 90esimo Percentile
        int p90_idx = (int)ceil(0.9 * N_RUNS) - 1;
        if (p90_idx >= N_RUNS) p90_idx = N_RUNS - 1;
        
        double time_p90 = run_times[p90_idx];

        // GFLOPs Calculation
        double gflops = (2.0 * static_cast<double>(total_nnz)) / (time_p90 * 1.0e9); // Calculate GFLOPS
        time_p90 *= 1000.0; // Convert time_p90 to milliseconds
        
        // --- E. Log to CSV ---
        log_metrics_csv(output_path, matrix_name, size, time_p90, gflops);
    }
}    