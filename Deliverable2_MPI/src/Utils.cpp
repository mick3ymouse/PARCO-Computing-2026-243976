#include "Utils.h"
#include <mpi.h>
#include <cstdlib> // For rand(), srand()
#include <ctime>   // For time()
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <set>
#include <cmath>
#include <algorithm>

using namespace std;

int PartitionLookup::find_owner(int global_idx) const {
    // Dato che i range sono ordinati per rank, possiamo scorrere o fare ricerca binaria.
    // Per size < 1000, un ciclo for è velocissimo.
    for(size_t r = 0; r < start_rows.size(); r++) {
        if (global_idx >= start_rows[r] && global_idx < end_rows[r]) {
            return r;
        }
    }
    // Fallback (non dovrebbe succedere se la matrice è ben formata)
    return -1; 
}

/**
 * @brief Builds the partition lookup table for global rows based on local row counts.
 * Each process contributes its local row count and global start row to build the table.
 * @param my_local_rows Number of rows owned by this rank.
 * @param my_global_start Global index of the first row owned by this rank.
 * @param size Total number of MPI processes.
 * @return PartitionLookup The constructed partition lookup table.
 */
PartitionLookup build_partition_table(int my_local_rows, int my_global_start, int size) {
    PartitionLookup table;
    table.start_rows.resize(size); // start_rows[rank] = global index of first row of 'rank'
    table.end_rows.resize(size);   // end_rows[rank]   = global index of last row of 'rank' (exclusive)
    
    vector<int> all_starts(size);  // To gather all starting rows
    vector<int> all_counts(size);  // To gather all local row counts
    
    // Ognuno dice a tutti dove inizia e quante righe ha
    MPI_Allgather(&my_global_start, 1, MPI_INT, all_starts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(&my_local_rows, 1, MPI_INT, all_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
                  
    for(int i=0; i<size; i++) {
        table.start_rows[i] = all_starts[i];
        table.end_rows[i]   = all_starts[i] + all_counts[i];
    }
    return table;
}

GhostCommunicationPlan setup_ghost_exchange(const CSRMatrix& mat, int rank, int size) {
    GhostCommunicationPlan plan;

    // Capire chi possiede i ghost che ci servono
    PartitionLookup lookup = build_partition_table(mat.local_rows, mat.global_start_row, size);
    
    // 1. Organizza richieste: Per ogni ghost, chi lo possiede?
    // La matrice ha già ghost_ids popolato. L'indice k in ghost_ids corrisponde 
    // all'indice locale (local_rows + k).
    vector<vector<int>> requests_by_rank(size);
    vector<vector<int>> target_local_indices_by_rank(size); // Ci serve sapere anche DOVE va messo quel dato (indice locale del ghost)
    
    for (size_t i = 0; i < mat.ghost_ids.size(); ++i) {
        int global_ghost_id = mat.ghost_ids[i];
        int target_local_idx = mat.local_rows + i; // Dove lo metterò nel mio vettore

        int owner = lookup.find_owner(global_ghost_id);
        if (owner != -1 && owner != rank) {
            requests_by_rank[owner].push_back(global_ghost_id);
            target_local_indices_by_rank[owner].push_back(target_local_idx);
        }
    }

    // 2. Costruzione Recv Plan & Mapping
    vector<int> flat_requests; // Lista piatta di ID globali da chiedere
    vector<int> req_counts(size, 0);
    vector<int> req_displs(size, 0);
    
    plan.map_recv_to_local_index.clear();

    for (int r = 0; r < size; r++) {
        req_counts[r] = requests_by_rank[r].size();
        req_displs[r] = (r == 0) ? 0 : req_displs[r-1] + req_counts[r-1];
        
        if (req_counts[r] > 0) {
            plan.neighbors_to_recv_from.push_back(r);
            // Appendo richieste
            flat_requests.insert(flat_requests.end(), requests_by_rank[r].begin(), requests_by_rank[r].end());
            // Appendo mapping (questo dato ricevuto da R andrà nell'indice X)
            plan.map_recv_to_local_index.insert(plan.map_recv_to_local_index.end(), 
                                                target_local_indices_by_rank[r].begin(), 
                                                target_local_indices_by_rank[r].end());
        }
    }
    
    plan.recv_counts = req_counts;
    plan.recv_displs = req_displs;
    plan.total_ghosts_to_recv = flat_requests.size();

    // 3. Handshake (Scambio numero richieste)
    vector<int> send_counts(size);
    MPI_Alltoall(req_counts.data(), 1, MPI_INT, send_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // 4. Ricezione degli ID specifici richiesti dagli altri
    vector<int> send_displs(size, 0);
    for(int i=1; i<size; i++) send_displs[i] = send_displs[i-1] + send_counts[i-1];
    
    int total_to_send = send_displs[size-1] + send_counts[size-1];
    vector<int> indices_requested_global(total_to_send);

    MPI_Alltoallv(flat_requests.data(), req_counts.data(), req_displs.data(), MPI_INT,
                  indices_requested_global.data(), send_counts.data(), send_displs.data(), MPI_INT,
                  MPI_COMM_WORLD);

    // 5. Costruzione Send Plan (Converti Global ID -> Mio Local ID)
    plan.indices_to_send.resize(total_to_send);
    for(int i=0; i<total_to_send; i++) {
        // Mi chiedono un indice che possiedo. La conversione è: Global - MioStart
        plan.indices_to_send[i] = indices_requested_global[i] - mat.global_start_row;
    }

    plan.send_counts = send_counts;
    plan.send_displs = send_displs;
    
    for(int r=0; r<size; r++) if(send_counts[r] > 0) plan.neighbors_to_send_to.push_back(r);

    return plan;
}

void generate_distributed_vector(int global_dimension, int rank, int size, vector<double>& local_vec) {
    // --- 1. Calculate Local Portion Size (Partitioning) ---
    // We determine how many elements this specific rank needs to generate.
    
    int base_count = global_dimension / size;
    int remainder = global_dimension % size;
    
    int my_count = base_count + (rank < remainder ? 1 : 0);

    // --- 2. Simple Random Generation (Your Approach) ---
    local_vec.resize(my_count);

    // CRITICAL: We add 'rank' to the seed.
    // Since MPI processes start simultaneously, time(nullptr) would be identical for all.
    // Adding 'rank' ensures each process generates a different sequence.
    srand(static_cast<unsigned int>(time(nullptr)) + rank);

    // Fill with random values in [-1, 1]
    for (int i = 0; i < my_count; ++i) {
        local_vec[i] = (static_cast<double>(rand()) / RAND_MAX) * 2.0 - 1.0;
    }
}
void log_strong_scaling_csv(const string& output_path, const string& matrix_name, int size, 
                            double time_p90_ms, double time_comm_p90_ms, double time_comp_p90_ms, 
                            double gflops) {
    // 1. Open file in Append Mode
    ofstream file(output_path, ios::app);
    
    // 2. Safety Check
    if (!file.is_open()) {
        cerr << "[Error] Could not open CSV file: " << output_path 
             << ". Ensure the 'results' directory exists!" << endl;
        return; 
    }

    // 3. Write Header if file is empty
    file.seekp(0, ios_base::end);
    if (file.tellp() == 0) {
        file << "MatrixName,MPI_Procs,Time_P90_ms,Time_Comm_ms,Time_Comp_ms,GFLOPS" << endl;
    }

    // 4. Write Data Row
    file << matrix_name << "," << size << "," 
         << time_p90_ms << "," << time_comm_p90_ms << "," << time_comp_p90_ms << "," 
         << gflops << endl;
    file.close();
}

void run_strong_scaling(const CSRMatrix& mat, const vector<double>& x_local, 
                              const string& matrix_name, const string& output_path, 
                              int rank, int size, long long total_nnz) {

    // 1. Setup Piano Comunicazione (mat è già locale/rinumerata)
    GhostCommunicationPlan plan = setup_ghost_exchange(mat, rank, size);

    // 2. Allocazione Vettore Compatto (Locale + Ghost)
    int compact_size = mat.local_rows + mat.ghost_ids.size();
    vector<double> x_compact(compact_size);

    // Copio i miei dati fissi all'inizio
    for(int i=0; i<mat.local_rows; i++) {
        x_compact[i] = x_local[i];
    }

    // Buffer MPI
    vector<double> send_buf(plan.indices_to_send.size());
    vector<double> recv_buf(plan.total_ghosts_to_recv);
    vector<MPI_Request> reqs;
    reqs.reserve(size * 2);

    const int N_RUNS = 10;

    // Vettori per salvare i tempi separati
    vector<double> run_times_total;
    vector<double> run_times_comm;
    vector<double> run_times_comp;

    if (rank == 0) {
        run_times_total.reserve(N_RUNS);
        run_times_comm.reserve(N_RUNS);
        run_times_comp.reserve(N_RUNS);
    }

    MPI_Barrier(MPI_COMM_WORLD); 

    for(int run = -1; run < N_RUNS; run++) {
        MPI_Barrier(MPI_COMM_WORLD);
        
        // --- START TOTAL TIMER ---
        double t_start_total = (run >= 0) ? MPI_Wtime() : 0.0;
        
        // --- START COMM TIMER ---
        double t_start_comm = (run >= 0) ? MPI_Wtime() : 0.0;

        reqs.clear();

        // A. Pack & Send
        for(int dest : plan.neighbors_to_send_to) {
            int count = plan.send_counts[dest];
            int offset = plan.send_displs[dest];
            for(int k=0; k<count; k++) {
                // Leggo dal mio x_compact (parte locale) e metto nel buffer
                send_buf[offset+k] = x_compact[plan.indices_to_send[offset+k]];
            }
            MPI_Request r;
            MPI_Isend(&send_buf[offset], count, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &r);
            reqs.push_back(r);
        }

        // B. Recv
        for(int src : plan.neighbors_to_recv_from) {
            MPI_Request r;
            MPI_Irecv(&recv_buf[plan.recv_displs[src]], plan.recv_counts[src], MPI_DOUBLE, src, 0, MPI_COMM_WORLD, &r);
            reqs.push_back(r);
        }

        if(!reqs.empty()) MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

        // C. Unpack (Con mappa di reindirizzamento)
        // I dati arrivati in recv_buf vanno distribuiti nell'area ghost di x_compact
        for(size_t i=0; i<plan.map_recv_to_local_index.size(); i++) {
            int target_idx = plan.map_recv_to_local_index[i];
            x_compact[target_idx] = recv_buf[i];
        }

        // --- END COMM TIMER / START COMP TIMER ---
        double t_end_comm_start_comp = (run >= 0) ? MPI_Wtime() : 0.0;

        // D. Compute 
        mat.multiply(x_compact);

        // --- END COMP TIMER / END TOTAL TIMER ---
        double t_end_total = (run >= 0) ? MPI_Wtime() : 0.0;

        if (run >= 0) {
            double local_total = t_end_total - t_start_total;
            double local_comm  = t_end_comm_start_comp - t_start_comm;
            double local_comp  = t_end_total - t_end_comm_start_comp;

            double max_total = 0.0, max_comm = 0.0, max_comp = 0.0;

            // Prendiamo il MAX su tutti i processi per vedere il collo di bottiglia
            MPI_Reduce(&local_total, &max_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&local_comm,  &max_comm,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&local_comp,  &max_comp,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

            if(rank == 0) {
                run_times_total.push_back(max_total);
                run_times_comm.push_back(max_comm);
                run_times_comp.push_back(max_comp);
            }
        }
    }

    // --- 4. Statistiche e Log (Solo Rank 0) ---
    if (rank == 0) {
        sort(run_times_total.begin(), run_times_total.end());
        sort(run_times_comm.begin(),  run_times_comm.end());
        sort(run_times_comp.begin(),  run_times_comp.end());
        
        // Calcolo 90esimo percentile
        int p90_idx = (int)ceil(0.9 * N_RUNS) - 1;
        if (p90_idx >= N_RUNS) p90_idx = N_RUNS - 1;
        
        double time_p90 = run_times_total[p90_idx];
        double comm_p90 = run_times_comm[p90_idx];
        double comp_p90 = run_times_comp[p90_idx];

        // Calcolo GFLOPS
        // GFLOPS = (2 * NNZ Totali) / (Tempo in secondi * 10^9)
        double gflops = (2.0 * static_cast<double>(total_nnz)) / (time_p90 * 1.0e9);
        
        // Log dei risultati
        log_strong_scaling_csv(output_path, matrix_name, size, 
                               time_p90 * 1000.0,   // Convert to ms
                               comm_p90 * 1000.0,   // Convert to ms
                               comp_p90 * 1000.0,   // Convert to ms
                               gflops);
    }
}

CSRMatrix generate_synthetic_matrix(int local_rows, int global_cols, int nnz_per_row, int rank) {
    CSRMatrix mat;

    // 1. Setup Dimensioni Locali
    mat.local_rows = local_rows;
    mat.local_nnz = local_rows * nnz_per_row;
    
    // Calcolo start row globale (In Weak Scaling è semplice: fisso per ogni rank)
    mat.global_start_row = rank * local_rows;

    // Resize vettori
    mat.row_ptr.resize(local_rows + 1);
    mat.col_ind.resize(mat.local_nnz);
    mat.values.resize(mat.local_nnz);

    // Seed diverso per ogni rank
    srand(time(nullptr) + rank);

    int current_nnz_idx = 0;
    mat.row_ptr[0] = 0;

    for (int i = 0; i < local_rows; i++) {
        // Usiamo un set per generare indici di colonna unici e ordinati
        set<int> cols;
        while (cols.size() < (size_t)nnz_per_row) {
            // Generiamo una colonna casuale nell'intervallo GLOBALE [0, global_cols)
            int rand_col = rand() % global_cols;
            cols.insert(rand_col);
        }

        // Copiamo nel vettore CSR
        for (int c : cols) {
            mat.col_ind[current_nnz_idx] = c; // Qui salviamo l'indice GLOBALE
            mat.values[current_nnz_idx] = (double)rand() / RAND_MAX; // Valore tra 0 e 1
            current_nnz_idx++;
        }
        
        mat.row_ptr[i+1] = current_nnz_idx;
    }

    // Converti indici globali in locali e popola ghost_ids
    convert_matrix_to_local(mat);

    return mat;
}

void log_weak_scaling_csv(const string& output_path, int size, int global_rows, int global_nnz, 
                          double time_p90_ms, double time_comm_p90_ms, double time_comp_p90_ms, 
                          double gflops) {
    // 1. Open file in Append Mode
    ofstream file(output_path, ios::app);
    
    // 2. Safety Check
    if (!file.is_open()) {
        cerr << "[Error] Could not open CSV file: " << output_path 
             << ". Ensure the 'results' directory exists!" << endl;
        return; 
    }

    // 3. Write Header if file is empty
    file.seekp(0, ios_base::end);
    if (file.tellp() == 0) {
        file << "MPI_Procs,Global_Rows,Global_NNZ,Time_P90_ms,Time_Comm_ms,Time_Comp_ms,GFLOPS" << endl;
    }

    // 4. Write Data Row
    file << size << "," << global_rows << "," << global_nnz << "," 
         << time_p90_ms << "," << time_comm_p90_ms << "," << time_comp_p90_ms << "," 
         << gflops << endl;
    file.close();
}

void run_weak_scaling(const CSRMatrix& mat, const vector<double>& x_local, 
                      const string& output_path, int rows_per_proc, 
                      long long global_nnz, int rank, int size) {
    
    // 1. Setup Piano Comunicazione (mat è già locale/rinumerata)
    GhostCommunicationPlan plan = setup_ghost_exchange(mat, rank, size);

    // 2. Allocazione Vettore Compatto (Locale + Ghost)
    int compact_size = mat.local_rows + mat.ghost_ids.size();
    vector<double> x_compact(compact_size);

    // Copio i miei dati fissi all'inizio
    for(int i=0; i<mat.local_rows; i++) {
        x_compact[i] = x_local[i];
    }

    // Buffer MPI
    vector<double> send_buf(plan.indices_to_send.size());
    vector<double> recv_buf(plan.total_ghosts_to_recv);
    vector<MPI_Request> reqs;
    reqs.reserve(size * 2);

    const int N_RUNS = 10;

    // Vettori per tempi separati
    vector<double> run_times_total;
    vector<double> run_times_comm;
    vector<double> run_times_comp;

    if (rank == 0) {
        run_times_total.reserve(N_RUNS);
        run_times_comm.reserve(N_RUNS);
        run_times_comp.reserve(N_RUNS);
    }

    MPI_Barrier(MPI_COMM_WORLD); // Sincronizzazione pre-benchmark

    for (int run = -1; run < N_RUNS; run++) {
        MPI_Barrier(MPI_COMM_WORLD); 
        
        // --- START TOTAL TIMER ---
        double t_start_total = (run >= 0) ? MPI_Wtime() : 0.0;
        
        // --- START COMM TIMER ---
        double t_start_comm = (run >= 0) ? MPI_Wtime() : 0.0;

        reqs.clear();

        // --- A. GHOST EXCHANGE (Halo) ---
        
        // 1. Pack & Send
        for(int dest : plan.neighbors_to_send_to) {
            int count = plan.send_counts[dest];
            int offset = plan.send_displs[dest];
            for(int k=0; k<count; k++) {
                send_buf[offset+k] = x_compact[plan.indices_to_send[offset+k]];
            }
            MPI_Request r;
            MPI_Isend(&send_buf[offset], count, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &r);
            reqs.push_back(r);
        }

        // 2. Recv
        for(int src : plan.neighbors_to_recv_from) {
            MPI_Request r;
            MPI_Irecv(&recv_buf[plan.recv_displs[src]], plan.recv_counts[src], MPI_DOUBLE, src, 0, MPI_COMM_WORLD, &r);
            reqs.push_back(r);
        }

        // 3. Wait
        if(!reqs.empty()) MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

        // 4. Unpack
        for(size_t i=0; i<plan.map_recv_to_local_index.size(); i++) {
            int target_idx = plan.map_recv_to_local_index[i];
            x_compact[target_idx] = recv_buf[i];
        }

        // --- END COMM TIMER / START COMP TIMER ---
        double t_end_comm_start_comp = (run >= 0) ? MPI_Wtime() : 0.0;

        // --- B. COMPUTE ---
        mat.multiply(x_compact);
        
        // --- END COMP TIMER / END TOTAL TIMER ---
        double t_end_total = (run >= 0) ? MPI_Wtime() : 0.0;

        if (run >= 0) {
            double local_total = t_end_total - t_start_total;
            double local_comm  = t_end_comm_start_comp - t_start_comm;
            double local_comp  = t_end_total - t_end_comm_start_comp;

            double max_total = 0.0, max_comm = 0.0, max_comp = 0.0;

            // Prendiamo il MAX su tutti i processi per vedere il collo di bottiglia
            MPI_Reduce(&local_total, &max_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&local_comm,  &max_comm,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&local_comp,  &max_comp,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

            if(rank == 0) {
                run_times_total.push_back(max_total);
                run_times_comm.push_back(max_comm);
                run_times_comp.push_back(max_comp);
            }
        }
    }

    // --- 4. Statistiche e Log (Solo Rank 0) ---
    if (rank == 0) {
        sort(run_times_total.begin(), run_times_total.end());
        sort(run_times_comm.begin(),  run_times_comm.end());
        sort(run_times_comp.begin(),  run_times_comp.end());
        
        // Calcolo 90esimo percentile
        int p90_idx = (int)ceil(0.9 * N_RUNS) - 1;
        if (p90_idx >= N_RUNS) p90_idx = N_RUNS - 1;
        
        double time_p90 = run_times_total[p90_idx];
        double comm_p90 = run_times_comm[p90_idx];
        double comp_p90 = run_times_comp[p90_idx];

        // Calcolo GFLOPS
        // GFLOPS = (2 * NNZ Totali) / (Tempo in secondi * 10^9)
        double gflops = (2.0 * static_cast<double>(total_nnz)) / (time_p90 * 1.0e9);
        
        // Log dei risultati
        log_weak_scaling_csv(output_path, size, global_rows, global_nnz, 
                             time_p90 * 1000.0,    // Convert to ms
                             comm_p90 * 1000.0,    // Convert to ms
                             comp_p90 * 1000.0,    // Convert to ms
                             gflops);
    }
}