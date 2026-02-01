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
#include <iomanip>
#include <algorithm>

using namespace std;

void PartitionLookup::build_partition_table(int my_local_rows, int my_global_start, int size) {
    start_rows.resize(size); // start_rows[rank] = global index of first row of 'rank'
    end_rows.resize(size);   // end_rows[rank]   = global index of last row of 'rank' (exclusive)
    
    vector<int> all_starts(size);  // To gather all starting rows
    vector<int> all_counts(size);  // To gather all local row counts
    
    // Gather data from all ranks
    MPI_Allgather(&my_global_start, 1, MPI_INT, all_starts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(&my_local_rows, 1, MPI_INT, all_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
                  
    for(int i=0; i<size; i++) {
        start_rows[i] = all_starts[i];
        end_rows[i]   = all_starts[i] + all_counts[i];
    }
}

int PartitionLookup::find_owner(int global_col_idx) const {
    // Scan all ranks to see which global row range contains 'global_col_idx'
    for(size_t r = 0; r < start_rows.size(); r++) {
        // Rank 'r' owns Row K  <==>  Rank 'r' owns Vector Element x[K]. (Thanks to 1D Row Partitioning)
        if (global_col_idx >= start_rows[r] && global_col_idx < end_rows[r]) {
            return r;
        }
    }
    // Fallback for security
    return -1; 
}

void LoadBalanceStats::calculate(const CSRMatrix& mat, const GhostCommunicationPlan& plan, int rank, int size) {
    // 1. NNZ Statistics
    long long local_nnz = mat.local_nnz;
    long long sum_nnz = 0;

    MPI_Reduce(&local_nnz, &nnz_min, 1, MPI_LONG_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_nnz, &nnz_max, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_nnz, &sum_nnz, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    
    nnz_avg = (double)sum_nnz / size;

    // 2. Communication Volume Statistics
    // Volume = Elements Sent + Elements Received
    // This represents the total "network traffic" load for this rank.
    long long local_comm_vol = 0;
    
    // Outgoing volume (sum of send counts)
    for(int c : plan.send_counts) local_comm_vol += c;
    
    // Incoming volume (total ghosts needed)
    local_comm_vol += plan.total_ghosts_to_recv;

    long long sum_comm = 0;

    MPI_Reduce(&local_comm_vol, &comm_min, 1, MPI_LONG_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_comm_vol, &comm_max, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_comm_vol, &sum_comm, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    comm_avg = (double)sum_comm / size;
}

void log_load_balance_csv(const string& output_path, const string& matrix_name, int size, 
                          const LoadBalanceStats& stats, bool is_weak_scaling) {
    
    // 1. Check if file exists to write header
    bool write_header = false;
    ifstream check_file(output_path);

    // If file does not exist or is empty, we need to write the header
    if (check_file.peek() == ifstream::traits_type::eof() || !check_file.good()) {
        write_header = true;
    }
    check_file.close();

    // 2. Open file in Append Mode
    ofstream file(output_path, ios::app);
    if (!file.is_open()) return;

    // 3. Write Header if needed
    if (write_header) {
        if (!is_weak_scaling) file << "MatrixName,";
        file << "MPI_Procs,NNZ_min,NNZ_max,NNZ_avg,CommVol_min,CommVol_max,CommVol_avg" << endl;
    }

    // 4. Write Data Row
    if (!is_weak_scaling) file << matrix_name << ",";
    
    file << size << "," 
         << stats.nnz_min << "," << stats.nnz_max << "," << fixed << setprecision(2) << stats.nnz_avg << ","
         << stats.comm_min << "," << stats.comm_max << "," << fixed << setprecision(2) << stats.comm_avg 
         << endl;
    file.close();
}

GhostCommunicationPlan setup_ghost_exchange(const CSRMatrix& mat, int rank, int size) {
    GhostCommunicationPlan plan;

    // 1. Build Partition Table to resolve global ID ownership
    PartitionLookup lookup;
    lookup.build_partition_table(mat.local_rows, mat.global_start_row, size);
    
    // 2. Organize Requests: Group required ghosts by their owner rank
    // mat.ghost_ids is already populated; index 'i' maps to local index (local_rows + i).
    vector<vector<int>> requests_by_rank(size);
    vector<vector<int>> target_local_indices_by_rank(size); // Local storage index
    
    for (size_t i = 0; i < mat.ghost_ids.size(); ++i) {
        int global_ghost_id = mat.ghost_ids[i];
        int target_local_idx = mat.local_rows + i; // Dove lo metterò nel mio vettore

        int owner = lookup.find_owner(global_ghost_id);
        if (owner != -1 && owner != rank) {
            requests_by_rank[owner].push_back(global_ghost_id);
            target_local_indices_by_rank[owner].push_back(target_local_idx);
        }
    }

    // 3. Build Receive Plan & Unpack Map
    vector<int> flat_requests; // Flattened list of global IDs to request
    vector<int> req_counts(size, 0);
    vector<int> req_displs(size, 0);
    
    plan.map_recv_to_local_index.clear();

    for (int r = 0; r < size; r++) {
        req_counts[r] = requests_by_rank[r].size();
        req_displs[r] = (r == 0) ? 0 : req_displs[r-1] + req_counts[r-1];
        
        if (req_counts[r] > 0) {
            plan.neighbors_to_recv_from.push_back(r);
            // Append requests
            flat_requests.insert(flat_requests.end(), requests_by_rank[r].begin(), requests_by_rank[r].end());
            // Append mapping (Network Buffer Index -> Local Ghost Index)
            plan.map_recv_to_local_index.insert(plan.map_recv_to_local_index.end(), 
                                                target_local_indices_by_rank[r].begin(), 
                                                target_local_indices_by_rank[r].end());
        }
    }
    
    plan.recv_counts = req_counts;
    plan.recv_displs = req_displs;
    plan.total_ghosts_to_recv = flat_requests.size();

    // 4. Handshake: Exchange request counts
    vector<int> send_counts(size);
    MPI_Alltoall(req_counts.data(), 1, MPI_INT, send_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // 5. Receive specific Global IDs requested by others
    vector<int> send_displs(size, 0);
    for(int i=1; i<size; i++) send_displs[i] = send_displs[i-1] + send_counts[i-1];
    
    int total_to_send = send_displs[size-1] + send_counts[size-1];
    vector<int> indices_requested_global(total_to_send);

    MPI_Alltoallv(flat_requests.data(), req_counts.data(), req_displs.data(), MPI_INT,
                  indices_requested_global.data(), send_counts.data(), send_displs.data(), MPI_INT,
                  MPI_COMM_WORLD);

    // 6. Build Send Plan (Convert Global ID -> Local Internal ID)
    plan.indices_to_send.resize(total_to_send);
    for(int i=0; i<total_to_send; i++) {
        // Requested Global Index -> My Local Internal Index (Offset by start_row)
        plan.indices_to_send[i] = indices_requested_global[i] - mat.global_start_row;
    }

    plan.send_counts = send_counts;
    plan.send_displs = send_displs;
    
    for(int r=0; r<size; r++) if(send_counts[r] > 0) plan.neighbors_to_send_to.push_back(r);

    return plan;
}

void generate_distributed_vector(int global_dimension, int rank, int size, vector<double>& local_vec) {
    // 1. Calculate Local Partition Size
    int base_count = global_dimension / size;
    int remainder = global_dimension % size;
    
    int my_count = base_count + (rank < remainder ? 1 : 0);

    // 2. Random Generation
    local_vec.resize(my_count);

    // Seed depends on rank to ensure different sequences across processes
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

    // 1. Setup Communication Plan
    GhostCommunicationPlan plan = setup_ghost_exchange(mat, rank, size);

    // 2. Allocate Compact Vector (Local + Ghost Area)
    int compact_size = mat.local_rows + mat.ghost_ids.size();
    vector<double> x_compact(compact_size);

    // Copy owned local data to the start of the compact vector
    for(int i=0; i<mat.local_rows; i++) {
        x_compact[i] = x_local[i];
    }

    // Allocate Buffers for MPI Communication
    vector<double> send_buf(plan.indices_to_send.size());
    vector<double> recv_buf(plan.total_ghosts_to_recv);
    vector<MPI_Request> reqs;
    reqs.reserve(size * 2);

    const int N_RUNS = 100;

    // Vectors to store separate timings
    vector<double> run_times_total;
    vector<double> run_times_comm;
    vector<double> run_times_comp;

    if (rank == 0) {
        run_times_total.reserve(N_RUNS);
        run_times_comm.reserve(N_RUNS);
        run_times_comp.reserve(N_RUNS);
    }

    MPI_Barrier(MPI_COMM_WORLD); 

    // 3. Benchmark Loop
    for(int run = -1; run < N_RUNS; run++) {
        MPI_Barrier(MPI_COMM_WORLD);
        
        // --- START TOTAL TIMER ---
        double t_start_total = (run >= 0) ? MPI_Wtime() : 0.0;
        
        // --- START COMM TIMER ---
        double t_start_comm = (run >= 0) ? MPI_Wtime() : 0.0;

        reqs.clear();

        // A. Pack & Send (Halo Exchange)
        for(int dest : plan.neighbors_to_send_to) {
            int count = plan.send_counts[dest];
            int offset = plan.send_displs[dest];
            for(int k=0; k<count; k++) {
                // Gather from Local Vector -> Network Buffer
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

        // C. Wait
        if(!reqs.empty()) MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

        // D. Unpack using the mapping
        // Data received in recv_buf is distributed into the ghost area of x_compact
        for(size_t i=0; i<plan.map_recv_to_local_index.size(); i++) {
            int target_idx = plan.map_recv_to_local_index[i];
            x_compact[target_idx] = recv_buf[i];
        }

        // --- END COMM TIMER / START COMP TIMER ---
        double t_end_comm_start_comp = (run >= 0) ? MPI_Wtime() : 0.0;

        // E. Compute 
        mat.multiply(x_compact);

        // --- END COMP TIMER / END TOTAL TIMER ---
        double t_end_total = (run >= 0) ? MPI_Wtime() : 0.0;

        // F. Collect Timings
        if (run >= 0) {
            double local_total = t_end_total - t_start_total;
            double local_comm  = t_end_comm_start_comp - t_start_comm;
            double local_comp  = t_end_total - t_end_comm_start_comp;

            double max_total = 0.0, max_comm = 0.0, max_comp = 0.0;

            // We take the MAX across all processes to identify the bottleneck
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

    // 4. Stats and Logging (Rank 0 only)
    if (rank == 0) {
        sort(run_times_total.begin(), run_times_total.end());
        sort(run_times_comm.begin(),  run_times_comm.end());
        sort(run_times_comp.begin(),  run_times_comp.end());
        
        // Calculate P90 (90th percentile)
        int p90_idx = (int)ceil(0.9 * N_RUNS) - 1;
        if (p90_idx >= N_RUNS) p90_idx = N_RUNS - 1;
        
        double time_p90 = run_times_total[p90_idx];
        double comm_p90 = run_times_comm[p90_idx];
        double comp_p90 = run_times_comp[p90_idx];

        // Calculate GFLOPS
        // GFLOPS = (2 * Total NNZ) / (Time in seconds * 10^9)
        double gflops = (2.0 * static_cast<double>(total_nnz)) / (time_p90 * 1.0e9);
        
        // Log results
        log_strong_scaling_csv(output_path, matrix_name, size, 
                               time_p90 * 1000.0,   // Convert to ms
                               comm_p90 * 1000.0,   // Convert to ms
                               comp_p90 * 1000.0,   // Convert to ms
                               gflops);
    }

    LoadBalanceStats lb_stats;
    lb_stats.calculate(mat, plan, rank, size);
    
    if (rank == 0) {
        string lb_output_path = "results/load_balance_strong.csv"; 
        log_load_balance_csv(lb_output_path, matrix_name, size, lb_stats, false);
    }
}

CSRMatrix generate_synthetic_matrix(int local_rows, int total_cols, int nnz_per_row, int rank) {
    CSRMatrix mat;

    // 1. Setup Local Dimensions
    mat.local_rows = local_rows;
    mat.local_nnz = local_rows * nnz_per_row;
    mat.global_start_row = rank * local_rows; // For weak scaling, global start row index is simply a fixed offset 

    mat.row_ptr.resize(local_rows + 1);
    mat.col_ind.resize(mat.local_nnz);
    mat.values.resize(mat.local_nnz);

    // Seed depends on rank to ensure different sequences across processes
    srand(time(nullptr) + rank);

    int current_nnz_idx = 0;
    mat.row_ptr[0] = 0;

    for (int i = 0; i < local_rows; i++) {
        // Use 'set' to ensure unique, sorted column indices
        set<int> cols;
        while (cols.size() < (size_t)nnz_per_row) {
            // Generate a random column in the GLOBAL range [0, total_cols)
            int rand_col = rand() % total_cols;
            cols.insert(rand_col);
        }

        // Fill CSR vectors
        for (int c : cols) {
            mat.col_ind[current_nnz_idx] = c; // Here we save the GLOBAL index initally
            mat.values[current_nnz_idx] = (double)rand() / RAND_MAX; // Value between 0 and 1
            current_nnz_idx++;
        }
        
        mat.row_ptr[i+1] = current_nnz_idx;
    }

    // 2. Convert Global Indices to Local/Ghost and populate ghost_ids
    convert_matrix_to_local(mat);

    return mat;
}

void log_weak_scaling_csv(const string& output_path, int size, int total_rows, int total_nnz, 
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
        file << "MPI_Procs,Total_Rows,Total_NNZ,Time_P90_ms,Time_Comm_ms,Time_Comp_ms,GFLOPS" << endl;
    }

    // 4. Write Data Row
    file << size << "," << total_rows << "," << total_nnz << "," 
         << time_p90_ms << "," << time_comm_p90_ms << "," << time_comp_p90_ms << "," 
         << gflops << endl;
    file.close();
}

void run_weak_scaling(const CSRMatrix& mat, const vector<double>& x_local, 
                      const string& output_path, int rows_per_proc, 
                      long long total_nnz, int rank, int size) {
    
    // 1. Setup Communication Plan
    GhostCommunicationPlan plan = setup_ghost_exchange(mat, rank, size);

    // 2. Allocate Compact Vector (Local + Ghost)
    int compact_size = mat.local_rows + mat.ghost_ids.size();
    vector<double> x_compact(compact_size);

    // Copy owned local data to the start of the compact vector
    for(int i=0; i<mat.local_rows; i++) {
        x_compact[i] = x_local[i];
    }

    // Allocate buffers for MPI Communication
    vector<double> send_buf(plan.indices_to_send.size());
    vector<double> recv_buf(plan.total_ghosts_to_recv);
    vector<MPI_Request> reqs;
    reqs.reserve(size * 2);

    const int N_RUNS = 100;

    // Vectors to store separate timings
    vector<double> run_times_total;
    vector<double> run_times_comm;
    vector<double> run_times_comp;

    if (rank == 0) {
        run_times_total.reserve(N_RUNS);
        run_times_comm.reserve(N_RUNS);
        run_times_comp.reserve(N_RUNS);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // 3. Benchmark Loop
    for (int run = -1; run < N_RUNS; run++) {
        MPI_Barrier(MPI_COMM_WORLD); 
        
        // --- START TOTAL TIMER ---
        double t_start_total = (run >= 0) ? MPI_Wtime() : 0.0;
        
        // --- START COMM TIMER ---
        double t_start_comm = (run >= 0) ? MPI_Wtime() : 0.0;

        reqs.clear();

        // A. Pack & Send (Halo Exchange)
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

        // B. Recv
        for(int src : plan.neighbors_to_recv_from) {
            MPI_Request r;
            MPI_Irecv(&recv_buf[plan.recv_displs[src]], plan.recv_counts[src], MPI_DOUBLE, src, 0, MPI_COMM_WORLD, &r);
            reqs.push_back(r);
        }

        // C. Wait
        if(!reqs.empty()) MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

        // D. Unpack
        for(size_t i=0; i<plan.map_recv_to_local_index.size(); i++) {
            int target_idx = plan.map_recv_to_local_index[i];
            x_compact[target_idx] = recv_buf[i];
        }

        // --- END COMM TIMER / START COMP TIMER ---
        double t_end_comm_start_comp = (run >= 0) ? MPI_Wtime() : 0.0;

        // E. Compute
        mat.multiply(x_compact);
        
        // --- END COMP TIMER / END TOTAL TIMER ---
        double t_end_total = (run >= 0) ? MPI_Wtime() : 0.0;

        // F. Collect Timings
        if (run >= 0) {
            double local_total = t_end_total - t_start_total;
            double local_comm  = t_end_comm_start_comp - t_start_comm;
            double local_comp  = t_end_total - t_end_comm_start_comp;

            double max_total = 0.0, max_comm = 0.0, max_comp = 0.0;

            // We take the MAX across all processes to identify the bottleneck
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

    // 4. Stats and Logging (Rank 0 only)
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
        
        int total_rows = rows_per_proc * size;

        // Log dei risultati
        log_weak_scaling_csv(output_path, size, total_rows, total_nnz, 
                             time_p90 * 1000.0,    // Convert to ms
                             comm_p90 * 1000.0,    // Convert to ms
                             comp_p90 * 1000.0,    // Convert to ms
                             gflops);
    }

    LoadBalanceStats lb_stats;
    lb_stats.calculate(mat, plan, rank, size);
    
    if (rank == 0) {
        string lb_output_path = "results/load_balance_weak.csv";
        log_load_balance_csv(lb_output_path, "", size, lb_stats, true);
    }
}