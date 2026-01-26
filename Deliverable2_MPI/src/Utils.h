#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>
#include <mpi.h>
#include "SparseMatrix.h"

using namespace std;

/** 
* @brief Structure to hold the communication plan for ghost exchanges.
* This includes what to send, what to receive, and the mapping of received data.
* @param indices_to_send_by_rank For each rank, the list of local indices to send.
* @param send_counts Number of elements to send to each rank.
* @param send_displs Displacements in the send buffer for each rank.
* @param neighbors_to_send_to List of ranks to which we will send data.
* @param recv_counts Number of elements to receive from each rank.
* @param recv_displs Displacements in the receive buffer for each rank.
* @param map_recv_to_global_index Mapping from received buffer indices to global indices.
* @param neighbors_to_recv_from List of ranks from which we will receive data.
* @param total_ghosts_to_recv Total number of ghost elements to receive. 
* @param map_recv_to_local_index Mapping from received buffer indices to local indices in the augmented vector. 
*/
struct GhostCommunicationPlan {
    // --- Send Side (Outgoing Data) ---

    // List of local indices to copy into the network send buffer.
    // These specific elements represent the data required by other ranks.
    vector<int> indices_to_send;   

    // Number of elements to send to each rank.
    // Required argument for MPI_Alltoallv to define packet sizes.
    vector<int> send_counts;       

    // Starting offset in 'indices_to_send' for each destination rank.
    // Marks where the data chunk for a specific neighbor begins.
    vector<int> send_displs;           

    // List of unique rank IDs that requested data from me.
    vector<int> neighbors_to_send_to;  


    // --- Receive Side (Incoming Ghosts) ---

    // MAPPING: Network Buffer Index -> Local Ghost Index.
    // The i-th element received from the network is stored at local_vector[map[i]].
    vector<int> map_recv_to_local_index; 

    // MAPPING: Network Buffer Index -> Global Column Index.
    // Maps received data back to the original matrix column IDs.
    // Primarily used for debugging and topology verification, not math.
    vector<int> map_recv_to_global_index; 

    // Number of ghost elements expected from each rank.
    // Tells MPI how many items to accept from each neighbor.
    vector<int> recv_counts;     
    
    // Starting offset in the receive buffer for data coming from each rank.
    // Marks where the incoming chunk from a specific neighbor is placed.
    vector<int> recv_displs;         
    
    // List of unique rank IDs that send ghost data to me.
    // These ranks own the columns that correspond to my ghost entries.
    vector<int> neighbors_to_recv_from;   

    // Total count of elements to receive (sum of recv_counts).
    // Defines the required size for the receive buffer and ghost storage.
    int total_ghosts_to_recv;              
};

/**
 * @brief Structure to manage the global row partitioning across MPI ranks.
 * Used to determine which rank owns a specific global index (critical for finding ghost owners).
 */
struct PartitionLookup {
    // Arrays containing the GLOBAL index boundaries for each rank.
    // Range for Rank 'r' is: [ start_rows[r], end_rows[r] )
    vector<int> start_rows; // start_rows[rank] = GLOBAL index of the first row owned by 'rank'
    vector<int> end_rows;   // end_rows[rank]   = GLOBAL index of the last row (exclusive) owned by 'rank'
    
    /**
     * @brief Populates the partition table by gathering data from all ranks.
     * Each process contributes its local row count and global start index.
     * @param my_local_rows Number of rows owned by this rank.
     * @param my_global_start Global index of the first row owned by this rank.
     * @param size Total number of MPI processes.
     */
    void build_partition_table(int my_local_rows, int my_global_start, int size);
    
    /**
     * @brief the rank that owns a specific global index.
     * In SpMV (y = Ax), vector 'x' is distributed exactly like matrix rows.
     * Thus, checking who owns Matrix Row 'j' (global) reveals who owns Vector Element x[j].
     * @param global_col_idx The global column index to search for.
     * @return int The rank ID of the owner.
     */
    int find_owner(int global_col_idx) const;
};

/**
 * @brief Sets up the ghost exchange communication plan for a given CSR matrix.
 * Analyzes the matrix to determine which off-rank data is needed for SpMV.
 * @param mat The local CSR matrix partition.
 * @param rank MPI Rank.
 * @param size MPI Size.
 * @return GhostCommunicationPlan The communication plan for ghost exchanges.
 */
GhostCommunicationPlan setup_ghost_exchange(const CSRMatrix& mat, int rank, int size);

/**
 * @brief Generates a random vector distributed across processes using rand().
 * * Strategy:
 * 1. Calculates how many elements each process must generate (Load Balancing).
 * 2. Uses rand() seeded with time + rank to generate unique sequences.
 * 3. Assembles the full vector using MPI_Allgatherv.
 * @param global_dimension Total size of the vector.
 * @param rank MPI Rank.
 * @param size MPI Size.
 * @param local_vec Output local portion of the vector.
 */
void generate_distributed_vector(int global_dimension, int rank, int size, vector<double>& local_vec);


// Functions for strong scaling:

/**
 * @brief Logs strong scaling benchmark results to a CSV file.
 * @param output_path Path to the CSV output file.
 * @param matrix_name Name of the matrix (for logging).
 * @param size Number of MPI processes.
 * @param time_p90_ms The 90th percentile of execution time.
 * @param gflops The calculated GFLOPS based on P90 time.
 */
void log_strong_scaling_csv(const string& output_path, const string& matrix_name, int size, 
                            double time_p90_ms, double time_comm_p90_ms, double time_comp_p90_ms,
                            double gflops);

/**
 * @brief Run the Strong Scaling benchmark.
 * 1. Uses the provided x_local for computations + ghost exchanges.
 * 2. Runs a Warmup.
 * 3. Executes the loop 10 times.
 * 4. Collects times and computes the 90th percentile.
 * 5. Calls log_strong_scaling_csv() to report results.
 * @param mat Local CSR Matrix partition.
 * @param x_local The local portion of the input vector.
 * @param matrix_name Name of the matrix (for logging).
 * @param output_path Path to the CSV output file.
 * @param total_cols Global width of the matrix (size of vector x).
 * @param total_nnz Total non-zeros (for GFLOPS calculation).
 * @param rank MPI Rank.
 * @param size MPI Size.
 */
void run_strong_scaling(const CSRMatrix& mat, const vector<double>& x_local, 
                              const string& matrix_name, 
                              const string& output_path, int rank, int size, 
                              long long total_nnz);


// Functions for weak scaling:

/**
 * @brief Generates a synthetic CSR matrix for weak scaling tests.
 * Each process generates a local portion of the matrix with a fixed number of rows
 * and non-zeros per row.
 * @param local_rows Number of rows for this process.
 * @param total_cols Total number of columns in the global matrix.
 * @param nnz_per_row Number of non-zero elements per row.
 * @param rank MPI Rank.
 * @return CSRMatrix The generated local CSR matrix partition.
 */
CSRMatrix generate_synthetic_matrix(int local_rows, int total_cols, int nnz_per_row, int rank);

/**
 * @brief Logs weak scaling benchmark results to a CSV file.
 * @param output_path Path to the CSV output file.
 * @param size Number of MPI processes.
 * @param total_rows Total number of rows in the global matrix.
 * @param total_nnz Total number of non-zero elements in the global matrix.
 * @param time_p90_ms The 90th percentile of execution time.
 * @param gflops The calculated GFLOPS based on P90 time.
 */
void log_weak_scaling_csv(const string& output_path, int size, int total_rows, int total_nnz, 
                          double time_p90_ms, double time_comm_p90_ms, double time_comp_p90_ms, 
                          double gflops);

/**
 * @brief Run the Weak Scaling benchmark.
 * 1. Uses the provided x_local for computations + ghost exchanges.
 * 2. Runs a Warmup.
 * 3. Executes the loop 10 times.
 * 4. Collects times and computes the 90th percentile.
 * 5. Calls log_weak_scaling_csv() to report results.
 * @param mat Local CSR Matrix partition.
 * @param x_local The input vector (Already generated and distributed).
 * @param output_path Path to the CSV output file.
 * @param rows_per_proc Number of rows assigned to each process.
 * @param total_nnz Total non-zeros in the global matrix (for GFLOPS calculation).
 * @param rank MPI Rank.
 * @param size MPI Size.
 */
void run_weak_scaling(const CSRMatrix& mat, const vector<double>& x_local, 
                      const string& output_path, int rows_per_proc, 
                      long long total_nnz, int rank, int size);

#endif // UTILS_H