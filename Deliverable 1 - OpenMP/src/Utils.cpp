#include "Utils.h"
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

void print_vector(vector<double> array) {
    cout << "[ ";
    for (const auto& item : array) {
        cout << item << " ";
    }
    cout << "]" << endl;
}

void print_dense_matrix(vector<vector<double>> matrix) {
    for (const auto& row : matrix) {
        cout << "[ ";
        for (const auto& item : row) {
            cout << item << " ";
        }
        cout << "]" << endl;
    }
}

void log_to_csv(const BenchmarkConfig& config, size_t nnz, double time_ms) {
    // Check if file is empty to write header
    ofstream file(config.timings_file, ios_base::app);
    if (!file) {
        throw runtime_error("Cannot open CSV file: " + config.timings_file);
    }

    // Extract matrix filename from path
    string matrix_name_for_log = config.matrix_file;
    size_t pos = config.matrix_file.find_last_of('/');
    if (pos != string::npos) {
        matrix_name_for_log = config.matrix_file.substr(pos + 1);
    }

    // Write header if file is empty
    file.seekp(0, ios_base::end);
    if (file.tellp() == 0) {
        file << "Mode,Matrix,Threads,Schedule,ChunkSize,Time_ms,GFLOPs\n";
    }

    // Calculate GFLOPs: GFLOPs = (2*nnz)/(time_ms*10^6)
    double glfops = ( 2.0 * static_cast<double>(nnz) )/(time_ms* 1e6);

    // Write benchmark data
    switch (config.execution_mode) {
        case ExecutionMode::SEQUENTIAL:
            file << "sequential" << ",";
            break;
        case ExecutionMode::PARALLEL:
            file << "parallel" << ",";
            break;
    }
    file << matrix_name_for_log << ","
         << config.num_threads << ","
         << config.s_kind << ","
         << config.s_chunk_size << ","
         << time_ms << ","
         << glfops << "\n";

    file.close();
}

void fill_random_vector(vector<double>& vec) {
    static bool seeded = false;
    if (!seeded) {
        srand(static_cast<unsigned int>(time(nullptr)));
        seeded = true;
    }

    // Fill with random values in [-1, 1)
    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = (static_cast<double>(rand()) / RAND_MAX) * 2.0 - 1.0;
    }
}