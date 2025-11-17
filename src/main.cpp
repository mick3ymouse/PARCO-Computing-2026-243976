#include <iostream>
#include <stdexcept>
#include <exception>
#include "SparseMatrix.h"
#include "Utils.h"
#include "Config.h"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

int main() {
    try {
        // 1. Load and validate configuration from environment
        BenchmarkConfig config;
        //config.print_config();

        // 2. Load sparse matrix
        cout << "Loading matrix: " << config.matrix_file << endl;
        SparseMatrixCSR mtx_csr(config.matrix_file);
        cout << "Matrix size: " << mtx_csr.get_rows() << "x" << mtx_csr.get_cols()
             << " (nnz=" << mtx_csr.get_nnz() << ")" << endl;

        // 3. Prepare input vector with random values
        vector<double> vec(mtx_csr.get_cols());
        fill_random_vector(vec);
        vector<double> res(mtx_csr.get_rows(), 0.0);

        // 4. Run benchmark based on config
        double start, end, time_ms;

        switch (config.benchmark_mode) {
            case BenchmarkMode::TIMINGS:
                for (int i=0; i < config.timings_runs; ++i) {
                    start = omp_get_wtime();
                    switch (config.execution_mode) {
                        case ExecutionMode::SEQUENTIAL:
                            res = mtx_csr.multiply_sequential(vec);
                            break;
                        case ExecutionMode::PARALLEL:
                            res = mtx_csr.multiply_parallel(vec);
                            break;
                    }
                    end = omp_get_wtime();
                    time_ms = (end - start) * 1000.0;
                    log_to_csv(config, time_ms);
                }
            break;
            case BenchmarkMode::CACHE:
                switch (config.execution_mode) {
                    case ExecutionMode::SEQUENTIAL:
                        res = mtx_csr.multiply_sequential(vec);
                        break;
                    case ExecutionMode::PARALLEL:
                        res = mtx_csr.multiply_parallel(vec);
                        break;
                }
            break;
        }

    } catch (const runtime_error& e) {
        cerr << "Runtime Error: " << e.what() << endl;
        return 1;
    } catch (const invalid_argument& e) {
        cerr << "Invalid Argument: " << e.what() << endl;
        return 1;
    } catch (const exception& e) {
        cerr << "Unexpected Error: " << e.what() << endl;
        return 1;
    } catch (...) {
        cerr << "Unknown error occurred" << endl;
        return 1;
    }

    return 0;
}
