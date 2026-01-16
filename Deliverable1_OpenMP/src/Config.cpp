#include "Config.h"
#include <stdexcept>
#include <cstdlib>

#ifdef _OPENMP
#include <omp.h>
#endif

// Helper to safely get environment variables
string BenchmarkConfig::get_env_var(const string& var_name) {
    char* env_val = getenv(var_name.c_str());
    if (env_val == nullptr) {
        throw runtime_error("Environment variable not set: " + var_name);
    }
    string result(env_val);
    return result;
}

// Helper to parse BENCHMARK variable
BenchmarkMode BenchmarkConfig::parse_benchmark_mode(const string& s) {
    if (s == "timings") return BenchmarkMode::TIMINGS;
    if (s == "cache") return BenchmarkMode::CACHE;
    throw invalid_argument("Invalid BENCHMARK value: " + s);
}

// Helper to parse MODE variable
ExecutionMode BenchmarkConfig::parse_execution_mode(const string& s) {
    if (s == "sequential") return ExecutionMode::SEQUENTIAL;
    if (s == "parallel") return ExecutionMode::PARALLEL;
    throw invalid_argument("Invalid MODE value: " + s);
}

// Constructor: Load all config from environment
BenchmarkConfig::BenchmarkConfig() {
    // Load main config
    benchmark_mode = parse_benchmark_mode(get_env_var("BENCHMARK"));
    execution_mode = parse_execution_mode(get_env_var("MODE"));
    matrix_file = get_env_var("MATRIX_FILE");

    // Set number of threads based on execution mode
    switch (execution_mode) {
        case ExecutionMode::PARALLEL:
#ifdef _OPENMP
            num_threads = omp_get_max_threads();
#else
            num_threads = 1;
#endif
            break;
        case ExecutionMode::SEQUENTIAL:
            num_threads = 1;
            break;
    }

    // Load benchmark-specific configuration
    switch (benchmark_mode) {
        case BenchmarkMode::TIMINGS:
            timings_runs = stoi(get_env_var("TIMINGS_RUNS"));
            timings_file = get_env_var("TIMINGS_FILE");

            // Load schedule config only for parallel timings
            if (execution_mode == ExecutionMode::PARALLEL) {
                s_kind = get_env_var("LOG_SCHED_KIND");
                s_chunk_size = get_env_var("LOG_SCHED_CHUNK");
            } else {
                s_kind = "NaN";
                s_chunk_size = "NaN";
            }
            break;
        case BenchmarkMode::CACHE:
            // CACHE mode doesn't need timings or schedule config
            timings_runs = 1;
            timings_file = "NaN";
            s_kind = "NaN";
            s_chunk_size = "NaN";
            break;
    }
}


// Print the loaded configuration
void BenchmarkConfig::print_config() const {
    cout << "--- Benchmark Configuration ---" << endl;
    cout << "  Matrix File: " << matrix_file << endl;

    // Print benchmark type
    switch (benchmark_mode) {
        case BenchmarkMode::TIMINGS:
            cout << "  Benchmark:   TIMINGS" << endl;
            cout << "  Timings Runs: " << timings_runs << endl;
            cout << "  Timings File: " << timings_file << endl;
            break;
        case BenchmarkMode::CACHE:
            cout << "  Benchmark:   CACHE_MISSES" << endl;
            break;
    }

    // Print execution mode
    switch (execution_mode) {
        case ExecutionMode::SEQUENTIAL:
            cout << "  Mode:        SEQUENTIAL" << endl;
            break;
        case ExecutionMode::PARALLEL:
            cout << "  Mode:        PARALLEL" << endl;
            cout << "  Threads:     " << num_threads << endl;

            // Print schedule info only for parallel timings
            if (benchmark_mode == BenchmarkMode::TIMINGS) {
                cout << "  Sched Kind:  " << s_kind << endl;
                cout << "  Sched Chunk: " << s_chunk_size << endl;
            }
            break;
    }

    cout << "-------------------------------" << endl;
}