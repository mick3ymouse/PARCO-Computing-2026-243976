#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

// Environmental variables that need to be set are:
// -BENCHMARK
// -MODE
// -MATRIX_FILE
// Only If BENCHMARK = "timings":
// -TIMINGS_RUNS
// -TIMINGS_FILE
// -LOG_SCHED_KIND
// -LOG_SCHED_CHUNK


// Enum for benchmark type
enum class BenchmarkMode {
    TIMINGS,
    CACHE
};

// Enum for execution (sequential or parallel)
enum class ExecutionMode {
    SEQUENTIAL,
    PARALLEL
};

class BenchmarkConfig {
private:
    // Helper parsers
    static string get_env_var(const string& var_name) ;
    BenchmarkMode parse_benchmark_mode(const string& s);
    ExecutionMode parse_execution_mode(const string& s);


public:
    // Configuration values loaded from environment
    BenchmarkMode benchmark_mode;
    ExecutionMode execution_mode;
    string matrix_file;
    size_t timings_runs;
    string timings_file;

    // OpenMP values for logging
    int num_threads;
    string s_kind;
    string s_chunk_size;

    // Loads all configuration from environment variables
    BenchmarkConfig();

    // Prints the loaded configuration
    void print_config() const;
};

#endif // CONFIG_H