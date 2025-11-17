#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include "Config.h"

using namespace std;

void print_vector(vector<double> array);
void print_dense_matrix(vector<vector<double>> matrix);
void log_to_csv(const BenchmarkConfig& config, double time_ms);
void fill_random_vector(vector<double>& vec);

#endif //UTILS_H