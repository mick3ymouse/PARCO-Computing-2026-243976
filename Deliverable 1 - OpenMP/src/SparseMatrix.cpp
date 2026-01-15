#include "SparseMatrix.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif


// ============================================
// BASE CLASS: SparseMatrix
// ============================================

SparseMatrix::SparseMatrix(size_t _rows, size_t _cols, size_t _nnz)
    : rows(_rows), cols(_cols), nnz(_nnz) {}

// ============================================
// COO FORMAT: SparseMatrixCOO
// ============================================

SparseMatrixCOO::SparseMatrixCOO(const vector<vector<double>>& dense_matrix)
    : SparseMatrix(dense_matrix.size(),
                    dense_matrix[0].size(),
                    0) {
    convert_from_dense(dense_matrix);
}

SparseMatrixCOO::SparseMatrixCOO(size_t _rows, size_t _cols,
                                const vector<size_t>& _row_indices,
                                const vector<size_t>& _col_indices,
                                const vector<double>& _values)
    : SparseMatrix(_rows, _cols, _values.size()),
      row_indices(_row_indices),
      col_indices(_col_indices),
      values(_values) {
    validate();
}

// Load from .mtx file and sort entries by row then column
SparseMatrixCOO::SparseMatrixCOO(const string &filename)
    : SparseMatrix (0, 0, 0) {

    ifstream file(filename);
    if (!file) {
        throw runtime_error("Cannot open .mtx file: " + filename);
    }

    string line;

    // Skip comment lines
    while (getline(file, line)) {
        if (!line.empty() && line[0] != '%') {
            break;
        }
    }

    // Parse header: rows cols nnz
    istringstream header(line);
    header >> rows >> cols >> nnz;

    size_t r, c;
    double val;

    // Temporary structure to hold triplets during sorting
    struct COO_triplet {
        size_t row;
        size_t col;
        double val;
    };

    vector<COO_triplet> entries;
    entries.reserve(nnz);

    // Read entries and convert from 1-based to 0-based indexing
    while (file >> r >> c >> val) {
        entries.push_back({r-1, c-1, val});
    }

    file.close();

    nnz = entries.size();

    // Sort by row, then by column
    sort(entries.begin(), entries.end(),
        [] (const COO_triplet& a, const COO_triplet& b) {
            if (a.row == b.row) {
                return a.col < b.col;
            }
            return a.row < b.row;
        }
    );

    // Fill COO arrays
    row_indices.reserve(nnz);
    col_indices.reserve(nnz);
    values.reserve(nnz);

    for (const auto& entry : entries) {
        row_indices.push_back(entry.row);
        col_indices.push_back(entry.col);
        values.push_back(entry.val);
    }

    validate();
}

void SparseMatrixCOO::convert_from_dense(const vector<vector<double>>& matrix) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            if (matrix[i][j] != 0) {
                row_indices.push_back(i);
                col_indices.push_back(j);
                values.push_back(matrix[i][j]);
            }
        }
    }
    nnz = values.size();
}

void SparseMatrixCOO::validate() const {
    if (row_indices.size() != col_indices.size() ||
        row_indices.size() != values.size()) {
        throw invalid_argument("COO arrays must have same length");
        }
}

// SpMV using COO format: result[i] += A[i,j] * vec[j] for each non-zero A[i,j]
vector<double> SparseMatrixCOO::multiply_sequential(const vector<double>& vec) const {
    if (vec.size() != cols) {
        throw invalid_argument("Vector size doesn't match matrix columns");
    }

    vector<double> result(rows, 0.0);

    for (size_t i = 0; i < nnz; i++) {
        result[row_indices[i]] += values[i] * vec[col_indices[i]];
    }

    return result;
}

// Parallel SpMV using OpenMP
vector<double> SparseMatrixCOO::multiply_parallel(const vector<double> &vec) const {
    if (vec.size() != cols) {
        throw invalid_argument("Vector size doesn't match matrix columns");
    }

    vector<double> result(rows, 0.0);

    #pragma omp parallel for default(none)\
    shared(result, vec) \
    schedule(runtime)
    for (size_t i = 0; i < nnz; i++) {
        #pragma omp atomic
        result[row_indices[i]] += values[i] * vec[col_indices[i]];
    }

    return result;
}

void SparseMatrixCOO::print() const {
    cout << "COO Format (" << rows << "x" << cols << ", "
    << nnz << " non-zero elements):" << endl;
    cout << "Row indices: [ ";
    for (const auto& idx_row : row_indices) cout << idx_row << " ";
    cout << "]" << endl;
    cout << "Col indices: [ ";
    for (const auto& idx_col : col_indices) cout << idx_col << " ";
    cout << "]" << endl;
    cout << "Values: [ ";
    for (const auto& val : values) cout << val << " ";
    cout << "]" << endl;
}


// ============================================
// CSR FORMAT: SparseMatrixCSR
// ============================================

SparseMatrixCSR::SparseMatrixCSR(const vector<vector<double>>& dense_matrix)
    : SparseMatrix(dense_matrix.size(),
                    dense_matrix[0].size(),
                    0) {
    convert_from_dense(dense_matrix);
}

SparseMatrixCSR::SparseMatrixCSR(size_t _rows, size_t _cols,
                                const vector<size_t>& _row_ptr,
                                const vector<size_t>& _col_indices,
                                const vector<double>& _values)
    : SparseMatrix(_rows, _cols, _values.size()),
                  row_ptr(_row_ptr),
                  col_indices(_col_indices),
                  values(_values) {
    validate();
}

// Load from .mtx file via COO intermediate format
SparseMatrixCSR::SparseMatrixCSR(const string &filename)
    : SparseMatrix(0, 0, 0) {

    // Load in COO format first
    SparseMatrixCOO coo(filename);

    rows = coo.get_rows();
    cols = coo.get_cols();
    nnz = coo.get_nnz();

    const vector<size_t>& coo_row_indices = coo.get_row_indices();

    // Allocate CSR storage
    row_ptr.resize(rows + 1, 0);
    col_indices = coo.get_col_indices();
    values = coo.get_values();

    // Build row_ptr by counting non-zeros per row
    for (size_t i = 0; i < nnz; i++) {
        row_ptr[coo_row_indices[i] + 1]++;
    }

    // Convert counts to cumulative offsets
    for (size_t i = 1; i <= rows; i++) {
        row_ptr[i] += row_ptr[i - 1];
    }

    validate();
}

void SparseMatrixCSR::convert_from_dense(const vector<vector<double>>& matrix) {
    row_ptr.push_back(0);

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            if (matrix[i][j] != 0) {
                col_indices.push_back(j);
                values.push_back(matrix[i][j]);
                ++nnz;
            }
        }
        row_ptr.push_back(nnz);
    }
}

void SparseMatrixCSR::validate() const {
    if (row_ptr.size() != rows + 1) {
        throw invalid_argument("row_ptr must have size rows+1");
    }
    if (col_indices.size() != values.size()) {
        throw invalid_argument("col_indices and values must have same size");
    }
}

// SpMV using CSR format: compute dot product for each row
vector<double> SparseMatrixCSR::multiply_sequential(const vector<double> &vec) const {
    if (vec.size() != static_cast<size_t>(cols)) {
        throw invalid_argument("Vector size doesn't match matrix columns");
    }

    vector<double> result(rows, 0.0);

    for (size_t i = 0; i < rows; i++) {
        size_t row_start = row_ptr[i];
        size_t row_end = row_ptr[i + 1];

        for (size_t idx = row_start; idx < row_end; idx++) {
            result[i] += values[idx] * vec[col_indices[idx]];
        }
    }

    return result;
}

// Parallel SpMV using OpenMP
vector<double> SparseMatrixCSR::multiply_parallel(const vector<double> &vec) const {
    if (vec.size() != static_cast<size_t>(cols)) {
        throw invalid_argument("Vector size doesn't match matrix columns");
    }

    vector<double> result(rows, 0.0);

    #pragma omp parallel for default(none)\
    shared(result, vec) \
    schedule(runtime)
    for (size_t i = 0; i < rows; i++) {
        size_t row_start = row_ptr[i];
        size_t row_end = row_ptr[i + 1];

        for (size_t idx = row_start; idx < row_end; idx++) {
            result[i] += values[idx] * vec[col_indices[idx]];
        }
    }

    return result;
}

void SparseMatrixCSR::print() const {
    cout << "CSR Format (" << rows << "x" << cols << ", " << nnz
         << " non-zero elements):" << endl;
    cout << "Row ptr: [ ";
    for (const auto idx_row : row_ptr) cout << idx_row << " ";
    cout << "]" << endl;
    cout << "Col indices: [ ";
    for (const auto idx_col : col_indices) cout << idx_col << " ";
    cout << "]" << endl;
    cout << "Values: [ ";
    for (const auto val : values) cout << val << " ";
    cout << "]" << endl;
}