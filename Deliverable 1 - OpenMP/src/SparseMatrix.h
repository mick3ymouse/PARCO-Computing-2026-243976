#ifndef SPARSEMATRIX_H
#define SPARSEMATRIX_H

#include <vector>
#include <string>
using namespace std;

// ============================================
// BASE CLASS: SparseMatrix
// ============================================
// Abstract base class for sparse matrix storage formats

class SparseMatrix {
protected:
    size_t rows;
    size_t cols;
    size_t nnz; // Number of non-zero elements

public:
    SparseMatrix(size_t _rows, size_t _cols, size_t _nnz);
    virtual ~SparseMatrix() = default;

    // Pure virtual functions - must be implemented by derived classes
    virtual void print() const = 0;
    virtual vector<double> multiply_sequential(const vector<double>& vec) const = 0;
    virtual vector<double> multiply_parallel(const vector<double>& vec) const = 0;

    // Getters
    size_t get_rows() const { return rows; }
    size_t get_cols() const { return cols; }
    size_t get_nnz() const { return nnz; }
};

// ============================================
// COO FORMAT: Coordinate List Format
// ============================================
// Stores each non-zero element as (row, col, value) triplet

class SparseMatrixCOO : public SparseMatrix {
private:
    vector<size_t> row_indices;     // Row index for each non-zero element
    vector<size_t> col_indices;     // Column index for each non-zero element
    vector<double> values;          // Value for each non-zero element

    void convert_from_dense(const vector<vector<double>>& matrix);  // Extract non-zero elements from dense matrix
    void validate() const;

public:
    // Construct from dense matrix
    SparseMatrixCOO(const vector<vector<double>>& dense_matrix);

    // Construct from COO arrays
    SparseMatrixCOO(size_t _rows, size_t _cols,
                    const vector<size_t>& _row_indices,
                    const vector<size_t>& _col_indices,
                    const vector<double>& _values);

    // Load from .mtx file
    SparseMatrixCOO(const string& filename);

    // Sparse matrix-vector multiplication (SpMV)
    vector<double> multiply_sequential(const vector<double>& vec) const override;
    vector<double> multiply_parallel(const vector<double>& vec) const override;

    void print() const override;

    // Getters
    const vector<size_t>& get_row_indices() const { return row_indices; }
    const vector<size_t>& get_col_indices() const { return col_indices; }
    const vector<double>& get_values() const { return values; }
};

// ============================================
// CSR FORMAT: Compressed Sparse Row Format
// ============================================
// Uses row_ptr array to compress row information

class SparseMatrixCSR : public SparseMatrix {
private:
    vector<size_t> row_ptr;        // Row pointers: row_ptr[i] = start index of row i in col_indices/values
    vector<size_t> col_indices;    // Column index for each non-zero element
    vector<double> values;         // Value for each non-zero element

    void convert_from_dense(const vector<vector<double>>& matrix);  // Extract non-zero elements from dense matrix
    void validate() const;

public:
    // Construct from dense matrix
    SparseMatrixCSR(const vector<vector<double>>& dense_matrix);

    // Construct from CSR arrays
    SparseMatrixCSR(size_t _rows, size_t _cols,
                    const vector<size_t>& _row_ptr,
                    const vector<size_t>& _col_indices,
                    const vector<double>& _values);

    // Load from .mtx file via COO conversion
    SparseMatrixCSR(const string& filename);

    // Sparse matrix-vector multiplication (SpMV)
    vector<double> multiply_sequential(const vector<double>& vec) const override;
    vector<double> multiply_parallel(const vector<double>& vec) const override;

    void print() const override;

    // Getters
    const vector<size_t>& get_row_ptr() const { return row_ptr; }
    const vector<size_t>& get_col_indices() const { return col_indices; }
    const vector<double>& get_values() const { return values; }
};

#endif //SPARSEMATRIX_H