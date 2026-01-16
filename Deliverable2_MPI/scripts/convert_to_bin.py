import sys
import os
import glob
import scipy.io
import scipy.sparse
import numpy as np
import struct

def convert_mtx_to_bin(input_path):
    """
    Reads an .mtx file, handles symmetry, converts to CSR, 
    and saves binary files for C++/MPI in a specific subfolder.
    """
    
    # --- 1. Setup Output Directory ---

    # Extract filename without extension (e.g., "matrices/memchip.mtx" -> "memchip")
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    # Create output folder (e.g., "matrices/memchip_bin")
    output_dir = os.path.join(os.path.dirname(input_path), f"{base_name}_bin")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[INFO] Created output directory: {output_dir}")


    # --- 2. Read Matrix Market File ---

    print(f"[INFO] Reading: {input_path}")

    # Check header for symmetry info before loading full data
    mm_info = scipy.io.mminfo(input_path)
    is_symmetric = 'symmetric' in mm_info
    
    # Load matrix as Coordinate format (COO)
    mat = scipy.io.mmread(input_path).tocoo()

    # --- 3. Handle Symmetry ---
    if is_symmetric:
        print("[INFO] Symmetric matrix detected. Expanding to full General format...")

        # Convert to CSR to allow arithmetic operations
        mat = mat.tocsr()

        # Formula: A_full = A + A^T - diag(A)
        mat = mat + mat.transpose() - scipy.sparse.diags(mat.diagonal())

    # --- 4. Convert to Final CSR ---
    # Ensure standard CSR format (sorted indices, no duplicates)
    mat = mat.tocsr()
    
    M, N = mat.shape
    nnz = mat.nnz
    print(f"[INFO] Final CSR: {M} rows, {N} cols, {nnz} nnz")

    # --- 5. Data Type Conversion ---
    # Crucial: Map Python types to C++ types (int -> int32, float -> double)
    row_ptr = mat.indptr.astype(np.int32)
    col_ind = mat.indices.astype(np.int32)
    values  = mat.data.astype(np.float64)

    # --- 6. Write Binary Files ---
    # Paths for output files
    path_meta = os.path.join(output_dir, "meta.bin")
    path_row  = os.path.join(output_dir, "row_ptr.bin")
    path_col  = os.path.join(output_dir, "col_ind.bin")
    path_val  = os.path.join(output_dir, "val.bin")

    # Write Metadata (Rows, Cols, NNZ) -> 3 * int32
    with open(path_meta, "wb") as f:
        f.write(struct.pack("iii", M, N, nnz))
    
    # Write CSR Arrays (Raw Binary)
    row_ptr.tofile(path_row)
    col_ind.tofile(path_col)
    values.tofile(path_val)

    print(f"[SUCCESS] Binaries saved in: {output_dir}/")

if __name__ == "__main__":

    # 1. Locate the 'matrices' directory relative to this script
    # Go up one level from 'scripts/' to find 'matrices/'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    matrices_dir = os.path.join(os.path.dirname(script_dir), "matrices")
    
    print(f"Looking for .mtx files in: {matrices_dir}")

    # 2. Find all .mtx files using glob
    mtx_files = glob.glob(os.path.join(matrices_dir, "*.mtx"))

    if not mtx_files:
        print("[WARNING] No .mtx files found!")
    else:
        print(f"Found {len(mtx_files)} files. Starting conversion...\n")
        
        # 3. Process each file
        for file_path in mtx_files:
            try:
                convert_mtx_to_bin(file_path)
            except Exception as e:
                print(f"[ERROR] Failed to convert {file_path}: {e}")

print("\n--- All files converted ---")