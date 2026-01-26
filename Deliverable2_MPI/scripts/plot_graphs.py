import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np
import colorsys
import random

# -------------------------------------------------------------------
# Global plot configuration
# -------------------------------------------------------------------
MARKER = "o"
LINEWIDTH = 2
MARKERSIZE = 8
GRID_ALPHA = 0.3
FONTSIZE_LABELS = 12
FONTSIZE_TITLE = 14
DPI = 600
FONT_FAMILY = "serif"

COLOR_COMP_STD = "#80b1d3"
COLOR_COMM_STD = "#fdb462"
COLOR_TREND = "#d62728"

plt.rcParams['font.family'] = FONT_FAMILY
plt.rcParams['mathtext.fontset'] = 'cm'

# -------------------------------------------------------------------
# Directory setup
# -------------------------------------------------------------------
def setup_dirs(base_path, sub_folder):
    path = base_path / sub_folder
    path.mkdir(parents=True, exist_ok=True)
    return path

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def finalize_plot(ax, xlabel, ylabel, title, output_file, xticks_labels, config_info=None):
    ax.set_xlabel(xlabel, fontsize=FONTSIZE_LABELS)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE_LABELS)
    ax.set_title(title, fontsize=FONTSIZE_TITLE)
    ax.grid(True, alpha=GRID_ALPHA)
    
    if config_info:
        ax.legend(loc='upper center', bbox_to_anchor=(0.35, -0.12),
                  fancybox=False, shadow=False, ncol=1, frameon=True)
        
        ax.text(0.65, -0.12, config_info, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='gray'))
    else:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                  fancybox=False, shadow=False, ncol=4, frameon=True)
    
    ax.set_xticks(range(len(xticks_labels)))
    ax.set_xticklabels(xticks_labels)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
    plt.close()

def calculate_metrics(df, is_weak):
    df = df.sort_values(by="MPI_Procs")
    
    t1_row = df[df["MPI_Procs"] == 1]
    
    if t1_row.empty:
        t1 = df["Time_P90_ms"].max() 
    else:
        t1 = t1_row["Time_P90_ms"].values[0]
    
    if is_weak:
        df["Efficiency"] = (t1 / df["Time_P90_ms"]) * 100.0 
        df["Speedup"] = df["MPI_Procs"] * (df["Efficiency"] / 100.0)
    else:
        df["Speedup"] = t1 / df["Time_P90_ms"]
        df["Efficiency"] = (df["Speedup"] / df["MPI_Procs"]) * 100.0
        
    return df

def generate_matrix_colors(n_matrices):
    colors = {}
    golden_ratio_conjugate = 0.618033988749895
    h = random.random()
    
    for i in range(n_matrices):
        h += golden_ratio_conjugate
        h %= 1
        
        r1, g1, b1 = colorsys.hsv_to_rgb(h, 0.4, 0.95)
        r2, g2, b2 = colorsys.hsv_to_rgb(h, 0.9, 0.7)
        
        colors[i] = {
            'comp': (r1, g1, b1),
            'comm': (r2, g2, b2)
        }
    return colors

# -------------------------------------------------------------------
# Plotting functions
# -------------------------------------------------------------------
def plot_time(df, output_path, title_prefix, xticks, config_info=None):
    plt.figure(figsize=(10, 7))
    proc_to_idx = {val: i for i, val in enumerate(xticks)}
    x_indices = df["MPI_Procs"].map(proc_to_idx)
    plt.plot(x_indices, df["Time_P90_ms"], marker=MARKER, linewidth=LINEWIDTH, 
             markersize=MARKERSIZE, label="Total Time (P90)")
    
    finalize_plot(plt.gca(), "MPI Processes", "Time (ms)", 
                  f"{title_prefix} - Execution Time", 
                  output_path / f"0_{title_prefix.replace(' ', '_')}_time.png", xticks, config_info)

def plot_speedup(df, output_path, title_prefix, xticks, config_info=None):
    plt.figure(figsize=(10, 7))
    proc_to_idx = {val: i for i, val in enumerate(xticks)}
    x_indices = df["MPI_Procs"].map(proc_to_idx)
    plt.plot(x_indices, df["Speedup"], marker=MARKER, linewidth=LINEWIDTH, 
             markersize=MARKERSIZE, label="Measured Speedup", color="tab:blue")
    
    finalize_plot(plt.gca(), "MPI Processes", "Speedup", 
                  f"{title_prefix} - Speedup", 
                  output_path / f"1_{title_prefix.replace(' ', '_')}_speedup.png", xticks, config_info)

def plot_efficiency(df, output_path, title_prefix, xticks, config_info=None):
    plt.figure(figsize=(10, 7))
    proc_to_idx = {val: i for i, val in enumerate(xticks)}
    x_indices = df["MPI_Procs"].map(proc_to_idx)
    plt.plot(x_indices, df["Efficiency"], marker=MARKER, linewidth=LINEWIDTH, 
             markersize=MARKERSIZE, color="tab:green", label="Efficiency (%)")
    
    finalize_plot(plt.gca(), "MPI Processes", "Efficiency (%)", 
                  f"{title_prefix} - Efficiency", 
                  output_path / f"2_{title_prefix.replace(' ', '_')}_efficiency.png", xticks, config_info)

def plot_gflops(df, output_path, title_prefix, xticks, config_info=None):
    plt.figure(figsize=(10, 7))
    proc_to_idx = {val: i for i, val in enumerate(xticks)}
    x_indices = df["MPI_Procs"].map(proc_to_idx)
    plt.plot(x_indices, df["GFLOPS"], marker=MARKER, linewidth=LINEWIDTH, 
             markersize=MARKERSIZE, color="tab:red", label="GFLOPS")

    finalize_plot(plt.gca(), "MPI Processes", "GFLOPS", 
                  f"{title_prefix} - Throughput", 
                  output_path / f"3_{title_prefix.replace(' ', '_')}_gflops.png", xticks, config_info)

def plot_comm_vs_comp(df, output_path, title_prefix, xticks, config_info=None):
    fig, ax1 = plt.subplots(figsize=(10, 7))
    comp_times = df["Time_Comp_ms"].values
    comm_times = df["Time_Comm_ms"].values
    x_indices = range(len(xticks))
    
    ax1.bar(x_indices, comp_times, label="Computation", color=COLOR_COMP_STD, alpha=0.9, edgecolor='black', linewidth=0.5)
    ax1.bar(x_indices, comm_times, bottom=comp_times, label="Communication", color=COLOR_COMM_STD, alpha=0.9, edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel("MPI Processes", fontsize=FONTSIZE_LABELS)
    ax1.set_ylabel("Time (ms)", fontsize=FONTSIZE_LABELS)
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(xticks)
    ax1.grid(axis='y', alpha=GRID_ALPHA)
    
    ax2 = ax1.twinx()
    ax2.plot(x_indices, comp_times, color=COLOR_TREND, marker="x", 
             linewidth=2, label="Comp. Time Trend", linestyle="--")
    ax2.set_ylabel("Computation Time (ms)", fontsize=FONTSIZE_LABELS, color=COLOR_TREND)
    ax2.tick_params(axis='y', labelcolor=COLOR_TREND)
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    
    # Anche qui logica divisa se c'è config_info
    if config_info:
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, 
                   loc='upper center', bbox_to_anchor=(0.35, -0.12),
                   fancybox=False, shadow=False, ncol=1, frameon=True)
        
        ax1.text(0.65, -0.12, config_info, transform=ax1.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='gray'))
    else:
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, 
                   loc='upper center', bbox_to_anchor=(0.5, -0.12),
                   fancybox=False, shadow=False, ncol=3, frameon=True)

    plt.title(f"{title_prefix} - CommVsComp Breakdown", fontsize=FONTSIZE_TITLE)
    plt.tight_layout()
    plt.savefig(output_path / f"4_{title_prefix.replace(' ', '_')}_commVScomp.png", dpi=DPI, bbox_inches='tight')
    plt.close()

def plot_strong_global_breakdown(df_strong, output_path):
    matrices = sorted(df_strong["MatrixName"].unique())
    procs = sorted(df_strong["MPI_Procs"].unique())
    
    n_matrices = len(matrices)
    n_procs = len(procs)
    
    matrix_colors = generate_matrix_colors(n_matrices)
    
    fig, ax = plt.subplots(figsize=(14, 8.5))
    
    total_width = 0.85
    bar_width = total_width / n_matrices
    offsets = np.linspace(-total_width/2 + bar_width/2, total_width/2 - bar_width/2, n_matrices)
    
    for j, matrix in enumerate(matrices):
        subset = df_strong[df_strong["MatrixName"] == matrix]
        
        t1_row = subset[subset["MPI_Procs"] == 1]
        if t1_row.empty:
            t_ref = subset["Time_P90_ms"].max()
        else:
            t_ref = t1_row["Time_P90_ms"].values[0]

        c_comp = matrix_colors[j]['comp']
        c_comm = matrix_colors[j]['comm']

        for i, p in enumerate(procs):
            row = subset[subset["MPI_Procs"] == p]
            
            if not row.empty:
                comp_norm = row["Time_Comp_ms"].values[0] / t_ref
                comm_norm = row["Time_Comm_ms"].values[0] / t_ref
                
                ax.bar(i + offsets[j], comp_norm, width=bar_width, 
                       color=c_comp, edgecolor='black', linewidth=0.3)
                ax.bar(i + offsets[j], comm_norm, bottom=comp_norm, width=bar_width, 
                       color=c_comm, edgecolor='black', linewidth=0.3)

    matrix_handles = []
    for j, matrix in enumerate(matrices):
        c_base = matrix_colors[j]['comm'] 
        patch = mpatches.Patch(facecolor=c_base, edgecolor='black', label=matrix)
        matrix_handles.append(patch)
    
    leg1 = ax.legend(handles=matrix_handles, loc='upper center', bbox_to_anchor=(0.35, -0.12),
              ncol=min(n_matrices, 4), title="Matrices", frameon=True)
    ax.add_artist(leg1)
    
    h_comp = mpatches.Patch(facecolor='lightgray', edgecolor='black', label='Computation (Light)')
    h_comm = mpatches.Patch(facecolor='gray', edgecolor='black', label='Communication (Dark)')
    
    leg2 = ax.legend(handles=[h_comp, h_comm], loc='upper center', bbox_to_anchor=(0.75, -0.12),
                     title=r"Normalization: Relative to Total Time $T_{1}$", 
                     frameon=True, ncol=1)

    ax.set_xlabel("MPI Processes", fontsize=FONTSIZE_LABELS)
    ax.set_ylabel(r"Normalized Time ($T_P / T_1$)", fontsize=FONTSIZE_LABELS)
    ax.set_title("Global Strong Scaling Breakdown", fontsize=FONTSIZE_TITLE)
    
    ax.set_xticks(range(n_procs))
    ax.set_xticklabels(procs)
    ax.grid(axis='y', alpha=GRID_ALPHA)
    
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path / "breakdown_global_normalized.png", dpi=DPI, bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------------
# Main plotting pipeline
# -------------------------------------------------------------------
def generate_plots_for_group(df, folder_path, title_prefix, is_weak, config_info=None):
    df = calculate_metrics(df, is_weak)
    xticks = sorted(df["MPI_Procs"].unique().tolist())
    
    plot_time(df, folder_path, title_prefix, xticks, config_info)
    plot_speedup(df, folder_path, title_prefix, xticks, config_info)
    plot_efficiency(df, folder_path, title_prefix, xticks, config_info)
    plot_gflops(df, folder_path, title_prefix, xticks, config_info)
    plot_comm_vs_comp(df, folder_path, title_prefix, xticks, config_info)

def run_analysis():
    current_cwd = Path.cwd()
    
    if (current_cwd / "results").exists():
        base_path = current_cwd
    elif (current_cwd / "Deliverable2_MPI" / "results").exists():
        base_path = current_cwd / "Deliverable2_MPI"
    else:
        print("ERROR: Cannot find 'results' directory.")
        return

    plots_root = base_path / "plots"
    results_root = base_path / "results"
    plots_root.mkdir(exist_ok=True)
    
    strong_file = results_root / "strong_scaling.csv"
    weak_file = results_root / "weak_scaling.csv"
    
    found_any = False

    if strong_file.exists():
        found_any = True
        print(">>> Processing Strong Scaling...")
        try:
            df_strong = pd.read_csv(strong_file)
            df_strong.columns = df_strong.columns.str.strip()
            df_strong["MatrixName"] = df_strong["MatrixName"].astype(str).str.replace("_bin", "").str.replace(".mtx", "")
            
            strong_root = setup_dirs(plots_root, "strong_scaling")
            print("   -> Generating Global Breakdown Chart...")
            plot_strong_global_breakdown(df_strong, strong_root)
            
            matrices = df_strong["MatrixName"].unique()
            for matrix in matrices:
                print(f"   -> Matrix: {matrix}")
                safe_name = matrix.replace(".", "_")
                folder = setup_dirs(strong_root, safe_name)
                subset = df_strong[df_strong["MatrixName"] == matrix].copy()
                generate_plots_for_group(subset, folder, str(matrix), is_weak=False)
        except Exception as e:
            print(f"ERROR processing strong scaling: {e}")

    if weak_file.exists():
        found_any = True
        print(">>> Processing Weak Scaling...")
        try:
            df_weak = pd.read_csv(weak_file)
            df_weak.columns = df_weak.columns.str.strip()
            df_weak["Base_Dim"] = (df_weak["Total_Rows"] / df_weak["MPI_Procs"]).astype(int)
            
            if "Total_NNZ" in df_weak.columns:
                df_weak["NNZ_Row_Avg"] = (df_weak["Total_NNZ"] / df_weak["Total_Rows"]).astype(int)
            else:
                df_weak["NNZ_Row_Avg"] = 50 

            folder = setup_dirs(plots_root, "weak_scaling")

            dims = df_weak["Base_Dim"].unique()
            for dim in dims:
                subset = df_weak[df_weak["Base_Dim"] == dim].copy()
                nnz_row = subset["NNZ_Row_Avg"].iloc[0]
                print(f"   -> Weak Config: {dim} Rows/Proc, {nnz_row} NNZ/Row")
                config_info = f"Weak Scaling Config:\nRows/Proc: {dim}\nNNZ/Row: {nnz_row}"
                title = "Weak Scaling"
                generate_plots_for_group(subset, folder, title, is_weak=True, config_info=config_info)
        except Exception as e:
            print(f"ERROR processing weak scaling: {e}")

    if not found_any:
        print("\nNO RESULTS PROCESSED. Check your paths.")

if __name__=="__main__":
    run_analysis()