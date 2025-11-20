import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------------------------------------------
# Global plot configuration
# -------------------------------------------------------------------
SCHEDULE_COLORS = {'static': 'blue', 'dynamic': 'green', 'guided': 'red'}
MARKER = 'o'
LINEWIDTH = 2
MARKERSIZE = 7
GRID_ALPHA = 0.3
FONTSIZE_LABELS = 11
FONTSIZE_TITLE = 12
ORDER_SCHEDULES = ['static', 'dynamic', 'guided']

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def setup_dirs(matrices):
    """Create directories for plots and results."""
    plots = Path("plots"); plots.mkdir(exist_ok=True)
    results = Path("results"); results.mkdir(exist_ok=True)

    dirs = {}
    for matrix in matrices:
        safe = matrix.replace(".", "_")
        matrix_base = plots / safe
        matrix_base.mkdir(exist_ok=True)

        dirs[matrix] = {}
        subfolders = ["0_tuning", "1_efficiency", "2_speedup", "3_ipc", "4_cache_misses"]
        for name in subfolders:
            subdir = matrix_base / name
            subdir.mkdir(exist_ok=True)
            dirs[matrix][name] = subdir

    print("✓ Directories set up")
    return dirs, results


def compute_p90(df):
    """Compute 90th percentile for time and GFLOPs."""
    group_cols = ['Mode', 'Matrix', 'Threads', 'Schedule', 'ChunkSize']
    p90_time = df.groupby(group_cols, sort=False)['Time_ms'].quantile(0.9).reset_index()
    p90_gflops = df.groupby(group_cols, sort=False)['GFLOPs'].quantile(0.9).reset_index()
    p90_time.rename(columns={'Time_ms': 'p90_ms'}, inplace=True)
    p90_gflops.rename(columns={'GFLOPs': 'p90_GFLOPs'}, inplace=True)
    p90_df = pd.merge(p90_time, p90_gflops, on=group_cols, how='left')
    print("✓ Computed 90th percentile metrics")
    return p90_df


def compute_performance(p90_df):
    """Compute speedup and efficiency."""
    seq = p90_df[p90_df['Mode'] == 'sequential']
    par = p90_df[p90_df['Mode'] == 'parallel']
    data = []

    for matrix in par['Matrix'].unique():
        t_seq = seq[seq['Matrix'] == matrix]['p90_ms'].values[0]
        df_m = par[par['Matrix'] == matrix]
        for _, row in df_m.iterrows():
            speedup = t_seq / row['p90_ms']
            efficiency = (speedup / row['Threads']) * 100
            data.append({
                'Matrix': matrix,
                'Schedule': row['Schedule'],
                'Threads': row['Threads'],
                'ChunkSize': row['ChunkSize'],
                'p90_ms': row['p90_ms'],
                'p90_GFLOPs': row['p90_GFLOPs'],
                'Speedup': speedup,
                'Efficiency': efficiency
            })
    df = pd.DataFrame(data)
    print("✓ Performance metrics computed (Speedup & Efficiency)")
    return df


def find_best_configurations(speedup_df, results_dir):
    """Find best chunk size per matrix and schedule based on median speedup."""
    median_speedups = speedup_df.groupby(
        ['Matrix', 'Schedule', 'ChunkSize'], sort=False
    )['Speedup'].median().reset_index()
    idx = median_speedups.groupby(['Matrix', 'Schedule'], sort=False)['Speedup'].idxmax()
    best_configs = median_speedups.loc[idx][['Matrix', 'Schedule', 'ChunkSize']]
    best_configs.to_csv(results_dir / "best_configs.csv", index=False)
    print("✓ Best configurations saved to CSV (original order preserved)")
    return best_configs


def load_best_speedup_curves(speedup_df, best_configs):
    """Filter speedup dataframe to only include best chunk sizes."""
    merged = pd.merge(speedup_df, best_configs[['Matrix', 'Schedule', 'ChunkSize']],
                      on=['Matrix', 'Schedule', 'ChunkSize'], how='inner')
    print("✓ Loaded best speedup curves")
    return merged


def export_cache_miss_rates(cache_df, results_dir):
    """Compute L1 and LLC miss rates and export CSV."""
    cache_df["L1_MissRate"] = (cache_df["Avg_L1_Misses"] / cache_df["Avg_L1_Loads"]) * 100
    cache_df["LLC_MissRate"] = (cache_df["Avg_LLC_Misses"] / cache_df["Avg_LLC_Loads"]) * 100
    out = cache_df[[
        "Mode", "Matrix", "Threads", "Schedule", "ChunkSize",
        "L1_MissRate", "LLC_MissRate"
    ]]
    out.to_csv(results_dir / "cache_miss_rates.csv", index=False)
    print("✓ Cache miss rates exported")
    return out

# -------------------------------------------------------------------
# Generic plotting helpers
# -------------------------------------------------------------------
def plot_line(x, y, label, color, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.plot(x, y, marker=MARKER, linewidth=LINEWIDTH, markersize=MARKERSIZE,
            color=color, label=label)
    return ax


def finalize_plot(ax, xlabel, ylabel, title, output_file):
    ax.set_xlabel(xlabel, fontsize=FONTSIZE_LABELS)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE_LABELS)
    ax.set_title(title, fontsize=FONTSIZE_TITLE)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_heatmap(pivot, xlabel, ylabel, title, output_file, cmap='viridis', vmin=None, vmax=None):
    plt.figure(figsize=(10, 6))
    im = plt.imshow(pivot, aspect='auto', cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar(im, label=ylabel)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            plt.text(j, i, f'{val:.2f}' if isinstance(val, float) else f'{val:.0f}%',
                     ha="center", va="center",
                     color="black", fontsize=9, weight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7))
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------------
# Plotting functions
# -------------------------------------------------------------------
def plot_heatmap_speedup(speedup_df, schedule, output_dir):
    matrix = speedup_df['Matrix'].iloc[0]
    pivot = speedup_df[speedup_df['Schedule'] == schedule].pivot_table(index='Threads', columns='ChunkSize', values='Speedup')
    safe = matrix.replace(".", "_")
    plot_heatmap(pivot, "Chunk Size", "Speedup", f"{matrix} - Speedup Heatmap ({schedule.upper()})",
                 output_dir / f"{safe}_speedup_heatmap_{schedule}.png")
    print(f"✓ Speedup heatmap plotted for {matrix} ({schedule})")


def plot_heatmap_efficiency(speedup_df, schedule, output_dir):
    matrix = speedup_df['Matrix'].iloc[0]
    pivot = speedup_df[speedup_df['Schedule'] == schedule].pivot_table(index='Threads', columns='ChunkSize', values='Efficiency')
    safe = matrix.replace(".", "_")
    plot_heatmap(pivot, "Chunk Size", "Efficiency (%)", f"{matrix} - Efficiency Heatmap ({schedule.upper()})",
                 output_dir / f"{safe}_efficiency_{schedule}.png", cmap='RdYlGn', vmin=0, vmax=100)
    print(f"✓ Efficiency heatmap plotted for {matrix} ({schedule})")


def plot_final_speedup_comparison(best_curves_df, output_dirs):
    matrices = best_curves_df['Matrix'].unique()
    for matrix in matrices:
        df_m = best_curves_df[best_curves_df['Matrix'] == matrix]
        output_dir = output_dirs[matrix]['2_speedup']
        fig, ax = plt.subplots(figsize=(10, 6))

        for sched in ORDER_SCHEDULES:
            df_s = df_m[df_m['Schedule'] == sched].sort_values('Threads')
            best_chunk = int(df_s['ChunkSize'].iloc[0])
            plot_line(df_s['Threads'], df_s['Speedup'], f"{sched.upper()} (Chunk={best_chunk})", SCHEDULE_COLORS[sched], ax=ax)

        safe = matrix.replace(".", "_")
        finalize_plot(ax, "Number of Threads", "Speedup",
                      f"{matrix} - Optimized Scalability (Best Chunks)",
                      output_dir / f"{safe}_FINAL_speedup.png")
        print(f"✓ Final speedup comparison plotted for {matrix}")


def plot_ipc_vs_threads(cache_df, output_dirs):
    matrices = cache_df['Matrix'].unique()
    for matrix in matrices:
        df_m = cache_df[cache_df['Matrix'] == matrix]
        output_dir = output_dirs[matrix]["3_ipc"]
        fig, ax = plt.subplots(figsize=(10, 6))

        for sched in ORDER_SCHEDULES:
            for chunk in df_m[df_m['Schedule'] == sched]['ChunkSize'].unique():
                df_g = df_m[(df_m['Schedule'] == sched) & (df_m['ChunkSize'] == chunk) & (df_m['Mode'] == 'parallel')].sort_values('Threads')
                plot_line(df_g["Threads"], df_g["Avg_IPC"], f"{sched.upper()} (Chunk={chunk})", SCHEDULE_COLORS[sched], ax=ax)

        seq_ipc = df_m[df_m["Mode"] == "sequential"]["Avg_IPC"].iloc[0]
        ax.scatter([1], [seq_ipc], color="black", marker="D", s=100, label="Sequential")
        safe = matrix.replace(".", "_")
        finalize_plot(ax, "Number of Threads", "Avg IPC", f"{matrix} - IPC vs Threads",
                      output_dir / f"{safe}_IPC_vs_threads.png")
        print(f"✓ IPC plot generated for {matrix}")


def plot_cache_miss_rates(cache_miss_df, output_dirs):
    matrices = cache_miss_df["Matrix"].unique()
    for matrix in matrices:
        df_m = cache_miss_df[cache_miss_df["Matrix"] == matrix]
        cache_dir = output_dirs[matrix]["4_cache_misses"]
        df_par = df_m[df_m["Mode"] == "parallel"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, miss_type in zip(axes, ["L1_MissRate", "LLC_MissRate"]):
            for sched in ORDER_SCHEDULES:
                for chunk in df_par[df_par['Schedule'] == sched]['ChunkSize'].unique():
                    df_g = df_par[(df_par['Schedule'] == sched) & (df_par['ChunkSize'] == chunk)].sort_values('Threads')
                    plot_line(df_g["Threads"], df_g[miss_type], f"{sched.upper()} (Chunk={chunk})", SCHEDULE_COLORS[sched], ax=ax)
            seq_val = df_m[df_m["Mode"] == "sequential"][miss_type].iloc[0]
            ax.scatter([1], [seq_val], color="black", marker="D", s=80, label="Sequential")
            ax.set_xlabel("Threads")
            ax.set_ylabel(f"{miss_type.replace('_',' ')} (%)")
            ax.set_title(f"{matrix} – {miss_type.replace('_',' ')}")
            ax.grid(True, alpha=GRID_ALPHA)
            ax.legend()

        plt.tight_layout()
        safe = matrix.replace(".", "_")
        plt.savefig(cache_dir / f"{safe}_cache_misses.png", dpi=300)
        plt.close()
        print(f"✓ Cache miss rates plotted for {matrix}")


# -------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------
def run_analysis():
    timngs_df = pd.read_csv("results/benchmark_timings.csv")
    timngs_df['Schedule'] = timngs_df['Schedule'].fillna('none')
    timngs_df['ChunkSize'] = timngs_df['ChunkSize'].fillna(0).astype(int)

    matrices = timngs_df['Matrix'].unique()
    dirs, results_dir = setup_dirs(matrices)

    p90_df = compute_p90(timngs_df)
    speedup_df = compute_performance(p90_df)
    speedup_df.to_csv(results_dir / "speedup_efficiency.csv", index=False)

    for matrix in matrices:
        df_matrix_speedup = speedup_df[speedup_df['Matrix'] == matrix]
        for schedule in ORDER_SCHEDULES:
            plot_heatmap_speedup(df_matrix_speedup, schedule, dirs[matrix]["0_tuning"])
            plot_heatmap_efficiency(df_matrix_speedup, schedule, dirs[matrix]["1_efficiency"])

    best_configs = find_best_configurations(speedup_df, results_dir)
    best_curves_df = load_best_speedup_curves(speedup_df, best_configs)
    plot_final_speedup_comparison(best_curves_df, dirs)

    cache_df = pd.read_csv("results/benchmark_cache.csv")
    cache_df['Schedule'] = cache_df['Schedule'].fillna('none')
    cache_df['ChunkSize'] = cache_df['ChunkSize'].fillna(0).astype(int)

    plot_ipc_vs_threads(cache_df, dirs)
    cache_miss_df = export_cache_miss_rates(cache_df, results_dir)
    plot_cache_miss_rates(cache_miss_df, dirs)


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------
if __name__ == "__main__":
    run_analysis()
