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
# Directory setup
# -------------------------------------------------------------------

"""Create directories for plots and results."""
def setup_dirs(matrices):
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


# -------------------------------------------------------------------
# Helpers
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


def plot_heatmap(pivot, xlabel, ylabel, title, output_file, highlight=None, cmap='viridis', vmin=None, vmax=None):
    plt.figure(figsize=(10, 6))
    im = plt.imshow(pivot, aspect='auto', cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar(im, label=title.split(" - ")[-1])  # usa Speedup o Efficiency come label della colorbar

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            bbox_style = dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7)
            if highlight is not None and pivot.index[i] == highlight['ChunkSize'] and pivot.columns[j] == highlight['Threads']:
                bbox_style = dict(boxstyle="round,pad=0.3", fc="yellow", ec="red", lw=2, alpha=0.9)
            plt.text(j, i, f'{val:.2f}' if isinstance(val, float) else f'{val:.0f}%',
                     ha="center", va="center", fontsize=9, weight='bold', bbox=bbox_style)

    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

"""Compute L1 and LLC miss rates and export CSV."""
def export_cache_miss_rates(cache_df, results_dir):
    cache_df["L1_MissRate"] = (cache_df["Avg_L1_Misses"] / cache_df["Avg_L1_Loads"]) * 100
    cache_df["LLC_MissRate"] = (cache_df["Avg_LLC_Misses"] / cache_df["Avg_LLC_Loads"]) * 100

    out = cache_df[[
        "Mode", "Matrix", "Threads", "Schedule", "ChunkSize",
        "L1_MissRate", "LLC_MissRate"
    ]]

    out.to_csv(results_dir / "cache_miss_rates.csv", index=False)
    print("✓ Cache miss rates exported to CSV")
    return out


# -------------------------------------------------------------------
# Plotting functions
# -------------------------------------------------------------------

def plot_heatmap_speedup(speedup_df, schedule, output_dir, best_configs=None):
    matrix = speedup_df['Matrix'].iloc[0]
    df_sched = speedup_df[speedup_df['Schedule'] == schedule]
    pivot = df_sched.pivot_table(index='ChunkSize', columns='Threads', values='Speedup')

    highlight = None
    if best_configs is not None:
        row = best_configs[(best_configs['Matrix'] == matrix) & (best_configs['Schedule'] == schedule)]
        if not row.empty:
            highlight = {'Threads': df_sched['Threads'].iloc[0], 'ChunkSize': int(row['ChunkSize'].iloc[0])}

    safe = matrix.replace(".", "_")
    plot_heatmap(pivot, "Threads", "Chunk Size", f"{matrix} - Speedup Heatmap ({schedule.upper()})",
                 output_dir / f"{safe}_speedup_heatmap_{schedule}.png", highlight=highlight)
    print(f"✓ Speedup heatmap plotted for {matrix} ({schedule})")


def plot_heatmap_efficiency(speedup_df, schedule, output_dir, best_configs=None):
    matrix = speedup_df['Matrix'].iloc[0]
    df_sched = speedup_df[speedup_df['Schedule'] == schedule]
    pivot = df_sched.pivot_table(index='ChunkSize', columns='Threads', values='Efficiency')

    highlight = None
    if best_configs is not None:
        row = best_configs[(best_configs['Matrix'] == matrix) & (best_configs['Schedule'] == schedule)]
        if not row.empty:
            highlight = {'Threads': df_sched['Threads'].iloc[0], 'ChunkSize': int(row['ChunkSize'].iloc[0])}

    safe = matrix.replace(".", "_")
    plot_heatmap(pivot, "Threads", "Chunk Size", f"{matrix} - Efficiency Heatmap ({schedule.upper()})",
                 output_dir / f"{safe}_efficiency_{schedule}.png", cmap='RdYlGn', vmin=0, vmax=100, highlight=highlight)
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
            for chunk in df_m[df_m['Schedule']==sched]['ChunkSize'].unique():
                df_g = df_m[(df_m['Schedule']==sched)&(df_m['ChunkSize']==chunk)&(df_m['Mode']=='parallel')].sort_values('Threads')
                plot_line(df_g["Threads"], df_g["Avg_IPC"], f"{sched.upper()} (Chunk={chunk})", SCHEDULE_COLORS[sched], ax=ax)

        seq_ipc = df_m[df_m["Mode"]=="sequential"]["Avg_IPC"].iloc[0]
        ax.scatter([1],[seq_ipc],color="black",marker="D",s=100,label="Sequential")
        safe = matrix.replace(".", "_")
        finalize_plot(ax,"Number of Threads","Avg IPC",f"{matrix} - IPC vs Threads",output_dir / f"{safe}_IPC_vs_threads.png")
        print(f"✓ IPC plot generated for {matrix}")


def plot_cache_miss_rates(cache_miss_df, output_dirs):
    matrices = cache_miss_df["Matrix"].unique()
    for matrix in matrices:
        df_m = cache_miss_df[cache_miss_df["Matrix"] == matrix]
        cache_dir = output_dirs[matrix]["4_cache_misses"]
        df_par = df_m[df_m["Mode"] == "parallel"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, miss_type in zip(axes, ["L1_MissRate","LLC_MissRate"]):
            for sched in ORDER_SCHEDULES:
                for chunk in df_par[df_par['Schedule']==sched]['ChunkSize'].unique():
                    df_g = df_par[(df_par['Schedule']==sched)&(df_par['ChunkSize']==chunk)].sort_values('Threads')
                    plot_line(df_g["Threads"], df_g[miss_type], f"{sched.upper()} (Chunk={chunk})", SCHEDULE_COLORS[sched], ax=ax)
            seq_val = df_m[df_m["Mode"]=="sequential"][miss_type].iloc[0]
            ax.scatter([1],[seq_val], color="black", marker="D", s=80, label="Sequential")
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
# Main plotting pipeline
# -------------------------------------------------------------------

def run_analysis():
    timings_df = pd.read_csv("results/benchmark_timings.csv")
    timings_df['Schedule'] = timings_df['Schedule'].fillna('none')
    timings_df['ChunkSize'] = timings_df['ChunkSize'].fillna(0).astype(int)

    matrices = timings_df['Matrix'].unique()
    dirs, results_dir = setup_dirs(matrices)

    speedup_df = pd.read_csv(results_dir / "speedup_efficiency.csv")
    best_configs = pd.read_csv(results_dir / "best_configs.csv")

    for matrix in matrices:
        df_matrix_speedup = speedup_df[speedup_df['Matrix']==matrix]
        for schedule in ORDER_SCHEDULES:
            plot_heatmap_speedup(df_matrix_speedup, schedule, dirs[matrix]["0_tuning"], best_configs)
            plot_heatmap_efficiency(df_matrix_speedup, schedule, dirs[matrix]["1_efficiency"], best_configs)

    best_curves_df = speedup_df.merge(best_configs, on=['Matrix','Schedule','ChunkSize'])
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

if __name__=="__main__":
    run_analysis()
