import pandas as pd
from pathlib import Path

# -------------------------------------------------------------------
# Tuning functions
# -------------------------------------------------------------------

"""Compute 90th percentile for time and GFLOPs."""
def compute_p90(df):
    group_cols = ['Mode', 'Matrix', 'Threads', 'Schedule', 'ChunkSize']
    p90_time = df.groupby(group_cols, sort=False)['Time_ms'].quantile(0.9).reset_index()
    p90_gflops = df.groupby(group_cols, sort=False)['GFLOPs'].quantile(0.9).reset_index()
    p90_time.rename(columns={'Time_ms': 'p90_Time_ms'}, inplace=True)
    p90_gflops.rename(columns={'GFLOPs': 'p90_GFLOPs'}, inplace=True)
    p90_df = pd.merge(p90_time, p90_gflops, on=group_cols, how='left')
    print("✓ Computed 90th percentile metrics")
    return p90_df

"""Compute speedup and efficiency."""
def compute_performance(p90_df):
    seq = p90_df[p90_df['Mode'] == 'sequential']
    par = p90_df[p90_df['Mode'] == 'parallel']
    data = []

    for matrix in par['Matrix'].unique():
        t_seq = seq[seq['Matrix'] == matrix]['p90_Time_ms'].values[0]
        df_m = par[par['Matrix'] == matrix]
        for _, row in df_m.iterrows():
            speedup = t_seq / row['p90_Time_ms']
            efficiency = (speedup / row['Threads']) * 100
            data.append({
                'Matrix': matrix,
                'Schedule': row['Schedule'],
                'Threads': row['Threads'],
                'ChunkSize': row['ChunkSize'],
                'p90_Time_ms': row['p90_Time_ms'],
                'p90_GFLOPs': row['p90_GFLOPs'],
                'Speedup': speedup,
                'Efficiency': efficiency
            })

    df = pd.DataFrame(data)
    print("✓ Performance metrics computed (Speedup & Efficiency)")
    return df


"""Find best chunk size per matrix and schedule based on median speedup."""
def find_best_configurations(speedup_df, results_dir):
    median_speedups = speedup_df.groupby(
        ['Matrix', 'Schedule', 'ChunkSize'], sort=False
    )['Speedup'].median().reset_index()

    idx = median_speedups.groupby(['Matrix', 'Schedule'], sort=False)['Speedup'].idxmax()

    best_configs = median_speedups.loc[idx][['Matrix', 'Schedule', 'ChunkSize']]
    best_configs.to_csv(results_dir / "best_configs.csv", index=False)

    print("✓ Best configurations saved to CSV (original order preserved)")
    return best_configs


# -------------------------------------------------------------------
# Main tuning pipeline
# -------------------------------------------------------------------

def run_tuning():
    results_dir = Path("results"); results_dir.mkdir(exist_ok=True)

    timings_df = pd.read_csv("results/benchmark_timings.csv")
    timings_df['Schedule'] = timings_df['Schedule'].fillna('none')
    timings_df['ChunkSize'] = timings_df['ChunkSize'].fillna(0).astype(int)

    p90_df = compute_p90(timings_df)
    speedup_df = compute_performance(p90_df)

    speedup_df.to_csv(results_dir / "speedup_efficiency.csv", index=False)

    find_best_configurations(speedup_df, results_dir)

    print("✓ Tuning completed")


if __name__ == "__main__":
    run_tuning()
