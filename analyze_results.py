import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

def main():
    input_file = os.path.join(PROJECT_ROOT, 'results', 'tables', 'experiment_results.csv')
    figures_dir = os.path.join(PROJECT_ROOT, 'results', 'figures')
    tables_dir = os.path.join(PROJECT_ROOT, 'results', 'tables')

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}.\nPlease run 'python run_experiments.py' first to generate the data.")
        return

    df = pd.read_csv(input_file)

    col_ghanbari = df['R_ghanbari']
    col_proposed = df['R_proposed']


    absolute_error = abs(col_ghanbari - col_proposed)
    valid_mask = col_ghanbari != 0
    relative_error = pd.Series(np.nan, index=df.index)
    relative_error[valid_mask] = absolute_error[valid_mask] / abs(col_ghanbari[valid_mask])

    valid_abs = absolute_error.replace([np.inf, -np.inf], np.nan).dropna()
    valid_rel = relative_error.replace([np.inf, -np.inf], np.nan).dropna()

    mean_abs = valid_abs.mean()
    median_abs = valid_abs.median()
    std_abs = valid_abs.std()
    max_abs = valid_abs.max()
    min_abs = valid_abs.min()
    q1_abs = valid_abs.quantile(0.25)
    q3_abs = valid_abs.quantile(0.75)
    count_abs = len(valid_abs)

    se_abs = std_abs / np.sqrt(count_abs)
    ci_lower_abs, ci_upper_abs = stats.t.interval(0.95, count_abs - 1, loc=mean_abs, scale=se_abs)


    mean_rel = valid_rel.mean()
    median_rel = valid_rel.median()
    std_rel = valid_rel.std()
    max_rel = valid_rel.max()
    min_rel = valid_rel.min()
    q1_rel = valid_rel.quantile(0.25)
    q3_rel = valid_rel.quantile(0.75)
    count_rel = len(valid_rel)

    se_rel = std_rel / np.sqrt(count_rel)
    ci_lower_rel, ci_upper_rel = stats.t.interval(0.95, count_rel - 1, loc=mean_rel, scale=se_rel)


    print("\n" + "=" * 60)
    print("Absolute Error Statistics |r_G - r_C|")
    print("=" * 60)
    print(f"Count: {count_abs}")
    print(f"Mean:  {mean_abs:.10f}")
    print(f"Median:{median_abs:.10f}")
    print(f"Std:   {std_abs:.10f}")
    print(f"Max:   {max_abs:.10f}")
    print(f"95% CI:[{ci_lower_abs:.10f}, {ci_upper_abs:.10f}]")

    print("\n" + "=" * 60)
    print("Relative Error Statistics |r_G - r_C| / |r_G|")
    print("=" * 60)
    print(f"Count: {count_rel}")
    print(f"Mean:  {mean_rel:.10f}")
    print(f"Median:{median_rel:.10f}")
    print(f"Std:   {std_rel:.10f}")
    print(f"Max:   {max_rel:.10f}")
    print(f"95% CI:[{ci_lower_rel:.10f}, {ci_upper_rel:.10f}]")
    print("=" * 60)


    stats_abs_df = pd.DataFrame({
        'Statistic': ['Count', 'Mean', 'Median', 'Std', 'Min', 'Max', 'Q1', 'Q3', 'CI_95_Lower', 'CI_95_Upper'],
        'Value': [count_abs, mean_abs, median_abs, std_abs, min_abs, max_abs, q1_abs, q3_abs, ci_lower_abs, ci_upper_abs]
    })
    stats_abs_df.to_csv(os.path.join(tables_dir, 'abs_error_stats.csv'), index=False)

    stats_rel_df = pd.DataFrame({
        'Statistic': ['Count', 'Mean', 'Median', 'Std', 'Min', 'Max', 'Q1', 'Q3', 'CI_95_Lower', 'CI_95_Upper'],
        'Value': [count_rel, mean_rel, median_rel, std_rel, min_rel, max_rel, q1_rel, q3_rel, ci_lower_rel, ci_upper_rel]
    })
    stats_rel_df.to_csv(os.path.join(tables_dir, 'rel_error_stats.csv'), index=False)


    plt.figure(figsize=(10, 6))
    plt.hist(valid_abs, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Absolute Error |r_C - r^_C|')
    plt.ylabel('Frequency')
    plt.title('Histogram of Absolute Errors')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(figures_dir, 'absolute_error_histogram.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(valid_rel, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Relative Error |r_C- r^_C| / |r_C|')
    plt.ylabel('Frequency')
    plt.title('Histogram of Relative Errors')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(figures_dir, 'relative_error_histogram.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    sorted_rel = np.sort(valid_rel)
    y_ecdf_rel = np.arange(1, len(sorted_rel) + 1) / len(sorted_rel)
    plt.step(sorted_rel, y_ecdf_rel, where='post')
    plt.xlabel('Relative Error |r_G - r_C| / |r_G|')
    plt.ylabel('Cumulative Probability')
    plt.title('ECDF of Relative Errors')
    plt.grid(True, linestyle='--', alpha=0.5)

    for thresh in [1e-4, 1e-3, 1e-2]:
        pct = (valid_rel < thresh).mean() * 100
        plt.axvline(x=thresh, color='red', linestyle='--', alpha=0.5)
        plt.text(thresh, 0.5, f'{pct:.1f}% < {thresh:.0e}', rotation=90, fontsize=9)

    plt.savefig(os.path.join(figures_dir, 'relative_error_ecdf.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nAnalysis complete. Figures saved to {figures_dir}, tables saved to {tables_dir}.")

if __name__ == "__main__":
    main()