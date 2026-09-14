import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from Ghanbari_method import calculate_R
from Proposed_method import simpsons_rule


def main():
    parser = argparse.ArgumentParser(description="Reproduce experiments for LR Fuzzy Numbers paper.")
    parser.add_argument('--seed', type=int, default=12, help='Random seed for reproducibility')
    parser.add_argument('--num_pairs', type=int, default=10000, help='Number of fuzzy number pairs')
    parser.add_argument('--mu_min', type=float, default=-5000, help='Minimum value for mu')
    parser.add_argument('--mu_max', type=float, default=5000, help='Maximum value for mu')
    parser.add_argument('--sigma_min', type=float, default=1, help='Minimum value for sigma')
    parser.add_argument('--sigma_max', type=float, default=500, help='Maximum value for sigma')

    args = parser.parse_args()

    print(f"Starting experiments with seed={args.seed}, num_pairs={args.num_pairs}...")

    np.random.seed(args.seed)
    mu1 = np.random.uniform(args.mu_min, args.mu_max, args.num_pairs)
    sigma1 = np.random.uniform(args.sigma_min, args.sigma_max, args.num_pairs)
    mu2 = np.random.uniform(args.mu_min, args.mu_max, args.num_pairs)
    sigma2 = np.random.uniform(args.sigma_min, args.sigma_max, args.num_pairs)

    print("Running Ghanbari et al. method (Reference)...")
    R_ghanbari = []
    start_time_ghanbari = time.time()
    for i in range(args.num_pairs):
        R = calculate_R((mu1[i], sigma1[i]), (mu2[i], sigma2[i]))
        R_ghanbari.append(R)
    end_time_ghanbari = time.time()
    time_ghanbari = end_time_ghanbari - start_time_ghanbari
    print(f"Ghanbari method execution time: {time_ghanbari:.2f} seconds")

    print("Running Proposed method (Simpson-based)...")
    R_proposed = []
    start_time_proposed = time.time()
    for i in range(args.num_pairs):
        R = simpsons_rule((mu1[i], sigma1[i]), (mu2[i], sigma2[i]))
        R_proposed.append(R)
    end_time_proposed = time.time()
    time_proposed = end_time_proposed - start_time_proposed
    print(f"Proposed method execution time: {time_proposed:.2f} seconds")

    results_df = pd.DataFrame({
        'mu1': mu1, 'sigma1': sigma1,
        'mu2': mu2, 'sigma2': sigma2,
        'R_ghanbari': R_ghanbari,
        'R_proposed': R_proposed
    })

    valid_mask = results_df['R_ghanbari'] != 0
    results_df.loc[valid_mask, 'relative_error'] = np.abs(
        results_df.loc[valid_mask, 'R_proposed'] - results_df.loc[valid_mask, 'R_ghanbari']
    ) / np.abs(results_df.loc[valid_mask, 'R_ghanbari'])
    results_df.loc[~valid_mask, 'relative_error'] = 0.0

    tables_dir = os.path.join(PROJECT_ROOT, 'results', 'tables')
    os.makedirs(tables_dir, exist_ok=True)
    output_path = os.path.join(tables_dir, 'experiment_results.csv')
    results_df.to_csv(output_path, index=False)

    print(f"\nResults saved to {output_path}")
    print("\n--- Summary Statistics ---")
    print(f"Max relative error: {results_df['relative_error'].max():.6f}")
    print(f"Mean relative error: {results_df['relative_error'].mean():.6f}")
    print(f"Execution time Ghanbari: {time_ghanbari:.2f} s")
    print(f"Execution time Proposed: {time_proposed:.2f} s")


if __name__ == "__main__":
    main()