import os
import sys
import random
import argparse
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from fuzzy_utils import fast_newton_xbar
from x_bar_reference import find_correct_intersection, Spline_Fitting


def generate_overlapping_gaussian_pair():
    """
    Generate a pair of Gaussian fuzzy numbers that overlap.
    """
    max_attempts = 1000
    for _ in range(max_attempts):
        mu1 = random.randint(-5000, 5000)
        sigma1 = random.randint(1, 500)
        mu2 = random.randint(-5000, 5000)
        sigma2 = random.randint(1, 500)

        left1 = mu1 - 3 * sigma1
        right1 = mu1 + 3 * sigma1
        left2 = mu2 - 3 * sigma2
        right2 = mu2 + 3 * sigma2

        if not (right1 < left2 or right2 < left1):
            return (mu1, sigma1, mu2, sigma2)

    return (0, 100, 0, 100)


def main():
    parser = argparse.ArgumentParser(description="Compare Newton-Raphson and Brent's method for intersection point (Table 1).")
    parser.add_argument('--num_pairs', type=int, default=10000, help='Number of overlapping pairs')
    parser.add_argument('--seed', type=int, default=12, help='Random seed')
    args = parser.parse_args()

    random.seed(args.seed)

    # Generate overlapping pairs
    print(f"Generating {args.num_pairs} overlapping Gaussian pairs (seed={args.seed})...")
    pairs = [generate_overlapping_gaussian_pair() for _ in range(args.num_pairs)]

    x_bar_list = []
    x_real_list = []

    print("Computing intersection points (Newton vs Brent)...")
    for index, (mu1, sigma1, mu2, sigma2) in enumerate(pairs):
        if mu1 > mu2:
            mu1, sigma1, mu2, sigma2 = mu2, sigma2, mu1, sigma1

        data1 = Spline_Fitting(mu1, sigma1)
        data2 = Spline_Fitting(mu2, sigma2)

        f1 = data1["mu_func"]
        f2 = data2["mu_func"]

        s1_min, s1_max = data1["support_min"], data1["support_max"]
        s2_min, s2_max = data2["support_min"], data2["support_max"]

        # Newton-Raphson (1 step)
        x_bar = fast_newton_xbar(data1, data2, x0=None)

        # Brent reference
        common_min = min(s1_min, s2_min)
        common_max = max(s1_max, s2_max)

        x_real = find_correct_intersection(f1, f2, mu1, mu2, common_min, common_max)

        x_bar_list.append(x_bar)
        x_real_list.append(x_real if x_real is not None else np.nan)

        if (index + 1) % 1000 == 0:
            print(f"  Processed {index + 1} pairs...")

    # Build DataFrame
    results_df = pd.DataFrame({
        'x_bar_newton': x_bar_list,
        'x_real_brent': x_real_list,
        'difference': [abs(x_bar_list[i] - x_real_list[i]) if not np.isnan(x_real_list[i]) else np.nan
                       for i in range(len(x_bar_list))]
    })

    # Save to results/tables
    tables_dir = os.path.join(PROJECT_ROOT, 'results', 'tables')
    os.makedirs(tables_dir, exist_ok=True)
    output_path = os.path.join(tables_dir, 'x_bar_comparison.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # ===== Statistics =====
    valid_diffs = results_df['difference'].dropna()

    mean_val = valid_diffs.mean()
    median_val = valid_diffs.median()
    variance_val = valid_diffs.var()
    max_val = valid_diffs.max()

    try:
        mode_series = valid_diffs.mode()
        mode_val = mode_series.iloc[0] if len(mode_series) > 0 else np.nan
    except Exception:
        mode_val = np.nan

    print("\n" + "=" * 60)
    print("Statistics of absolute errors |x_bar - x_real| (Newton vs Brent)")
    print("=" * 60)
    print(f"Number of valid data: {len(valid_diffs)}")
    print(f"Mean:                {mean_val:.10e}")
    print(f"Median:              {median_val:.10e}")
    print(f"Variance:            {variance_val:.10e}")
    print(f"Max:                 {max_val:.10e}")
    if not np.isnan(mode_val):
        print(f"Mode:                {mode_val:.10e}")
    else:
        print("Mode:                undefined")
    print("=" * 60)

    # ===== Percentage below thresholds =====
    print("\n--- Percentage of errors below thresholds ---")
    for thresh in [1e-4, 1e-5, 1e-6]:
        count = (valid_diffs < thresh).sum()
        pct = (count / len(valid_diffs)) * 100
        print(f"Threshold {thresh:.0e}: {count} errors ({pct:.2f}%)")


if __name__ == "__main__":
    main()