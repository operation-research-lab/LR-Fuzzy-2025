import os
import sys
import time
import argparse
import random
import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR

from Calculate_rG import r_gaussian
from Ghanbari_method import calculate_R
from Proposed_method import simpsons_rule


def generate_random_gaussian_fuzzy():
    """Generate a single random Gaussian fuzzy number (mu, sigma)."""
    mu = random.randint(-5000, 5000)
    sigma = random.randint(1, 500)
    return (mu, sigma)


def main():
    parser = argparse.ArgumentParser(description="Error decomposition for Table 4.")
    parser.add_argument('--num_pairs', type=int, default=10000, help='Number of pairs')
    parser.add_argument('--seed', type=int, default=12, help='Random seed')
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Generating {args.num_pairs} Gaussian pairs (seed={args.seed})...")
    all_numbers = [generate_random_gaussian_fuzzy() for _ in range(2 * args.num_pairs)]
    data1 = all_numbers[:args.num_pairs]
    data2 = all_numbers[args.num_pairs:]

    # ============================================
    # 1. Compute r_G (Gaussian reference)
    # ============================================
    print("Computing r_G (Gaussian reference)...")
    start = time.time()
    r_G_list = []
    for i in range(args.num_pairs):
        mu1, sigma1 = data1[i]
        mu2, sigma2 = data2[i]
        rG = r_gaussian(mu1, sigma1, mu2, sigma2)
        r_G_list.append(rG)
    print(f"  Done in {time.time() - start:.2f} seconds")

    # ============================================
    # 2. Compute r_C (cubic LR, integral-based)
    # ============================================
    print("Computing r_C (cubic LR, integral-based)...")
    start = time.time()
    r_C_list = []
    for i in range(args.num_pairs):
        rC = calculate_R(data1[i], data2[i])
        r_C_list.append(rC)
    print(f"  Done in {time.time() - start:.2f} seconds")

    # ============================================
    # 3. Compute r_hat_C (cubic LR, Simpson-based)
    # ============================================
    print("Computing r_hat_C (Simpson-based)...")
    start = time.time()
    r_hat_list = []
    for i in range(args.num_pairs):
        rH = simpsons_rule(data1[i], data2[i])
        r_hat_list.append(rH)
    print(f"  Done in {time.time() - start:.2f} seconds")

    # ============================================
    # 4. Compute the three error components
    # ============================================
    r_G_arr = np.array(r_G_list, dtype=float)
    r_C_arr = np.array(r_C_list, dtype=float)
    r_hat_arr = np.array(r_hat_list, dtype=float)

    # Absolute errors
    E_rep_abs = np.abs(r_G_arr - r_C_arr)
    e_S_abs = np.abs(r_C_arr - r_hat_arr)
    E_tot_abs = np.abs(r_G_arr - r_hat_arr)

    # Relative errors (avoid division by zero)
    valid_G = r_G_arr != 0
    valid_C = r_C_arr != 0

    E_rep_rel = np.full_like(E_rep_abs, np.nan)
    E_rep_rel[valid_G] = E_rep_abs[valid_G] / np.abs(r_G_arr[valid_G])

    e_S_rel = np.full_like(e_S_abs, np.nan)
    e_S_rel[valid_C] = e_S_abs[valid_C] / np.abs(r_C_arr[valid_C])

    E_tot_rel = np.full_like(E_tot_abs, np.nan)
    E_tot_rel[valid_G] = E_tot_abs[valid_G] / np.abs(r_G_arr[valid_G])

    # Clean NaNs/Infs
    E_rep_abs = E_rep_abs[np.isfinite(E_rep_abs)]
    e_S_abs = e_S_abs[np.isfinite(e_S_abs)]
    E_tot_abs = E_tot_abs[np.isfinite(E_tot_abs)]

    E_rep_rel = E_rep_rel[np.isfinite(E_rep_rel)]
    e_S_rel = e_S_rel[np.isfinite(e_S_rel)]
    E_tot_rel = E_tot_rel[np.isfinite(E_tot_rel)]

    # ============================================
    # 5. Compute statistics for each component
    # ============================================
    def stats_row(abs_arr, rel_arr, name):
        mean_abs = np.mean(abs_arr)
        max_abs = np.max(abs_arr)
        mean_rel = np.mean(rel_arr)
        max_rel = np.max(rel_arr)
        std_rel = np.std(rel_arr, ddof=1)
        n = len(rel_arr)
        se = std_rel / np.sqrt(n)
        ci_lo, ci_hi = stats.t.interval(0.95, n - 1, loc=mean_rel, scale=se)
        return {
            'Error_Component': name,
            'Mean_Abs': mean_abs,
            'Max_Abs': max_abs,
            'Mean_Rel': mean_rel,
            'Max_Rel': max_rel,
            'Std_Rel': std_rel,
            'CI_95_Lower': ci_lo,
            'CI_95_Upper': ci_hi,
        }

    rows = [
        stats_row(E_rep_abs, E_rep_rel, 'Representation E_rep'),
        stats_row(e_S_abs,   e_S_rel,   'Simpson e_S'),
        stats_row(E_tot_abs, E_tot_rel, 'Total E_tot'),
    ]
    df = pd.DataFrame(rows)

    # Save to CSV
    output_path = os.path.join(PROJECT_ROOT, 'results', 'tables', 'error_decomposition.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    # ============================================
    # 6. Print results in a format matching Table 4
    # ============================================
    print("\n" + "=" * 110)
    print("Table 4: Empirical Decomposition of the Approximation Error")
    print("=" * 110)
    header = f"{'Error Component':<25} | {'Mean Abs':>10} | {'Max Abs':>10} | {'Mean Rel':>10} | {'Max Rel':>10} | {'Std. Rel':>10} | {'95% CI (Mean Rel)':>25}"
    print(header)
    print("-" * 110)
    for r in rows:
        ci_str = f"[{r['CI_95_Lower']:.6f}, {r['CI_95_Upper']:.6f}]"
        print(f"{r['Error_Component']:<25} | {r['Mean_Abs']:>10.4f} | {r['Max_Abs']:>10.4f} | "
              f"{r['Mean_Rel']:>10.6f} | {r['Max_Rel']:>10.6f} | {r['Std_Rel']:>10.6f} | {ci_str:>25}")
    print("=" * 110)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()