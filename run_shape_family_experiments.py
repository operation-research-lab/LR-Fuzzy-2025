import os
import sys
import time
import argparse
import random
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

import shape_family_linear, shape_family_Quintic, shape_family_cubic, shape_family_quadratic
def generate_random_gaussian_fuzzy():
    """
    Generate a single random Gaussian fuzzy number (a, b, c).
    Exactly matching the original data generation code.
    """
    mu = random.randint(-5000, 5000)
    sigma = random.randint(1, 500)

    a = mu - 3 * sigma
    b = mu
    c = mu + 3 * sigma

    return (a, b, c)


def generate_dataset(num_pairs, seed):
    """
    Generate two datasets of fuzzy numbers exactly as in the original code:
    - Set the seed ONCE.
    - Then generate 2 * num_pairs numbers consecutively.
    - First num_pairs go to dataset1, next num_pairs go to dataset2.
    """
    random.seed(seed)

    all_numbers = [generate_random_gaussian_fuzzy() for _ in range(2 * num_pairs)]

    data1 = all_numbers[:num_pairs]
    data2 = all_numbers[num_pairs:]

    return data1, data2


def run_family_experiment(family_name, module, num_pairs=10000, seed=12):
    print(f"\n{'='*60}")
    print(f"--- Testing {family_name} Shape Family ---")
    print(f"{'='*60}")

    # Generate data exactly like the original code
    data1, data2 = generate_dataset(num_pairs, seed)

    # Ghanbari (Reference Integral-based Method)
    print("Running Ghanbari method (reference)...")
    start_g = time.time()
    R_ghanbari = [module.calculate_R(f1, f2) for f1, f2 in zip(data1, data2)]
    time_g = time.time() - start_g
    print(f"  Time: {time_g:.2f} seconds")

    # Proposed (Simpson-based Method)
    print("Running Proposed method (Simpson-based)...")
    start_s = time.time()
    R_simpson = [module.simpsons_rule(f1, f2) for f1, f2 in zip(data1, data2)]
    time_s = time.time() - start_s
    print(f"  Time: {time_s:.2f} seconds")

    # Calculate errors
    R_ghanbari = np.array(R_ghanbari)
    R_simpson = np.array(R_simpson)

    abs_error = np.abs(R_ghanbari - R_simpson)
    valid_mask = R_ghanbari != 0
    rel_error = np.zeros_like(abs_error)
    rel_error[valid_mask] = abs_error[valid_mask] / np.abs(R_ghanbari[valid_mask])

    return {
        'Shape_Family': family_name,
        'Max_Relative_Error': np.max(rel_error),
        'Mean_Relative_Error': np.mean(rel_error),
        'Max_Absolute_Error': np.max(abs_error),
        'Time_Proposed': time_s,
        'Time_Reference': time_g
    }


def main():
    parser = argparse.ArgumentParser(description="Run experiments for different LR shape families (Table 5).")
    parser.add_argument('--num_pairs', type=int, default=10000, help='Number of pairs per family')
    parser.add_argument('--seed', type=int, default=12, help='Random seed')
    args = parser.parse_args()

    results = []

    # Run experiments for each shape family
    results.append(run_family_experiment("Linear",    shape_family_linear,    args.num_pairs, args.seed))
    results.append(run_family_experiment("Quadratic", shape_family_quadratic, args.num_pairs, args.seed))
    results.append(run_family_experiment("Cubic",     shape_family_cubic,     args.num_pairs, args.seed))
    results.append(run_family_experiment("Quintic",   shape_family_Quintic,   args.num_pairs, args.seed))

    # Save results to CSV
    df = pd.DataFrame(results)
    output_path = os.path.join(PROJECT_ROOT, 'results', 'tables', 'shape_family_results.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print("\n" + "="*60)
    print("Final Results for Table 5 (Shape Families)")
    print("="*60)
    print(df.to_string(index=False))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()