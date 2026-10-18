import os
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def main():
    input_file = os.path.join(PROJECT_ROOT, 'results', 'tables', 'experiment_results.csv')
    output_file = os.path.join(PROJECT_ROOT, 'results', 'tables', 'sign_mismatches.csv')

    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}.\nPlease run 'python run_experiments.py' first.")
        return

    df = pd.read_csv(input_file)
    col_ghanbari = df['R_ghanbari']
    col_proposed = df['R_proposed']

    sign_ghanbari = np.sign(col_ghanbari)
    sign_proposed = np.sign(col_proposed)

    mismatches_mask = (sign_ghanbari != sign_proposed)
    mismatches = df[mismatches_mask].copy()

    total_pairs = len(df)
    disagreements = len(mismatches)
    agreements = total_pairs - disagreements
    agreement_rate = (agreements / total_pairs) * 100

    print(f"Total Pairs: {total_pairs}")
    print(f"Agreement Rate: {agreement_rate:.2f}%")

    if disagreements > 0:
        mismatches.to_csv(output_file, index=False)
        print(f"Found {disagreements} mismatches. Saved to {output_file}.")
    else:
        pd.DataFrame({'Message': ['No mismatches found.']}).to_csv(output_file, index=False)
        print("No mismatches found. All signs are consistent (including zeros).")


if __name__ == "__main__":
    main()
