import os
import pandas as pd

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

    mismatches = []
    for index, (val_ghanbari, val_proposed) in enumerate(zip(col_ghanbari, col_proposed)):
        if (val_ghanbari > 0 and val_proposed < 0) or (val_ghanbari < 0 and val_proposed > 0):
            mismatches.append({'Row_Index': index, 'R_ghanbari': val_ghanbari, 'R_proposed': val_proposed})

    if mismatches:
        pd.DataFrame(mismatches).to_csv(output_file, index=False)
        print(f"Found {len(mismatches)} mismatches. Saved to {output_file}.")
    else:
        pd.DataFrame({'Message': ['No mismatches found.']}).to_csv(output_file, index=False)
        print("No mismatches found. All signs are consistent.")

if __name__ == "__main__":
    main()