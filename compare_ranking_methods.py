import os
import sys
import random
import argparse
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR

from Yager import compare_yager
from Liou_Wang import compare_liou_wang

from fuzzy_utils import (left_membership1, right_membership1, left_membership2,
                         right_membership2, cubic_Lagrange_interpolant, fast_newton_xbar)

def simpsons_rule(Gussi_fuzzy1, Gussi_fuzzy2):
    s = 1
    global R
    mu1, sigma1 = Gussi_fuzzy1
    mu2, sigma2 = Gussi_fuzzy2
    tolerance = 1e-5

    fuzzy_data1 = cubic_Lagrange_interpolant(mu1, sigma1)
    fuzzy_data2 = cubic_Lagrange_interpolant(mu2, sigma2)

    # Ensure fuzzy_data1 has the smaller core for consistent logic
    if fuzzy_data1["fuzzy_number"][1] > fuzzy_data2["fuzzy_number"][1]:
        fuzzy_data1, fuzzy_data2 = fuzzy_data2, fuzzy_data1
        s = -1
    a1, b1, c1 = fuzzy_data1["fuzzy_number"]
    a2, b2, c2 = fuzzy_data2["fuzzy_number"]

    R = 0

    if c1 <= a2:

        mid2 = (a1 + b1) / 2
        mid3 = (b1 + c1) / 2
        mid7 = (a2 + b2) / 2
        mid8 = (b2 + c2) / 2

        R = (2 / 3) * ((b1 - a1) * left_membership1(mid2, a1, b1, fuzzy_data1) + (c1 - b1) * right_membership1(mid3, b1, c1, fuzzy_data1)
                       + (b2 - a2) * left_membership2(mid7, a2, b2, fuzzy_data2) + (c2 - b2) * right_membership2(mid8, b2, c2, fuzzy_data2)) + (
                    1 / 6) * (-a1 + c1 - a2 + c2)

    ## case: 2
    elif b1 == b2:

        max_a = max(a1, a2)
        min_a = min(a1, a2)
        min_c = min(c1, c2)
        max_c = max(c1, c2)

        mid_1_1 = (max_a + min_a) / 2
        mid1 = (max_a + b1) / 2
        mid_1_5 = (min_c + b1) / 2
        mid_3_1 = (max_c + min_c) / 2

        #ML_mia_a = left_membership1(min_a, a1, b1, fuzzy_data1)
        #NL_min_a = left_membership2(min_a, a2, b2, fuzzy_data2)
        ML_mid_1_1 = left_membership1(mid_1_1, a1, b1, fuzzy_data1)
        NL_mid_1_1 = left_membership2(mid_1_1, a2, b2, fuzzy_data2)
        ML_max_a = left_membership1(max_a, a1, b1, fuzzy_data1)
        NL_max_a = left_membership2(max_a, a2, b2, fuzzy_data2)

        I_1_1 = ((max_a - min_a) / 6) * (4 * (ML_mid_1_1 - NL_mid_1_1) + (ML_max_a - NL_max_a))
######################################################################################################################
        ML_mid1 = left_membership1(mid1, a1, b1, fuzzy_data1)
        NL_mid1 = left_membership2(mid1, a2, b2, fuzzy_data2)
        I1 = ((b1 - max_a) / 6) * ((ML_max_a - NL_max_a) + 4 * (ML_mid1 - NL_mid1))
        ##################################################################################################################
        NR_mid_1_5 = right_membership2(mid_1_5, b2, c2, fuzzy_data2)
        MR_mid_1_5 = right_membership1(mid_1_5, b1, c1, fuzzy_data1)
        NR_min_c = right_membership2(min_c, b2, c2, fuzzy_data2)
        MR_min_c = right_membership1(min_c, b1, c1, fuzzy_data1)

        I2 = ((min_c - b1) / 6) * (4 * (NR_mid_1_5 - MR_mid_1_5) + (NR_min_c - MR_min_c))
    ##################################################################################################################
        #MR_max_c = right_membership1(max_c, b1, c1, fuzzy_data1)
        #NR_max_c = right_membership2(max_c, b2, c2, fuzzy_data2)
        MR_mid_3_1 = right_membership1(mid_3_1, b1, c1, fuzzy_data1)
        NR_mid_3_1 = right_membership2(mid_3_1, b2, c2, fuzzy_data2)

        I_4_1 = ((max_c - min_c) / 6) * ((NR_min_c - MR_min_c) + 4 * (NR_mid_3_1 - MR_mid_3_1))

        R = I1 + I2 + I_4_1 + I_1_1
        #print(R)
    ####################################################################################################################
    ## case:3
    elif b1 < b2 and a2 <= b1 and b2 <= c1:

        max_a = max(a1, a2)
        min_a = min(a1, a2)
        min_c = min(c1, c2)
        max_c = max(c1, c2)

        #x_bar = (b2 * c1 - b1 * a2) / ((b2 - a2) + (c1 - b1))
        #x_bar = (mu1 * sigma2 + mu2 * sigma1) / (sigma1 + sigma2)
        x_bar = fast_newton_xbar(fuzzy_data1, fuzzy_data2, x0=None)
        mid1 = (max_a + b1) / 2
        mid_1_1 = (max_a + min_a) / 2
        mid3 = (b2 + min_c) / 2
        mid_xbar1 = (b1 + x_bar) / 2
        mid_xbar2 = (b2 + x_bar) / 2
        mid_3_1 = (max_c + min_c) / 2
        ###################################################################################
        #ML_mia_a = left_membership1(min_a, a1, b1, fuzzy_data1)
        #NL_min_a = left_membership2(min_a, a2, b2, fuzzy_data2)

        ML_mid_1_1 = left_membership1(mid_1_1, a1, b1, fuzzy_data1)
        NL_mid_1_1 = left_membership2(mid_1_1, a2, b2, fuzzy_data2)

        ML_max_a = left_membership1(max_a, a1, b1, fuzzy_data1)
        NL_max_a = left_membership2(max_a, a2, b2, fuzzy_data2)

        I_1_1 = ((max_a - min_a) / 6) * (4 * (ML_mid_1_1 - NL_mid_1_1) + (ML_max_a - NL_max_a))
        #################################################################################################################
        ML_mid1 = left_membership1(mid1, a1, b1, fuzzy_data1)
        NL_mid1 = left_membership2(mid1, a2, b2, fuzzy_data2)

        NL_b1 = left_membership2(b1, a2, b2, fuzzy_data2)
        I1 = ((b1 - max_a) / 6) * ((ML_max_a - NL_max_a) + 4 * (ML_mid1 - NL_mid1) + (1 - NL_b1))
        ################################################################################################################
        MR_mid_xbar1 = right_membership1(mid_xbar1, b1, c1, fuzzy_data1)
        NL_mid_xbar1 = left_membership2(mid_xbar1, a2, b2, fuzzy_data2)
        #MR_x_bar = right_membership1(x_bar, b1, c1, fuzzy_data1)
        #NL_x_bar = left_membership2(x_bar, a2, b2, fuzzy_data2)
        I2 = ((x_bar - b1) / 6) * ((1 - NL_b1) + 4 * (MR_mid_xbar1 - NL_mid_xbar1))
        ####################################################################################################################
        MR_mid_xbar2 = right_membership1(mid_xbar2, b1, c1, fuzzy_data1)
        NL_mid_xbar2 = left_membership2(mid_xbar2, a2, b2, fuzzy_data2)
        MR_b2 = right_membership1(b2, b1, c1, fuzzy_data1)
        I3 = ((b2 - x_bar) / 6) * (4 * (NL_mid_xbar2 - MR_mid_xbar2) + (1 - MR_b2))
        ####################################################################################################################
        MR_mid3 = right_membership1(mid3, b1, c1, fuzzy_data1)
        NR_mid3 = right_membership2(mid3, b2, c2, fuzzy_data2)
        MR_b2 = right_membership1(b2, b1, c1, fuzzy_data1)
        MR_min_c = right_membership1(min_c, b1, c1, fuzzy_data1)
        NR_min_c = right_membership2(min_c, b2, c2, fuzzy_data2)

        I4 = ((min_c - b2) / 6) * ((1 - MR_b2) + 4 * (NR_mid3 - MR_mid3) + (NR_min_c - MR_min_c))
        #############################################################################
        #MR_max_c = right_membership1(max_c, b1, c1, fuzzy_data1)
        #NR_max_c = right_membership2(max_c, b2, c2, fuzzy_data2)
        MR_mid_3_1 = right_membership1(mid_3_1, b1, c1, fuzzy_data1)
        NR_mid_3_1 = right_membership2(mid_3_1, b2, c2, fuzzy_data2)

        I_4_1 = ((max_c - min_c) / 6) * ((NR_min_c - MR_min_c) + 4 * (NR_mid_3_1 - MR_mid_3_1))

        R = I1 + I2 + I3 + I4 + I_4_1 + I_1_1
    ####################################################################################################################
    ## case:5
    elif b1 < b2 and b2 <= c1 and a2 >= b1:
        max_a = max(a1, a2)
        min_a = min(a1, a2)
        min_c = min(c1, c2)
        max_c = max(c1, c2)

        #x_bar = (b2 * c1 - b1 * a2) / ((b2 - a2) + (c1 - b1))
        #x_bar = (mu1 * sigma2 + mu2 * sigma1) / (sigma1 + sigma2)
        x_bar = fast_newton_xbar(fuzzy_data1, fuzzy_data2, x0=None)
        mid1_1 = (min_a + b1) / 2
        mid_1_2 = (max_a + b1) / 2
        mid3 = (b2 + min_c) / 2
        mid_xbar1_1 = (max_a + x_bar) / 2
        mid_xbar2 = (b2 + x_bar) / 2
        mid_3_1 = (max_c + min_c) / 2
        ##11111#################################################################################
        ML_mid1_1 = left_membership1(mid1_1, a1, b1, fuzzy_data1)
        I_1_1 = ((b1 - min_a) / 6) * (4 * ML_mid1_1 + 1)
        ####2222222#############################################################################################################
        MR_mid_1_2 = right_membership1(mid_1_2, b1, c1, fuzzy_data1)
        #NL_mid_1_2 = left_membership2(mid_1_2, a2, b2, fuzzy_data2)
        MR_max_a = right_membership1(max_a, b1, c1, fuzzy_data1)
        #NL_b1 = left_membership2(b1, a2, b2, fuzzy_data2)
        I1 = ((max_a - b1) / 6) * (1 + 4 * MR_mid_1_2 + MR_max_a)
        ################################################################################################################
        MR_mid_xbar1_1 = right_membership1(mid_xbar1_1, b1, c1, fuzzy_data1)
        NL_mid_xbar1_1 = left_membership2(mid_xbar1_1, a2, b2, fuzzy_data2)
        #MR_x_bar = right_membership1(x_bar, b1, c1, fuzzy_data1)
        #NL_x_bar = left_membership2(x_bar, a2, b2, fuzzy_data2)
        I2 = ((x_bar - max_a) / 6) * (MR_max_a + 4 * (MR_mid_xbar1_1 - NL_mid_xbar1_1))
        ####################################################################################################################
        MR_mid_xbar2 = right_membership1(mid_xbar2, b1, c1, fuzzy_data1)
        NL_mid_xbar2 = left_membership2(mid_xbar2, a2, b2, fuzzy_data2)
        MR_b2 = right_membership1(b2, b1, c1, fuzzy_data1)
        I3 = ((b2 - x_bar) / 6) * (4 * (NL_mid_xbar2 - MR_mid_xbar2) + (1 - MR_b2))
        ####################################################################################################################
        MR_mid3 = right_membership1(mid3, b1, c1, fuzzy_data1)
        NR_mid3 = right_membership2(mid3, b2, c2, fuzzy_data2)
        MR_b2 = right_membership1(b2, b1, c1, fuzzy_data1)
        MR_min_c = right_membership1(min_c, b1, c1, fuzzy_data1)
        NR_min_c = right_membership2(min_c, b2, c2, fuzzy_data2)

        I4 = ((min_c - b2) / 6) * ((1 - MR_b2) + 4 * (NR_mid3 - MR_mid3) + (NR_min_c - MR_min_c))
        #############################################################################
        #MR_max_c = right_membership1(max_c, b1, c1, fuzzy_data1)
        #NR_max_c = right_membership2(max_c, b2, c2, fuzzy_data2)
        MR_mid_3_1 = right_membership1(mid_3_1, b1, c1, fuzzy_data1)
        NR_mid_3_1 = right_membership2(mid_3_1, b2, c2, fuzzy_data2)

        I_4_1 = ((max_c - min_c) / 6) * ((NR_min_c - MR_min_c) + 4 * (NR_mid_3_1 - MR_mid_3_1))

        R = I1 + I2 + I3 + I4 + I_4_1 + I_1_1
    ####################################################################################################################
    ## case: 4
    elif b1 < b2 and c1 <= b2 and a2 <= b1:

        max_a = max(a1, a2)
        min_a = min(a1, a2)
        min_c = min(c1, c2)
        max_c = max(c1, c2)

        #x_bar = (b2 * c1 - b1 * a2) / ((b2 - a2) + (c1 - b1))
        #x_bar = (mu1 * sigma2 + mu2 * sigma1) / (sigma1 + sigma2)
        x_bar = fast_newton_xbar(fuzzy_data1, fuzzy_data2, x0=None)
        mid1 = (max_a + b1) / 2
        mid_1_1 = (max_a + min_a) / 2
        mid3 = (b2 + min_c) / 2
        mid_xbar1 = (b1 + x_bar) / 2
        mid_xbar2_2 = (min_c + x_bar) / 2
        mid_3_2 = (max_c + b2) / 2

        ###################################################################################
        #ML_mia_a = left_membership1(min_a, a1, b1, fuzzy_data1)
        #NL_min_a = left_membership2(min_a, a2, b2, fuzzy_data2)

        ML_mid_1_1 = left_membership1(mid_1_1, a1, b1, fuzzy_data1)
        NL_mid_1_1 = left_membership2(mid_1_1, a2, b2, fuzzy_data2)

        ML_max_a = left_membership1(max_a, a1, b1, fuzzy_data1)
        NL_max_a = left_membership2(max_a, a2, b2, fuzzy_data2)

        I_1_1 = ((max_a - min_a) / 6) * (4 * (ML_mid_1_1 - NL_mid_1_1) + (ML_max_a - NL_max_a))
        #################################################################################################################
        ML_mid1 = left_membership1(mid1, a1, b1, fuzzy_data1)
        NL_mid1 = left_membership2(mid1, a2, b2, fuzzy_data2)
        NL_b1 = left_membership2(b1, a2, b2, fuzzy_data2)
        I1 = ((b1 - max_a) / 6) * ((ML_max_a - NL_max_a) + 4 * (ML_mid1 - NL_mid1) + (1 - NL_b1))
        ################################################################################################################
        MR_mid_xbar1 = right_membership1(mid_xbar1, b1, c1, fuzzy_data1)
        NL_mid_xbar1 = left_membership2(mid_xbar1, a2, b2, fuzzy_data2)
        #MR_x_bar = right_membership1(x_bar, b1, c1, fuzzy_data1)
        #NL_x_bar = left_membership2(x_bar, a2, b2, fuzzy_data2)
        I2 = ((x_bar - b1) / 6) * ((1 - NL_b1) + 4 * (MR_mid_xbar1 - NL_mid_xbar1))
        ####################################################################################################################
        MR_mid_xbar2_2 = right_membership1(mid_xbar2_2, b1, c1, fuzzy_data1)
        NL_mid_xbar2_2 = left_membership2(mid_xbar2_2, a2, b2, fuzzy_data2)
        NL_c1 = left_membership2(c1, a2, b2, fuzzy_data2)
        I3 = ((min_c - x_bar) / 6) * (4 * (NL_mid_xbar2_2 - MR_mid_xbar2_2) + NL_c1)
        ####################################################################################################################
        NL_min_c = left_membership2(min_c, a2, b2, fuzzy_data2)
        NL_mid3 = left_membership2(mid3, a2, b2, fuzzy_data2)

        I4 = ((b2 - min_c) / 6) * (NL_min_c + 4 * NL_mid3 + 1)
        #############################################################################

        NR_mid_3_2 = right_membership2(mid_3_2, b2, c2, fuzzy_data2)
        I_4_1 = ((max_c - b2) / 6) * (1 + 4 * NR_mid_3_2)
        ###############################################################################
        R = I1 + I2 + I3 + I4 + I_4_1 + I_1_1
    ####################################################################################################################
    ## case: 6
    else:
        max_a = max(a1, a2)
        min_a = min(a1, a2)
        min_c = min(c1, c2)
        max_c = max(c1, c2)

        #x_bar = (b2 * c1 - b1 * a2) / ((b2 - a2) + (c1 - b1))
        #x_bar = (mu1 * sigma2 + mu2 * sigma1) / (sigma1 + sigma2)
        x_bar = fast_newton_xbar(fuzzy_data1, fuzzy_data2, x0=None)
        mid1 = (max_a + b1) / 2
        mid_2_1 = (min_a + b1) / 2
        mid3 = (b2 + min_c) / 2
        mid_xbar2_2 = (min_c + x_bar) / 2
        mid_xbar2_3 = (max_a + x_bar) / 2
        mid_3_2 = (max_c + b2) / 2
        ###################################################################################
        ML_mid_2_1 = left_membership1(mid_2_1, a1, b1, fuzzy_data1)
        I_1_1 = ((b1 - a1) / 6) * (4 * ML_mid_2_1 + 1)
        #################################################################################################################
        MR_max_a = right_membership1(max_a, b1, c1, fuzzy_data1)
        MR_mid1 = right_membership1(mid1, b1, c1, fuzzy_data1)
        I1 = ((max_a - b1) / 6) * (1 + 4 * MR_mid1 + MR_max_a)
        ################################################################################################################
        MR_mid_xbar2_3 = right_membership1(mid_xbar2_3, b1, c1, fuzzy_data1)
        NL_mid_xbar2_3 = left_membership2(mid_xbar2_3, a2, b2, fuzzy_data2)
        #MR_x_bar = right_membership1(x_bar, b1, c1, fuzzy_data1)
        #NL_x_bar = left_membership2(x_bar, a2, b2, fuzzy_data2)
        I2 = ((x_bar - max_a) / 6) * (MR_max_a + 4 * (MR_mid_xbar2_3 - NL_mid_xbar2_3))
        ####################################################################################################################
        MR_mid_xbar2_2 = right_membership1(mid_xbar2_2, b1, c1, fuzzy_data1)
        NL_mid_xbar2_2 = left_membership2(mid_xbar2_2, a2, b2, fuzzy_data2)
        NL_c1 = left_membership2(c1, a2, b2, fuzzy_data2)
        I3 = ((min_c - x_bar) / 6) * (4 * (NL_mid_xbar2_2 - MR_mid_xbar2_2) + NL_c1)
        ####################################################################################################################
        NL_mid3 = left_membership2(mid3, a2, b2, fuzzy_data2)
        I4 = ((b2 - min_c) / 6) * (NL_c1 + 4 * NL_mid3 + 1)
        #############################################################################
        NR_mid_3_2 = right_membership2(mid_3_2, b2, c2, fuzzy_data2)
        I_4_1 = ((max_c - b2) / 6) * (1 + 4 * NR_mid_3_2)
        ###############################################################################
        R = I1 + I2 + I3 + I4 + I_4_1 + I_1_1
    if abs(R) < tolerance:
        R = 0
    return R , s

def generate_random_gaussian_fuzzy():
    """Generate a single random Gaussian fuzzy number (mu, sigma)."""
    mu = random.randint(-5000, 5000)
    sigma = random.randint(1, 500)
    return (mu, sigma)


def main():
    parser = argparse.ArgumentParser(description="Compare proposed method with Yager and Liou-Wang (Table 6).")
    parser.add_argument('--num_pairs', type=int, default=10000, help='Number of pairs')
    parser.add_argument('--seed', type=int, default=12, help='Random seed')
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Generating {args.num_pairs} Gaussian fuzzy pairs (seed={args.seed})...")

    # Generate all numbers at once (matching the paper's data generation)
    all_numbers = [generate_random_gaussian_fuzzy() for _ in range(2 * args.num_pairs)]
    data1 = all_numbers[:args.num_pairs]
    data2 = all_numbers[args.num_pairs:]

    # ============================================
    # Run the proposed method and compute ordering
    # ============================================
    print("Running proposed method...")
    proposed_signs = []
    for i in range(args.num_pairs):
        R, s = simpsons_rule(data1[i], data2[i])
        R_real = s * R
        if R_real > 0:
            proposed_signs.append(-1)
        elif R_real < 0:
            proposed_signs.append(1)
        else:
            proposed_signs.append(0)

    # ============================================
    # Run Yager's method
    # ============================================
    print("Running Yager's method...")
    yager_signs = []
    for i in range(args.num_pairs):
        s = compare_yager(data1[i][0], data1[i][1], data2[i][0], data2[i][1])
        yager_signs.append(s)

    # ============================================
    # Run Liou-Wang with beta = 0, 0.5, 1
    # ============================================
    print("Running Liou-Wang method (beta=0, 0.5, 1)...")
    lw_0_signs = []
    lw_05_signs = []
    lw_1_signs = []
    for i in range(args.num_pairs):
        mu1, sigma1 = data1[i]
        mu2, sigma2 = data2[i]
        lw_0_signs.append(compare_liou_wang(mu1, sigma1, mu2, sigma2, beta=0.0))
        lw_05_signs.append(compare_liou_wang(mu1, sigma1, mu2, sigma2, beta=0.5))
        lw_1_signs.append(compare_liou_wang(mu1, sigma1, mu2, sigma2, beta=1.0))

    # ============================================
    # Compute agreement percentages
    # ============================================
    def agreement_pct(proposed, other):
        proposed = np.array(proposed)
        other = np.array(other)
        agree = np.sum(proposed == other)
        return 100.0 * agree / len(proposed)

    results = [
        {'Ranking_Method': 'Yager',                     'Agreement_Percent': agreement_pct(proposed_signs, yager_signs)},
        {'Ranking_Method': 'Liou-Wang (beta=0)',        'Agreement_Percent': agreement_pct(proposed_signs, lw_0_signs)},
        {'Ranking_Method': 'Liou-Wang (beta=0.5)',      'Agreement_Percent': agreement_pct(proposed_signs, lw_05_signs)},
        {'Ranking_Method': 'Liou-Wang (beta=1)',        'Agreement_Percent': agreement_pct(proposed_signs, lw_1_signs)},
    ]

    df = pd.DataFrame(results)
    output_path = os.path.join(PROJECT_ROOT, 'results', 'tables', 'ranking_comparison.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    # ============================================
    # Print results
    # ============================================
    print("\n" + "=" * 60)
    print("Ordering Agreement with Proposed Method (Table 6)")
    print("=" * 60)
    for r in results:
        print(f"  {r['Ranking_Method']:<25} : {r['Agreement_Percent']:.2f}%")
    print("=" * 60)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
