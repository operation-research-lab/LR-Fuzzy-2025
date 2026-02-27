import pandas as pd
import numpy as np
import time
from scipy.interpolate import CubicSpline

def simpsons_rule(Gussi_fuzzy1, Gussi_fuzzy2):
    global R
    mu1, sigma1 = Gussi_fuzzy1
    mu2, sigma2 = Gussi_fuzzy2
    tolerance = 1e-5

    fuzzy_data1 = Spline_Fitting(mu1, sigma1)
    fuzzy_data2 = Spline_Fitting(mu2, sigma2)

    # Ensure fuzzy_data1 has the smaller core for consistent logic
    if fuzzy_data1["fuzzy_number"][1] > fuzzy_data2["fuzzy_number"][1]:
        fuzzy_data1, fuzzy_data2 = fuzzy_data2, fuzzy_data1

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

        MR_mid_3_1 = right_membership1(mid_3_1, b1, c1, fuzzy_data1)
        NR_mid_3_1 = right_membership2(mid_3_1, b2, c2, fuzzy_data2)

        I_4_1 = ((max_c - min_c) / 6) * ((NR_min_c - MR_min_c) + 4 * (NR_mid_3_1 - MR_mid_3_1))

        R = I1 + I2 + I_4_1 + I_1_1
    ####################################################################################################################
    ## case:3
    elif b1 < b2 and a2 <= b1 and b2 <= c1:

        max_a = max(a1, a2)
        min_a = min(a1, a2)
        min_c = min(c1, c2)
        max_c = max(c1, c2)


        x_bar = fast_newton_xbar(fuzzy_data1, fuzzy_data2, x0=None)
        mid1 = (max_a + b1) / 2
        mid_1_1 = (max_a + min_a) / 2
        mid3 = (b2 + min_c) / 2
        mid_xbar1 = (b1 + x_bar) / 2
        mid_xbar2 = (b2 + x_bar) / 2
        mid_3_1 = (max_c + min_c) / 2
        ###################################################################################


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
        MR_max_a = right_membership1(max_a, b1, c1, fuzzy_data1)
        I1 = ((max_a - b1) / 6) * (1 + 4 * MR_mid_1_2 + MR_max_a)
        ################################################################################################################
        MR_mid_xbar1_1 = right_membership1(mid_xbar1_1, b1, c1, fuzzy_data1)
        NL_mid_xbar1_1 = left_membership2(mid_xbar1_1, a2, b2, fuzzy_data2)
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

        x_bar = fast_newton_xbar(fuzzy_data1, fuzzy_data2, x0=None)
        mid1 = (max_a + b1) / 2
        mid_1_1 = (max_a + min_a) / 2
        mid3 = (b2 + min_c) / 2
        mid_xbar1 = (b1 + x_bar) / 2
        mid_xbar2_2 = (min_c + x_bar) / 2
        mid_3_2 = (max_c + b2) / 2

        ###################################################################################
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
    return R

################################################################
def left_membership1(x, a, b, fuzzy_data1):
    if x <= a or x > b:
        raw_value = 0
    else:
        raw_value = fuzzy_data1["left_function"](x)
    return max(0, raw_value)


def right_membership1(x, b, c, fuzzy_data1):
    if x >= c or x < b:
        raw_value = 0
    else:
        raw_value = fuzzy_data1["right_function"](x)
    return max(0, raw_value)


def left_membership2(x, a, b, fuzzy_data2):
    if x <= a or x > b:
        raw_value = 0
    else:
        raw_value = fuzzy_data2["left_function"](x)
    return max(0, raw_value)


def right_membership2(x, b, c, fuzzy_data2):
    if x >= c or x < b:
        raw_value = 0
    else:
        raw_value = fuzzy_data2["right_function"](x)
    return max(0, raw_value)



def lagrange_interpolation(xs, ys):

    def poly(x):
        total = 0.0
        n = len(xs)
        for i in range(n):
            term = ys[i]
            for j in range(n):
                if i != j:
                    term *= (x - xs[j]) / (xs[i] - xs[j])
            total += term
        return total

    return poly
def lagrange_coefficients_fast_O1(xs, ys):

    x0, x1, x2, x3 = xs[0], xs[1], xs[2], xs[3]
    y0, y1, y2, y3 = ys[0], ys[1], ys[2], ys[3]

    d01 = x0 - x1
    d02 = x0 - x2
    d03 = x0 - x3
    d12 = x1 - x2
    d13 = x1 - x3
    d23 = x2 - x3

    d10 = -d01
    d20 = -d02
    d21 = -d12
    d30 = -d03
    d31 = -d13
    d32 = -d23

    denom0 = d01 * d02 * d03  # (x0-x1)(x0-x2)(x0-x3)
    denom1 = d10 * d12 * d13  # (x1-x0)(x1-x2)(x1-x3)
    denom2 = d20 * d21 * d23  # (x2-x0)(x2-x1)(x2-x3)
    denom3 = d30 * d31 * d32  # (x3-x0)(x3-x1)(x3-x2)


    # P(x) = y0*L0(x) + y1*L1(x) + y2*L2(x) + y3*L3(x)

    # coeff x³
    a = (y0 / denom0 + y1 / denom1 + y2 / denom2 + y3 / denom3)

    # coeff x²
    b = -(y0 * (x1 + x2 + x3) / denom0 +
          y1 * (x0 + x2 + x3) / denom1 +
          y2 * (x0 + x1 + x3) / denom2 +
          y3 * (x0 + x1 + x2) / denom3)

    # coeff x¹
    c = (y0 * (x1 * x2 + x1 * x3 + x2 * x3) / denom0 +
         y1 * (x0 * x2 + x0 * x3 + x2 * x3) / denom1 +
         y2 * (x0 * x1 + x0 * x3 + x1 * x3) / denom2 +
         y3 * (x0 * x1 + x0 * x2 + x1 * x2) / denom3)

    # fixed coeff
    d = -(y0 * x1 * x2 * x3 / denom0 +
          y1 * x0 * x2 * x3 / denom1 +
          y2 * x0 * x1 * x3 / denom2 +
          y3 * x0 * x1 * x2 / denom3)

    return a, b, c, d

def Spline_Fitting(mu, sigma):
    ks = np.arange(-3, 4)
    xs_all = mu + ks * sigma
    ys_all = np.exp(-((xs_all - mu) ** 2) / (2 * sigma ** 2))

    xs_left = xs_all[0:4]
    ys_left = ys_all[0:4].copy()
    ys_left[0] = 0.0
    ys_left[3] = 1.0

    xs_right = xs_all[3:7]
    ys_right = ys_all[3:7].copy()
    ys_right[0] = 1.0
    ys_right[3] = 0.0

    left_func = lagrange_interpolation(xs_left, ys_left)
    right_func = lagrange_interpolation(xs_right, ys_right)

    aR, bR, cR, dR = lagrange_coefficients_fast_O1(xs_right, ys_right)
    aL, bL, cL, dL = lagrange_coefficients_fast_O1(xs_left, ys_left)

    return {
        "fuzzy_number": (xs_all[0], mu, xs_all[-1]),
        "mu": mu,
        "sigma": sigma,
        "left_coeffs": (aL, bL, cL, dL),
        "right_coeffs": (aR, bR, cR, dR),
        "left_function": left_func,
        "right_function": right_func
    }


def fast_newton_xbar(data1, data2, x0=None):

    mu1, sigma1 = data1["mu"], data1["sigma"]
    mu2, sigma2 = data2["mu"], data2["sigma"]

    if x0 is None:
        x0 = (mu1 * sigma2 + mu2 * sigma1) / (sigma1 + sigma2)

    aR1, bR1, cR1, dR1 = data1["right_coeffs"]
    aL2, bL2, cL2, dL2 = data2["left_coeffs"]

    x = x0
    x2 = x * x
    x3 = x2 * x

    # f(x) = (aR1 - aL2)x³ + (bR1 - bL2)x² + (cR1 - cL2)x + (dR1 - dL2)
    a_diff = aR1 - aL2
    b_diff = bR1 - bL2
    c_diff = cR1 - cL2
    d_diff = dR1 - dL2

    f_val = a_diff * x3 + b_diff * x2 + c_diff * x + d_diff

    if abs(f_val) < 1e-5:
        return x0

    # f'(x) = 3a_diff*x² + 2b_diff*x + c_diff
    f_prime_val = 3 * a_diff * x2 + 2 * b_diff * x + c_diff

    if abs(f_prime_val) < 1e-12:
        return x0
    print(x0)
    print(x0 - (f_val / f_prime_val))
    return x0 - (f_val / f_prime_val)



df1 = pd.read_excel('fuzzy_numbers.xlsx', sheet_name='Sheet1')
df2 = pd.read_excel('fuzzy_numbers.xlsx', sheet_name='Sheet2')

R_values = []

start_time = time.time()
for index, (row1, row2) in enumerate(zip(df1.iterrows(), df2.iterrows())):
    triangular_fuzzy1 = (row1[1]['mu'], row1[1]['sigma'])
    triangular_fuzzy2 = (row2[1]['mu'], row2[1]['sigma'])
    R = simpsons_rule(triangular_fuzzy1, triangular_fuzzy2)
    R_values.append(R)
end_time = time.time()
results_df = pd.DataFrame(R_values, columns=['simpson'])

with pd.ExcelWriter('fuzzy_numbers.xlsx', engine='openpyxl', mode='a') as writer:
    results_df.to_excel(writer, sheet_name='simpson', index=False)

print("Data processed and saved to fuzzy_numbers.xlsx")

execution_time = end_time - start_time
print(f": execution time is: {execution_time:.2f} Seconds")