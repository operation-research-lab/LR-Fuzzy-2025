from scipy.integrate import quad
import pandas as pd
import numpy as np
import time

def calculate_R(Gussi_fuzzy1, Gussi_fuzzy2):

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

    x_bar = fast_newton_xbar(fuzzy_data1, fuzzy_data2, x0=None)

    def diff_function1(x, a1, b1, a2, b2):
        return left_membership1(x, a1, b1, fuzzy_data1) - left_membership2(x, a2, b2, fuzzy_data2)

    I1, error1 = quad(diff_function1, min(a1, a2), b1, args=(a1, b1, a2, b2), limit=50, epsabs=1e-6, epsrel=1e-6)

    def diff_function2(x, b1, c1, a2, b2):
        return right_membership1(x, b1, c1, fuzzy_data1) - left_membership2(x, a2, b2, fuzzy_data2)

    I2, error2 = quad(diff_function2, b1, x_bar, args=(b1, c1, a2, b2), limit=50, epsabs=1e-6, epsrel=1e-6)

    def diff_function3(x, b1, c1, a2, b2):
        return -right_membership1(x, b1, c1, fuzzy_data1) + left_membership2(x, a2, b2, fuzzy_data2)

    I3, error3 = quad(diff_function3, x_bar, b2, args=(b1, c1, a2, b2), limit=50, epsabs=1e-6, epsrel=1e-6)

    def diff_function4(x, b2, c2, b1, c1):
        return right_membership2(x, b2, c2, fuzzy_data2) - right_membership1(x, b1, c1, fuzzy_data1)

    I4, error4 = quad(diff_function4, b2, max(c1, c2), args=(b2, c2, b1, c1), limit=50, epsabs=1e-6, epsrel=1e-6)

    R = I1 + I2 + I3 + I4

    if abs(R) < tolerance:
        R = 0
    return R
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


    # coeef x³
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

    aL, bL, cL, dL = lagrange_coefficients_fast_O1(xs_left, ys_left)
    aR, bR, cR, dR = lagrange_coefficients_fast_O1(xs_right, ys_right)

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

        #  right_func1 - left_func2
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

    return x0 - (f_val / f_prime_val)



df1 = pd.read_excel('fuzzy_numbers.xlsx', sheet_name='Sheet1')
df2 = pd.read_excel('fuzzy_numbers.xlsx', sheet_name='Sheet2')

R_values = []
start_time = time.time()
for index, (row1, row2) in enumerate(zip(df1.iterrows(), df2.iterrows())):
    triangular_fuzzy1 = (row1[1]['mu'], row1[1]['sigma'])
    triangular_fuzzy2 = (row2[1]['mu'], row2[1]['sigma'])

    R = calculate_R(triangular_fuzzy1, triangular_fuzzy2)
    R_values.append(R)
end_time = time.time()

results_df = pd.DataFrame(R_values, columns=['calculate_R'])

with pd.ExcelWriter('fuzzy_numbers.xlsx', engine='openpyxl', mode='a') as writer:
    results_df.to_excel(writer, sheet_name='calculate_R', index=False)

print("Data processed and saved to fuzzy_numbers.xlsx")


execution_time = end_time - start_time
print(f": execution time is: {execution_time:.2f} Seconds")