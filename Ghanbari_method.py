from scipy.integrate import quad
from fuzzy_utils import (left_membership1, right_membership1, left_membership2,
                         right_membership2, cubic_Lagrange_interpolant, fast_newton_xbar)

def calculate_R(Gussi_fuzzy1, Gussi_fuzzy2):

    mu1, sigma1 = Gussi_fuzzy1
    mu2, sigma2 = Gussi_fuzzy2

    tolerance = 1e-5

    fuzzy_data1 = cubic_Lagrange_interpolant(mu1, sigma1)
    fuzzy_data2 = cubic_Lagrange_interpolant(mu2, sigma2)

    # Ensure fuzzy_data1 has the smaller core for consistent logic
    if fuzzy_data1["fuzzy_number"][1] > fuzzy_data2["fuzzy_number"][1]:
        fuzzy_data1, fuzzy_data2 = fuzzy_data2, fuzzy_data1

    a1, b1, c1 = fuzzy_data1["fuzzy_number"]
    a2, b2, c2 = fuzzy_data2["fuzzy_number"]

    #x_bar = (mu1 * sigma2 + mu2 * sigma1) / (sigma1 + sigma2)
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
