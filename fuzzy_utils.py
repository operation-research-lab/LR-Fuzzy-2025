import numpy as np
def left_membership1(x, a, b, fuzzy_data1):
    if x <= a or x > b:
        raw_value = 0
    else:
        raw_value = fuzzy_data1["left_function"](x)
    return max(0, raw_value)
    #return raw_value

def right_membership1(x, b, c, fuzzy_data1):
    if x >= c or x < b:
        raw_value = 0
    else:
        raw_value = fuzzy_data1["right_function"](x)
    return max(0, raw_value)
    #return raw_value

def left_membership2(x, a, b, fuzzy_data2):
    if x <= a or x > b:
        raw_value = 0
    else:
        raw_value = fuzzy_data2["left_function"](x)
    return max(0, raw_value)
    #return raw_value

def right_membership2(x, b, c, fuzzy_data2):
    if x >= c or x < b:
        raw_value = 0
    else:
        raw_value = fuzzy_data2["right_function"](x)
    return max(0, raw_value)
    #return raw_value

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

    # L0(x) = (x-x1)(x-x2)(x-x3) / denom0
    # L1(x) = (x-x0)(x-x2)(x-x3) / denom1
    # L2(x) = (x-x0)(x-x1)(x-x3) / denom2
    # L3(x) = (x-x0)(x-x1)(x-x2) / denom3

    # P(x) = y0*L0(x) + y1*L1(x) + y2*L2(x) + y3*L3(x)


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

def cubic_Lagrange_interpolant(mu, sigma):
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

    #left_func = np.poly1d([aL, bL, cL, dL])
    #right_func = np.poly1d([aR, bR, cR, dR])
    """
    def left_func(x):
        return aL * x ** 3 + bL * x ** 2 + cL * x + dL

    def right_func(x):
        return aR * x ** 3 + bR * x ** 2 + cR * x + dR
    """
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

        # حالت معمول: right_func1 - left_func2
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
