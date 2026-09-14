import numpy as np
from scipy.optimize import brentq
from fuzzy_utils import lagrange_coefficients_fast_O1
def Spline_Fitting(mu, sigma):
    ks = np.arange(-3, 4)
    xs_all = mu + ks * sigma
    ys_all = np.exp(-((xs_all - mu) ** 2) / (2 * sigma ** 2))
    a = xs_all[0]
    c = xs_all[-1]

    xs_left = xs_all[0:4]
    ys_left = ys_all[0:4].copy()
    ys_left[0] = 0.0
    aL, bL, cL, dL = lagrange_coefficients_fast_O1(xs_left, ys_left)
    left_func = lambda x: aL * x ** 3 + bL * x ** 2 + cL * x + dL

    xs_right = xs_all[3:7]
    ys_right = ys_all[3:7].copy()
    ys_right[3] = 0.0
    aR, bR, cR, dR = lagrange_coefficients_fast_O1(xs_right, ys_right)
    right_func = lambda x: aR * x ** 3 + bR * x ** 2 + cR * x + dR

    def membership_function(x):
        if x <= a or x >= c:
            return 0.0
        elif x <= mu:
            return max(0.0, left_func(x))
        else:
            return max(0.0, right_func(x))

    return {
        "mu_func": membership_function,
        "support_min": a,
        "support_max": c,
        "breakpoints": [a, mu, c],
        "mu": mu,
        "sigma": sigma,
        "left_coeffs": (aL, bL, cL, dL),
        "right_coeffs": (aR, bR, cR, dR),
    }
# -----------------------------------------

def find_reference_intersections(f1, f2, common_min, common_max, num_steps=100000, xtol=1e-12):
    """
    Find all intersections between two membership functions using Brent's method
    as a high-accuracy reference.
    """
    x_values = np.linspace(common_min, common_max, num_steps)
    diff = np.array([f1(x) - f2(x) for x in x_values])
    intersections = []

    for i in range(num_steps - 1):
        if diff[i] == 0:
            intersections.append(x_values[i])
        elif diff[i] * diff[i + 1] < 0:
            try:
                root = brentq(
                    lambda x: f1(x) - f2(x),
                    x_values[i],
                    x_values[i + 1],
                    xtol=xtol,
                    rtol=xtol
                )
                intersections.append(root)
            except (ValueError, RuntimeError):
                # In case of non-convergence, use linear interpolation
                x0, x1 = x_values[i], x_values[i + 1]
                y0, y1 = diff[i], diff[i + 1]
                root = x0 - y0 * (x1 - x0) / (y1 - y0)
                intersections.append(root)

    # Remove duplicate points using tolerance
    if len(intersections) > 1:
        intersections = sorted(intersections)
        unique_intersections = []
        for x in intersections:
            if not unique_intersections or abs(x - unique_intersections[-1]) > 1e-10:
                unique_intersections.append(x)
        intersections = unique_intersections

    # Filter points with membership value above threshold
    intersections = [x for x in intersections if f1(x) > 1e-3 and f2(x) > 1e-3]

    return intersections


def find_correct_intersection(f1, f2, mu1, mu2, common_min, common_max):
    """
    Find the correct intersection point (between MR and NL branches).
    """
    x_reals = find_reference_intersections(f1, f2, common_min, common_max)

    if not x_reals:
        return None

    mu_min = min(mu1, mu2)
    mu_max = max(mu1, mu2)

    filtered = [x for x in x_reals if mu_min <= x <= mu_max]

    if filtered:
        return filtered[0]
    else:
        closest = min(x_reals, key=lambda x: min(abs(x - mu1), abs(x - mu2)))
        return closest