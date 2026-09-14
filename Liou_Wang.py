import numpy as np

def liou_wang_score(mu, sigma, beta=0.5):
    """
    Compute Liou-Wang index for a Gaussian fuzzy number.
    Formula: IV_T^beta = mu + (2*beta - 1) * sigma * sqrt(pi/2)
    """
    return mu + (2 * beta - 1) * sigma * np.sqrt(np.pi / 2)

def compare_liou_wang(mu1, sigma1, mu2, sigma2, beta=0.5):
    """
    Compare two Gaussian fuzzy numbers using Liou-Wang method.
    Returns: 1 if first > second, -1 if second > first, 0 if equal.
    """
    s1 = liou_wang_score(mu1, sigma1, beta)
    s2 = liou_wang_score(mu2, sigma2, beta)

    if s1 > s2:
        return 1
    elif s1 < s2:
        return -1
    else:
        return 0