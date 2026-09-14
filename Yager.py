
def yager_score(mu, sigma):

    return mu

def compare_yager(mu1, sigma1, mu2, sigma2):
    s1 = yager_score(mu1, sigma1)
    s2 = yager_score(mu2, sigma2)

    if s1 > s2:
        return 1
    elif s1 < s2:
        return -1
    else:
        return 0

