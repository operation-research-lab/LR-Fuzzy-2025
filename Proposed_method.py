
from fuzzy_utils import (left_membership1, right_membership1, left_membership2,
                         right_membership2, cubic_Lagrange_interpolant, fast_newton_xbar)
def simpsons_rule(Gussi_fuzzy1, Gussi_fuzzy2):
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
    return R
