
from scipy.integrate import quad

def smooth_shape(t):
    return 1 - 10*t**3 + 15*t**4 - 6*t**5

def left_membership1(x, a, b):
    if x <= a:
        return 0.0
    elif x <= b:
        t = (b - x) / (b - a)
        return smooth_shape(t)
    else:
        return 0.0

def right_membership1(x, b, c):
    if x < b:
        return 0.0
    elif x < c:
        t = (x - b) / (c - b)
        return smooth_shape(t)
    else:
        return 0.0

def left_membership2(x, a, b):
    if x <= a:
        return 0.0
    elif x <= b:
        t = (b - x) / (b - a)
        return smooth_shape(t)
    else:
        return 0.0

def right_membership2(x, b, c):
    if x < b:
        return 0.0
    elif x < c:
        t = (x - b) / (c - b)
        return smooth_shape(t)
    else:
        return 0.0

def simpsons_rule(triangular_fuzzy1, triangular_fuzzy2):

    a1, b1, c1 = triangular_fuzzy1
    a2, b2, c2 = triangular_fuzzy2
    tolerance = 1e-5
    if b1 > b2:
        a1, b1, c1 = triangular_fuzzy2
        a2, b2, c2 = triangular_fuzzy1

    R = 0

    if c1 <= a2:

        mid2 = (a1 + b1) / 2
        mid3 = (b1 + c1) / 2
        mid7 = (a2 + b2) / 2
        mid8 = (b2 + c2) / 2

        R = (2 / 3) * ((b1 - a1) * left_membership1(mid2, a1, b1) + (c1 - b1) * right_membership1(mid3, b1, c1)
                       + (b2 - a2) * left_membership2(mid7, a2, b2) + (c2 - b2) * right_membership2(mid8, b2, c2)) + (
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
        ML_mid_1_1 = left_membership1(mid_1_1, a1, b1)
        NL_mid_1_1 = left_membership2(mid_1_1, a2, b2)
        ML_max_a = left_membership1(max_a, a1, b1)
        NL_max_a = left_membership2(max_a, a2, b2)

        I_1_1 = ((max_a - min_a) / 6) * (4 * (ML_mid_1_1 - NL_mid_1_1) + (ML_max_a - NL_max_a))
######################################################################################################################
        ML_mid1 = left_membership1(mid1, a1, b1)
        NL_mid1 = left_membership2(mid1, a2, b2)
        I1 = ((b1 - max_a) / 6) * ((ML_max_a - NL_max_a) + 4 * (ML_mid1 - NL_mid1))
        ##################################################################################################################
        NR_mid_1_5 = right_membership2(mid_1_5, b2, c2)
        MR_mid_1_5 = right_membership1(mid_1_5, b1, c1)
        NR_min_c = right_membership2(min_c, b2, c2)
        MR_min_c = right_membership1(min_c, b1, c1)

        I2 = ((min_c - b1) / 6) * (4 * (NR_mid_1_5 - MR_mid_1_5) + (NR_min_c - MR_min_c))
    ##################################################################################################################
        #MR_max_c = right_membership1(max_c, b1, c1, fuzzy_data1)
        #NR_max_c = right_membership2(max_c, b2, c2, fuzzy_data2)
        MR_mid_3_1 = right_membership1(mid_3_1, b1, c1)
        NR_mid_3_1 = right_membership2(mid_3_1, b2, c2)

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

        # فرمول صحیح برای توابع سهموی
        x_bar = (b1 * a2 - b2 * c1) / (a2 - b2 - c1 + b1)

        mid1 = (max_a + b1) / 2
        mid_1_1 = (max_a + min_a) / 2
        mid3 = (b2 + min_c) / 2
        mid_xbar1 = (b1 + x_bar) / 2
        mid_xbar2 = (b2 + x_bar) / 2
        mid_3_1 = (max_c + min_c) / 2
        ###################################################################################
        #ML_mia_a = left_membership1(min_a, a1, b1, fuzzy_data1)
        #NL_min_a = left_membership2(min_a, a2, b2, fuzzy_data2)

        ML_mid_1_1 = left_membership1(mid_1_1, a1, b1)
        NL_mid_1_1 = left_membership2(mid_1_1, a2, b2)

        ML_max_a = left_membership1(max_a, a1, b1)
        NL_max_a = left_membership2(max_a, a2, b2)

        I_1_1 = ((max_a - min_a) / 6) * (4 * (ML_mid_1_1 - NL_mid_1_1) + (ML_max_a - NL_max_a))
        #################################################################################################################
        ML_mid1 = left_membership1(mid1, a1, b1)
        NL_mid1 = left_membership2(mid1, a2, b2)

        NL_b1 = left_membership2(b1, a2, b2)
        I1 = ((b1 - max_a) / 6) * ((ML_max_a - NL_max_a) + 4 * (ML_mid1 - NL_mid1) + (1 - NL_b1))
        ################################################################################################################
        MR_mid_xbar1 = right_membership1(mid_xbar1, b1, c1)
        NL_mid_xbar1 = left_membership2(mid_xbar1, a2, b2)
        #MR_x_bar = right_membership1(x_bar, b1, c1, fuzzy_data1)
        #NL_x_bar = left_membership2(x_bar, a2, b2, fuzzy_data2)
        I2 = ((x_bar - b1) / 6) * ((1 - NL_b1) + 4 * (MR_mid_xbar1 - NL_mid_xbar1))
        ####################################################################################################################
        MR_mid_xbar2 = right_membership1(mid_xbar2, b1, c1)
        NL_mid_xbar2 = left_membership2(mid_xbar2, a2, b2)
        MR_b2 = right_membership1(b2, b1, c1)
        I3 = ((b2 - x_bar) / 6) * (4 * (NL_mid_xbar2 - MR_mid_xbar2) + (1 - MR_b2))
        ####################################################################################################################
        MR_mid3 = right_membership1(mid3, b1, c1)
        NR_mid3 = right_membership2(mid3, b2, c2)
        MR_b2 = right_membership1(b2, b1, c1)
        MR_min_c = right_membership1(min_c, b1, c1)
        NR_min_c = right_membership2(min_c, b2, c2)

        I4 = ((min_c - b2) / 6) * ((1 - MR_b2) + 4 * (NR_mid3 - MR_mid3) + (NR_min_c - MR_min_c))
        #############################################################################
        #MR_max_c = right_membership1(max_c, b1, c1, fuzzy_data1)
        #NR_max_c = right_membership2(max_c, b2, c2, fuzzy_data2)
        MR_mid_3_1 = right_membership1(mid_3_1, b1, c1)
        NR_mid_3_1 = right_membership2(mid_3_1, b2, c2)

        I_4_1 = ((max_c - min_c) / 6) * ((NR_min_c - MR_min_c) + 4 * (NR_mid_3_1 - MR_mid_3_1))

        R = I1 + I2 + I3 + I4 + I_4_1 + I_1_1
    ####################################################################################################################
    ## case:5
    elif b1 < b2 and b2 <= c1 and a2 >= b1:
        max_a = max(a1, a2)
        min_a = min(a1, a2)
        min_c = min(c1, c2)
        max_c = max(c1, c2)

        # فرمول صحیح برای توابع سهموی
        x_bar = (b1 * a2 - b2 * c1) / (a2 - b2 - c1 + b1)

        mid1_1 = (min_a + b1) / 2
        mid_1_2 = (max_a + b1) / 2
        mid3 = (b2 + min_c) / 2
        mid_xbar1_1 = (max_a + x_bar) / 2
        mid_xbar2 = (b2 + x_bar) / 2
        mid_3_1 = (max_c + min_c) / 2
        ##11111#################################################################################
        ML_mid1_1 = left_membership1(mid1_1, a1, b1)
        I_1_1 = ((b1 - min_a) / 6) * (4 * ML_mid1_1 + 1)
        ####2222222#############################################################################################################
        MR_mid_1_2 = right_membership1(mid_1_2, b1, c1)
        #NL_mid_1_2 = left_membership2(mid_1_2, a2, b2, fuzzy_data2)
        MR_max_a = right_membership1(max_a, b1, c1)
        #NL_b1 = left_membership2(b1, a2, b2, fuzzy_data2)
        I1 = ((max_a - b1) / 6) * (1 + 4 * MR_mid_1_2 + MR_max_a)
        ################################################################################################################
        MR_mid_xbar1_1 = right_membership1(mid_xbar1_1, b1, c1)
        NL_mid_xbar1_1 = left_membership2(mid_xbar1_1, a2, b2)
        #MR_x_bar = right_membership1(x_bar, b1, c1, fuzzy_data1)
        #NL_x_bar = left_membership2(x_bar, a2, b2, fuzzy_data2)
        I2 = ((x_bar - max_a) / 6) * (MR_max_a + 4 * (MR_mid_xbar1_1 - NL_mid_xbar1_1))
        ####################################################################################################################
        MR_mid_xbar2 = right_membership1(mid_xbar2, b1, c1)
        NL_mid_xbar2 = left_membership2(mid_xbar2, a2, b2)
        MR_b2 = right_membership1(b2, b1, c1)
        I3 = ((b2 - x_bar) / 6) * (4 * (NL_mid_xbar2 - MR_mid_xbar2) + (1 - MR_b2))
        ####################################################################################################################
        MR_mid3 = right_membership1(mid3, b1, c1)
        NR_mid3 = right_membership2(mid3, b2, c2)
        MR_b2 = right_membership1(b2, b1, c1)
        MR_min_c = right_membership1(min_c, b1, c1)
        NR_min_c = right_membership2(min_c, b2, c2)

        I4 = ((min_c - b2) / 6) * ((1 - MR_b2) + 4 * (NR_mid3 - MR_mid3) + (NR_min_c - MR_min_c))
        #############################################################################
        #MR_max_c = right_membership1(max_c, b1, c1, fuzzy_data1)
        #NR_max_c = right_membership2(max_c, b2, c2, fuzzy_data2)
        MR_mid_3_1 = right_membership1(mid_3_1, b1, c1)
        NR_mid_3_1 = right_membership2(mid_3_1, b2, c2)

        I_4_1 = ((max_c - min_c) / 6) * ((NR_min_c - MR_min_c) + 4 * (NR_mid_3_1 - MR_mid_3_1))

        R = I1 + I2 + I3 + I4 + I_4_1 + I_1_1
    ####################################################################################################################
    ## case: 4
    elif b1 < b2 and c1 <= b2 and a2 <= b1:

        max_a = max(a1, a2)
        min_a = min(a1, a2)
        min_c = min(c1, c2)
        max_c = max(c1, c2)


        x_bar = (b1 * a2 - b2 * c1) / (a2 - b2 - c1 + b1)

        mid1 = (max_a + b1) / 2
        mid_1_1 = (max_a + min_a) / 2
        mid3 = (b2 + min_c) / 2
        mid_xbar1 = (b1 + x_bar) / 2
        mid_xbar2_2 = (min_c + x_bar) / 2
        mid_3_2 = (max_c + b2) / 2

        ###################################################################################
        #ML_mia_a = left_membership1(min_a, a1, b1, fuzzy_data1)
        #NL_min_a = left_membership2(min_a, a2, b2, fuzzy_data2)

        ML_mid_1_1 = left_membership1(mid_1_1, a1, b1)
        NL_mid_1_1 = left_membership2(mid_1_1, a2, b2)

        ML_max_a = left_membership1(max_a, a1, b1)
        NL_max_a = left_membership2(max_a, a2, b2)

        I_1_1 = ((max_a - min_a) / 6) * (4 * (ML_mid_1_1 - NL_mid_1_1) + (ML_max_a - NL_max_a))
        #################################################################################################################
        ML_mid1 = left_membership1(mid1, a1, b1)
        NL_mid1 = left_membership2(mid1, a2, b2)
        NL_b1 = left_membership2(b1, a2, b2)
        I1 = ((b1 - max_a) / 6) * ((ML_max_a - NL_max_a) + 4 * (ML_mid1 - NL_mid1) + (1 - NL_b1))
        ################################################################################################################
        MR_mid_xbar1 = right_membership1(mid_xbar1, b1, c1)
        NL_mid_xbar1 = left_membership2(mid_xbar1, a2, b2)
        #MR_x_bar = right_membership1(x_bar, b1, c1, fuzzy_data1)
        #NL_x_bar = left_membership2(x_bar, a2, b2, fuzzy_data2)
        I2 = ((x_bar - b1) / 6) * ((1 - NL_b1) + 4 * (MR_mid_xbar1 - NL_mid_xbar1))
        ####################################################################################################################
        MR_mid_xbar2_2 = right_membership1(mid_xbar2_2, b1, c1)
        NL_mid_xbar2_2 = left_membership2(mid_xbar2_2, a2, b2)
        NL_c1 = left_membership2(c1, a2, b2)
        I3 = ((min_c - x_bar) / 6) * (4 * (NL_mid_xbar2_2 - MR_mid_xbar2_2) + NL_c1)
        ####################################################################################################################
        NL_min_c = left_membership2(min_c, a2, b2)
        NL_mid3 = left_membership2(mid3, a2, b2)

        I4 = ((b2 - min_c) / 6) * (NL_min_c + 4 * NL_mid3 + 1)
        #############################################################################

        NR_mid_3_2 = right_membership2(mid_3_2, b2, c2)
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

        # فرمول صحیح برای توابع سهموی
        x_bar = (b1 * a2 - b2 * c1) / (a2 - b2 - c1 + b1)

        mid1 = (max_a + b1) / 2
        mid_2_1 = (min_a + b1) / 2
        mid3 = (b2 + min_c) / 2
        mid_xbar2_2 = (min_c + x_bar) / 2
        mid_xbar2_3 = (max_a + x_bar) / 2
        mid_3_2 = (max_c + b2) / 2
        ###################################################################################
        ML_mid_2_1 = left_membership1(mid_2_1, a1, b1)
        I_1_1 = ((b1 - a1) / 6) * (4 * ML_mid_2_1 + 1)
        #################################################################################################################
        MR_max_a = right_membership1(max_a, b1, c1)
        MR_mid1 = right_membership1(mid1, b1, c1)
        I1 = ((max_a - b1) / 6) * (1 + 4 * MR_mid1 + MR_max_a)
        ################################################################################################################
        MR_mid_xbar2_3 = right_membership1(mid_xbar2_3, b1, c1)
        NL_mid_xbar2_3 = left_membership2(mid_xbar2_3, a2, b2)
        #MR_x_bar = right_membership1(x_bar, b1, c1, fuzzy_data1)
        #NL_x_bar = left_membership2(x_bar, a2, b2, fuzzy_data2)
        I2 = ((x_bar - max_a) / 6) * (MR_max_a + 4 * (MR_mid_xbar2_3 - NL_mid_xbar2_3))
        ####################################################################################################################
        MR_mid_xbar2_2 = right_membership1(mid_xbar2_2, b1, c1)
        NL_mid_xbar2_2 = left_membership2(mid_xbar2_2, a2, b2)
        NL_c1 = left_membership2(c1, a2, b2)
        I3 = ((min_c - x_bar) / 6) * (4 * (NL_mid_xbar2_2 - MR_mid_xbar2_2) + NL_c1)
        ####################################################################################################################
        NL_mid3 = left_membership2(mid3, a2, b2)
        I4 = ((b2 - min_c) / 6) * (NL_c1 + 4 * NL_mid3 + 1)
        #############################################################################
        NR_mid_3_2 = right_membership2(mid_3_2, b2, c2)
        I_4_1 = ((max_c - b2) / 6) * (1 + 4 * NR_mid_3_2)
        ###############################################################################
        R = I1 + I2 + I3 + I4 + I_4_1 + I_1_1
    if abs(R) < tolerance:
        R = 0
    return R
def calculate_R(fuzzy1, fuzzy2):
    a1, b1, c1 = fuzzy1
    a2, b2, c2 = fuzzy2
    tolerance = 1e-5

    if b1 > b2:
        a1, b1, c1 = fuzzy2
        a2, b2, c2 = fuzzy1

    x_bar = (b1 * a2 - b2 * c1) / (a2 - b2 - c1 + b1)

    def diff_function1(x, a1, b1, a2, b2):
        return left_membership1(x, a1, b1) - left_membership2(x, a2, b2)

    I1, error1 = quad(diff_function1, min(a1, a2), b1, args=(a1, b1, a2, b2))

    def diff_function2(x, b1, c1, a2, b2):
        return right_membership1(x, b1, c1) - left_membership2(x, a2, b2)

    I2, error2 = quad(diff_function2, b1, x_bar, args=(b1, c1, a2, b2))

    def diff_function3(x, b1, c1, a2, b2):
        return -right_membership1(x, b1, c1) + left_membership2(x, a2, b2)

    I3, error3 = quad(diff_function3, x_bar, b2, args=(b1, c1, a2, b2))

    def diff_function4(x, b2, c2, b1, c1):
        return right_membership2(x, b2, c2) - right_membership1(x, b1, c1)

    I4, error4 = quad(diff_function4, b2, max(c1, c2), args=(b2, c2, b1, c1))

    R = I1 + I2 + I3 + I4

    if abs(R) < tolerance:
        R = 0

    return R
