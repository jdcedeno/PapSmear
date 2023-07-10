from numpy import zeros, shape, exp, negative, max as maxx, min as minn
from matplotlib.pyplot import figure, plot, grid


# ======================================== EULER CONTRAST ENHANCEMENT MAPPING ======================================== #


def euler_ce_map(x, times=1.0):
    y_exp = zeros(shape(x))
    for curr_count in range(len(x)):
        curr_y = (exp(x[curr_count] ** times)) - 1
        y_exp[curr_count] = curr_y
    return y_exp


# ======================================== ROOT CONTRAST ENHANCEMENT MAPPING ========================================= #


def root_ce_map(x, root=2.0):
    y_root = zeros(shape(x))
    for curr_count in range(len(x)):
        if x[curr_count] < 0:
            curr_y = 0
            y_root[curr_count] = curr_y
        else:
            curr_y = x[curr_count] ** (1 / root)
            y_root[curr_count] = curr_y
    return y_root


# ======================================== NINT CONTRAST ENHANCEMENT MAPPING ========================================= #


def nint_ce_map(x, t=6.0, factor=0.5):
    y_nint = zeros(shape(x))
    for curr_count in range(len(x)):
        curr_y = 1 / (1 + exp(-t * (x[curr_count] - factor)))
        y_nint[curr_count] = curr_y
    return y_nint


# ====================================== GAUSSIAN CONTRAST ENHANCEMENT MAPPING ======================================= #


def gaussian_ce_map(x, height, mean, std):
    y_gauss = zeros(shape(x))
    for curr_count in range(len(x)):
        if x[curr_count] <= mean:
            curr_x = x[curr_count]
            curr_y = height * exp(negative((curr_x - mean) ** 2 / (2 * (std ** 2))))
            y_gauss[curr_count] = curr_y
        else:
            curr_y = height
            y_gauss[curr_count] = curr_y
    return y_gauss


# ====================================== GAUSSIAN CONTRAST ENHANCEMENT MAPPING ======================================= #


def linearly_normalize_gauss(x, height, mean, std):
    max_val = height
    min_val = gaussian_ce_map([0], height, mean, std)
    m = 1 / (max_val - min_val)
    b = 1 - m
    y = zeros(shape(x))
    for count in range(len(x)):
        curr_x = x[count]
        curr_y = (m * curr_x) + b
        y[count] = curr_y
    return y


# ======================================= PLOT MAPPING FUNCTIONS IN DICTIONARY ======================================= #


def plot_dictionary(x, dict_in):
    for key, value in dict_in.items():
        figure(key)
        plot(x, value)
        grid()
