from scipy.ndimage.filters import uniform_filter
import numpy as np


def decibel_to_linear(band):
    return np.power(10, np.array(band) / 10)


def linear_to_decibel(band):
    return 10 * np.log10(band)


def lee_filter(band, window, var_noise=0.25):
    lin_band = decibel_to_linear(band)
    mean_window = uniform_filter(lin_band, window)
    mean_sqr_window = uniform_filter(lin_band ** 2, window)
    var_window = mean_sqr_window - mean_window ** 2

    weights = var_window / (var_window + var_noise)
    band_filtered = mean_window + weights * (lin_band - mean_window)
    band_filtered = linear_to_decibel(band_filtered)

    return band_filtered
