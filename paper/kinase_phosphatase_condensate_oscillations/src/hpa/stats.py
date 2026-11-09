"""
Statistics, fit models and correlations of time series.

- fit models: linear, tanh_general
- errors: mean_err (replicas), average / sigma_blocking / autocorr_bin (correlated samples)
- correlations: cross_correlation, sliding_window_correlation
"""
import numpy as np
import scipy.signal as signal
from matplotlib import pyplot as plt


# ---------------------------------------------------------------------------
# Fit models
# ---------------------------------------------------------------------------

def linear(x, a, b):
    """a * x + b"""
    return a * x + b


def tanh_general(x, a, b, c, d):
    """a + b * tanh(c * (x - d)): density profile across an interface at d, width 1/c."""
    return a + b * np.tanh(c * (x - d))


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

def mean_err(values):
    """Mean and standard error over independent replicas."""
    return np.mean(values), np.std(values) / np.sqrt(len(values) - 1)


def average(sample, n_term=0):
    """Sample average; the first n_term data are discarded."""
    return (sample[n_term:]).sum() / (len(sample) - n_term)


def sigma_blocking(sample, pow_bin=0, n_term=0):
    """Standard error of a correlated sample with the blocking technique (block size 2**pow_bin);
    the first n_term data are discarded."""
    dim_bin = 2**pow_bin
    n_bin = int((len(sample[n_term:])) / dim_bin)
    N = n_bin * dim_bin

    mu = average(sample[n_term:n_term + N])
    sum_block = (np.array([(average(sample[n_term + i * dim_bin: n_term + (i + 1) * dim_bin]) - mu)**2
                           for i in range(n_bin)])).sum()

    return np.sqrt(sum_block) * dim_bin / N


def autocorr_bin(sample, pow_bin_max=10, n_term=0):
    """Plot sigma_blocking / sigma_blocking(pow_bin=0) vs pow_bin: the plateau gives the correlation length."""
    pow_bin = list(range(pow_bin_max))
    sigma = np.array([sigma_blocking(sample, x, n_term) for x in pow_bin])
    sigma = sigma / sigma[0]
    plt.figure(10)
    plt.grid(True)
    plt.plot(pow_bin, sigma)
    plt.xlabel('log$_{10}$ correlation block size')
    plt.ylabel('error estimate with blocking')
    plt.show()


# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------

def cross_correlation(x, y):
    """Normalised cross-correlation of the mean-subtracted signals.

    Returns lags (in frames) and correlation.
    """
    x = np.asarray(x) - np.mean(x)
    y = np.asarray(y) - np.mean(y)
    corr = signal.correlate(x, y, mode='full') / np.sqrt(np.sum(x**2) * np.sum(y**2))
    return signal.correlation_lags(len(x), len(y), mode='full'), corr


def sliding_window_correlation(x, y, window, step, dt=1.0):
    """Normalised cross-correlation in sliding windows of `window` frames, shifted by `step` frames.

    dt is the spacing of the series (e.g. 0.04 us for the MD size files).
    Returns C (n_windows, 2*window-1), lags and window centres, both in units of dt.
    """
    correlations, times = [], []
    for start in range(0, len(x) - window, step):
        xx = x[start:start + window] - np.mean(x[start:start + window])
        yy = y[start:start + window] - np.mean(y[start:start + window])
        correlations.append(signal.correlate(xx, yy, mode='full') / np.sqrt(np.sum(xx**2) * np.sum(yy**2)))
        times.append((start + window / 2) * dt)
    lags = signal.correlation_lags(window, window, mode='full') * dt
    return np.asarray(correlations), lags, np.asarray(times)
