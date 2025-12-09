import numpy as np

from scipy.signal import filtfilt
from scipy.stats import pearsonr


def estimate_correlation(s1, s2, b, a):
    s = np.vstack((s1, s2))
    s_filt = filtfilt(b, a, s)
    corr = pearsonr(s_filt[0, :], s_filt[1, :]).statistic
    return corr


METRICS = {
    "corr": estimate_correlation,
}
