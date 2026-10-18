"""Hand-crafted statistical features for the Random Forest model."""

import numpy as np
from scipy import stats

FEATURE_NAMES = [
    "Mean", "Std Dev", "Max", "Min", "Range",
    "Kurtosis", "Skewness", "Median", "Q1", "Q3",
    "IQR", "Sum Abs Diff", "RMS",
]


def extract_statistical_features(X):
    """Turn each 187-point heartbeat into 13 statistical features.

    X has shape (n_samples, 187); the result has shape (n_samples, 13).
    """
    X = np.asarray(X, dtype=np.float64)
    q1 = np.percentile(X, 25, axis=1)
    q3 = np.percentile(X, 75, axis=1)
    x_max = X.max(axis=1)
    x_min = X.min(axis=1)
    return np.column_stack([
        X.mean(axis=1),
        X.std(axis=1),
        x_max,
        x_min,
        x_max - x_min,
        stats.kurtosis(X, axis=1),
        stats.skew(X, axis=1),
        np.median(X, axis=1),
        q1,
        q3,
        q3 - q1,
        np.abs(np.diff(X, axis=1)).sum(axis=1),
        np.sqrt((X ** 2).mean(axis=1)),
    ])
