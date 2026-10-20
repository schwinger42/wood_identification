"""Spectral preprocessing as sklearn transformers, so everything fitted stays inside the CV fold."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Crop(BaseEstimator, TransformerMixin):
    """Keep one or more Raman-shift windows, e.g. [(400, 1800)] for the fingerprint region."""

    def __init__(self, wavenumbers: np.ndarray, windows: list[tuple[float, float]]):
        self.wavenumbers = wavenumbers
        self.windows = windows

    def fit(self, X, y=None):
        wn = np.asarray(self.wavenumbers)
        mask = np.zeros_like(wn, dtype=bool)
        for lo, hi in self.windows:
            mask |= (wn >= lo) & (wn <= hi)
        if not mask.any():
            raise ValueError(f"No points inside windows {self.windows}")
        self.mask_ = mask
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.mask_]


class SNV(BaseEstimator, TransformerMixin):
    """Standard normal variate: per-spectrum centring and scaling (stateless)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True)
        return (X - mu) / np.where(sd == 0, 1.0, sd)


def spike_score_2nd_diff(X: np.ndarray) -> np.ndarray:
    """Largest |2nd difference| / its median, per spectrum. Large values flag cosmic-ray spikes.

    Detect and report, never silently clip. Whole-spectrum outliers (saturation, fluorescence bursts)
    are a different failure and need a level-based check instead.
    """
    d = np.abs(np.diff(np.asarray(X, dtype=np.float64), n=2, axis=1))
    med = np.median(d, axis=1)
    return d.max(axis=1) / np.where(med == 0, 1e-9, med)
