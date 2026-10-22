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

    def __sklearn_is_fitted__(self):  # stateless
        return True

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True)
        return (X - mu) / np.where(sd == 0, 1.0, sd)


class SGDerivative(BaseEstimator, TransformerMixin):
    """Savitzky-Golay derivative (stateless). Suppresses broad background, keeps narrow bands."""

    def __init__(self, deriv: int = 1, window: int = 11, polyorder: int = 2):
        self.deriv = deriv
        self.window = window
        self.polyorder = polyorder

    def __sklearn_is_fitted__(self):  # stateless
        return True

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        from scipy.signal import savgol_filter

        return savgol_filter(
            np.asarray(X, dtype=np.float64), self.window, self.polyorder, deriv=self.deriv, axis=1
        )


class ASPLSBaseline(BaseEstimator, TransformerMixin):
    """Per-spectrum asPLS baseline subtraction via pybaselines (stateless)."""

    def __init__(self, wavenumbers: np.ndarray, lam: float = 1e5):
        self.wavenumbers = wavenumbers
        self.lam = lam

    def __sklearn_is_fitted__(self):  # stateless
        return True

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        from pybaselines import Baseline

        fitter = Baseline(x_data=np.asarray(self.wavenumbers))
        X = np.asarray(X, dtype=np.float64)
        return np.vstack([x - fitter.aspls(x, lam=self.lam)[0] for x in X])


class NuisanceProjector(BaseEstimator, TransformerMixin):
    """Project out a fixed nuisance subspace: x -> x (I - V V^T).  (EPO, Roger et al. 2003.)

    `basis` (k, p) must be estimated from TRAINING data only, e.g. with `session_difference_basis`.
    """

    def __init__(self, basis: np.ndarray):
        self.basis = basis

    def __sklearn_is_fitted__(self):  # stateless
        return True

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        V = np.asarray(self.basis)
        X = np.asarray(X, dtype=np.float64)
        return X - (X @ V.T) @ V


def session_difference_basis(Z: np.ndarray, meta, k: int = 3) -> np.ndarray:
    """EPO basis from units measured in more than one session (same block + face, different dates).

    Z: already-preprocessed training spectra; meta: matching rows. Returns (k, p) orthonormal rows,
    or an empty (0, p) basis if no repeated unit exists in the training data.
    """
    diffs = []
    for _, g in meta.reset_index(drop=True).groupby(["block_id", "face"]):
        dates = sorted(g.date.unique())
        if len(dates) < 2:
            continue
        means = [Z[g.index[g.date == d]].mean(0) for d in dates]
        diffs += [m - means[0] for m in means[1:]]
    if not diffs:
        return np.zeros((0, Z.shape[1]))
    _, _, vt = np.linalg.svd(np.vstack(diffs), full_matrices=False)
    return vt[:k]


def spike_score_2nd_diff(X: np.ndarray) -> np.ndarray:
    """Largest |2nd difference| / its median, per spectrum. Large values flag cosmic-ray spikes.

    Detect and report, never silently clip. Whole-spectrum outliers (saturation, fluorescence bursts)
    are a different failure and need a level-based check instead.
    """
    d = np.abs(np.diff(np.asarray(X, dtype=np.float64), n=2, axis=1))
    med = np.median(d, axis=1)
    return d.max(axis=1) / np.where(med == 0, 1e-9, med)
