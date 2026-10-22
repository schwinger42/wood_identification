"""Baseline classifiers. Every DL model must be compared against these under the same split."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


class PLSDA(BaseEstimator, ClassifierMixin):
    """PLS-DA: PLS2 regression on one-hot labels, predict argmax."""

    def __init__(self, n_components: int = 10):
        self.n_components = n_components

    def fit(self, X, y):
        self.classes_, yi = np.unique(y, return_inverse=True)
        Y = np.eye(len(self.classes_))[yi]
        n = min(self.n_components, X.shape[0] - 1, X.shape[1])
        self.pls_ = PLSRegression(n_components=n, scale=False).fit(X, Y)
        return self

    def predict(self, X):
        return self.classes_[self.pls_.predict(X).argmax(1)]


def baseline_classifiers(seed: int = 0) -> dict:
    """Fixed, untuned defaults (4 held-out trees cannot support hyperparameter tuning)."""
    return {
        "PCA-LDA": lambda: make_pipeline(PCA(20, svd_solver="full"), LinearDiscriminantAnalysis()),
        "PLS-DA": lambda: PLSDA(10),
        "LinearSVM": lambda: make_pipeline(StandardScaler(), LinearSVC(C=0.01, max_iter=20000)),
        "LogReg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000)),
        "RandomForest": lambda: RandomForestClassifier(500, random_state=seed, n_jobs=-1),
        "MLP": lambda: make_pipeline(
            StandardScaler(),
            MLPClassifier((128, 32), alpha=1e-2, early_stopping=True, max_iter=500, random_state=seed),
        ),
    }
