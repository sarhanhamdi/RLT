# RLT/Models/rlt.py

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

class RLTBase(BaseEstimator):
    def __init__(self,
                 n_estimators=100,
                 n_min=5,
                 muting="moderate",   # "none", "moderate", "aggressive"
                 k=1,                  # nb max de variables dans la combinaison linéaire
                 random_state=None):
        self.n_estimators = n_estimators
        self.n_min = n_min
        self.muting = muting
        self.k = k
        self.random_state = random_state

    def _fit_single_tree(self, X, y):
        # Ici on mettra l’algorithme RLT complet (sections 2.3–2.7 de l’article)
        # Pour démarrer : on stocke juste la moyenne.
        return {
            "prediction": np.mean(y)
        }

class RLTRegressor(RLTBase, RegressorMixin):
    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        n, _ = X.shape
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, size=n)
            tree = self._fit_single_tree(X[idx], y[idx])
            self.trees_.append(tree)
        return self

    def predict(self, X):
        preds = np.array([t["prediction"] for t in self.trees_])
        return np.repeat(preds.mean(), X.shape[0])

class RLTClassifier(RLTBase, ClassifierMixin):
    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        n, _ = X.shape
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, size=n)
            tree = self._fit_single_tree(X[idx], y[idx])
            self.trees_.append(tree)
        return self

    def predict_proba(self, X):
        preds = np.array([t["prediction"] for t in self.trees_])
        p = 1 / (1 + np.exp(-preds.mean()))
        proba = np.tile([1 - p, p], (X.shape[0], 1))
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
