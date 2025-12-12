# rlt_python.py
# Implémentation Python d'un modèle RLT-like (Reinforcement Learning Trees)
# Compatible sklearn : RLTClassifier et RLTRegressor

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import unique_labels


# ---------- Fonctions de critère (gini / mse) ----------

def _gini_impurity(y):
    m = len(y)
    if m == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / m
    return 1.0 - np.sum(p ** 2)


def _mse(y):
    m = len(y)
    if m == 0:
        return 0.0
    mean = np.mean(y)
    return np.mean((y - mean) ** 2)


# ---------- Classe interne : un arbre avec splits (feature + threshold OU linéaire) ----------

class _RLTNode:
    __slots__ = (
        "is_leaf",
        "feature",      # int ou None
        "threshold",    # float
        "left",         # int index enfant
        "right",        # int index enfant
        "weight",       # vecteur pour combsplit (ou None)
        "value",        # prédiction au noeud feuille
    )

    def __init__(self):
        self.is_leaf = True
        self.feature = None
        self.threshold = None
        self.left = -1
        self.right = -1
        self.weight = None
        self.value = None


class _RLTEnsemble:
    """
    Ensemble d'arbres RLT (interne). Gère classification OU regression
    """

    def __init__(
        self,
        task="classification",
        ntrees=100,
        max_depth=None,
        min_samples_leaf=5,
        mtry=None,
        combsplit=3,
        n_linear_splits=10,
        reinforcement=True,
        muting_percent=0.4,
        random_state=None,
    ):
        self.task = task
        self.ntrees = ntrees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.mtry = mtry
        self.combsplit = max(1, int(combsplit))
        self.n_linear_splits = n_linear_splits
        self.reinforcement = reinforcement
        self.muting_percent = muting_percent
        self.random_state = check_random_state(random_state)

        self.trees_ = []  # liste de (nodes, feature_importances_)
        self.n_features_ = None
        self.classes_ = None

    # ----- fonctions utilitaires -----

    def _criterion(self, y):
        if self.task == "classification":
            return _gini_impurity(y)
        else:
            return _mse(y)

    def _leaf_value(self, y):
        if self.task == "classification":
            # prédiction = classe majoritaire
            values, counts = np.unique(y, return_counts=True)
            return values[np.argmax(counts)]
        else:
            return float(np.mean(y))

    # ----- split axis-aligned -----

    def _best_axis_split(self, X, y, feat_idx):
        """Meilleur split pour une seule feature, retourne (gain, threshold)."""
        x = X[:, feat_idx]
        # tri par x
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]

        m = len(y)
        if m < 2 * self.min_samples_leaf:
            return -np.inf, None

        # pré-computes
        if self.task == "classification":
            base_imp = _gini_impurity(y_sorted)
        else:
            base_imp = _mse(y_sorted)

        best_gain = -np.inf
        best_thr = None

        # on teste des coupures entre valeurs distinctes
        for i in range(self.min_samples_leaf, m - self.min_samples_leaf):
            if x_sorted[i] == x_sorted[i - 1]:
                continue
            left = y_sorted[:i]
            right = y_sorted[i:]
            if self.task == "classification":
                imp_left = _gini_impurity(left)
                imp_right = _gini_impurity(right)
            else:
                imp_left = _mse(left)
                imp_right = _mse(right)
            gain = base_imp - (
                len(left) / m * imp_left + len(right) / m * imp_right
            )
            if gain > best_gain:
                best_gain = gain
                best_thr = (x_sorted[i] + x_sorted[i - 1]) / 2.0

        return best_gain, best_thr

    # ----- split linéaire (combsplit) -----

    def _best_linear_split(self, X, y, feature_indices):
        """
        Teste plusieurs directions linéaires aléatoires (combsplit)
        sur un sous-ensemble de features.
        """
        m, p = X.shape
        base_imp = self._criterion(y)

        best_gain = -np.inf
        best_w = None
        best_thr = None

        subX = X[:, feature_indices]

        for _ in range(self.n_linear_splits):
            # vecteur aléatoire
            w = self.random_state.normal(size=len(feature_indices))
            w = w / (np.linalg.norm(w) + 1e-12)

            z = subX @ w
            order = np.argsort(z)
            z_sorted = z[order]
            y_sorted = y[order]

            if m < 2 * self.min_samples_leaf:
                continue

            for i in range(self.min_samples_leaf, m - self.min_samples_leaf):
                if z_sorted[i] == z_sorted[i - 1]:
                    continue
                left = y_sorted[:i]
                right = y_sorted[i:]

                imp_left = self._criterion(left)
                imp_right = self._criterion(right)

                gain = base_imp - (
                    len(left) / m * imp_left + len(right) / m * imp_right
                )

                if gain > best_gain:
                    best_gain = gain
                    best_thr = (z_sorted[i] + z_sorted[i - 1]) / 2.0
                    best_w = np.zeros(p)
                    best_w[feature_indices] = w

        return best_gain, best_thr, best_w

    # ----- construction d'un arbre -----

    def _build_tree(self, X, y, depth, feature_weights):
        nodes = []

        def _grow(node_idx, X_node, y_node, depth):
            node = _RLTNode()
            nodes.append(node)

            # conditions d'arrêt
            if (
                len(y_node) < 2 * self.min_samples_leaf
                or (self.max_depth is not None and depth >= self.max_depth)
                or self._criterion(y_node) == 0.0
            ):
                node.is_leaf = True
                node.value = self._leaf_value(y_node)
                return

            m, p = X_node.shape

            # sélection de variables (mtry) avec renforcement
            if self.mtry is None:
                mtry = max(1, int(np.sqrt(p)))
            else:
                mtry = min(p, self.mtry)

            # distribution de probas pour le renforcement
            if self.reinforcement and feature_weights is not None:
                probs = np.maximum(feature_weights, 1e-9)
                probs = probs / probs.sum()
                feat_subset = self.random_state.choice(
                    p, size=mtry, replace=False, p=probs
                )
            else:
                feat_subset = self.random_state.choice(p, size=mtry, replace=False)

            # 1) meilleur split axis-aligned
            best_gain = -np.inf
            best_feat = None
            best_thr = None
            best_weight = None

            for f in feat_subset:
                gain_f, thr_f = self._best_axis_split(X_node, y_node, f)
                if gain_f > best_gain:
                    best_gain = gain_f
                    best_feat = f
                    best_thr = thr_f
                    best_weight = None

            # 2) meilleur split linéaire (combsplit)
            if self.combsplit > 1 and len(feat_subset) >= self.combsplit:
                # on prend un sous-ensemble pour combsplit
                comb_feats = self.random_state.choice(
                    feat_subset, size=self.combsplit, replace=False
                )
                gain_l, thr_l, w_l = self._best_linear_split(X_node, y_node, comb_feats)
                if gain_l > best_gain:
                    best_gain = gain_l
                    best_feat = None  # linéaire
                    best_thr = thr_l
                    best_weight = w_l

            # si aucun gain positif => feuille
            if best_gain <= 0 or best_thr is None:
                node.is_leaf = True
                node.value = self._leaf_value(y_node)
                return

            # split définitif
            node.is_leaf = False
            node.threshold = best_thr
            node.feature = best_feat
            node.weight = best_weight

            # découpe des données
            if best_weight is None:
                mask_left = X_node[:, best_feat] <= best_thr
            else:
                proj = X_node @ best_weight
                mask_left = proj <= best_thr

            X_left, y_left = X_node[mask_left], y_node[mask_left]
            X_right, y_right = X_node[~mask_left], y_node[~mask_left]

            if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                node.is_leaf = True
                node.value = self._leaf_value(y_node)
                return

            # indices enfants
            node.left = len(nodes)
            _grow(node.left, X_left, y_left, depth + 1)

            node.right = len(nodes)
            _grow(node.right, X_right, y_right, depth + 1)

            # importance des variables utilisées dans ce noeud
            if best_weight is None:
                used = [best_feat]
                contrib = 1.0
            else:
                used = np.flatnonzero(np.abs(best_weight) > 1e-9)
                contrib = 1.0 / len(used)

            return used, contrib

        # initialisation des poids
        used, contrib = _grow(0, X, y, depth)
        return nodes

    # ----- fit / predict -----

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        self.n_features_ = n_features

        if self.task == "classification":
            self.classes_ = unique_labels(y)

        feature_weights = np.ones(n_features, dtype=float)

        self.trees_ = []
        for t in range(self.ntrees):
            # bootstrap
            idx = self.random_state.randint(0, n_samples, size=n_samples)
            Xb = X[idx]
            yb = y[idx]

            # build tree
            nodes = self._build_tree(Xb, yb, depth=0, feature_weights=feature_weights)

            # pas de calcul complexe d'importance : on renforce uniformément
            if self.reinforcement:
                # "muting" simple : on baisse un peu toutes les features,
                # puis on remonte celles utilisées dans l'arbre
                feature_weights *= (1.0 - self.muting_percent)
                feature_weights = np.clip(feature_weights, 1e-4, None)
                # on renforce toutes les features (approximation)
                feature_weights += 1.0

            self.trees_.append(nodes)

        return self

    def _predict_tree(self, nodes, x):
        idx = 0
        while True:
            node = nodes[idx]
            if node.is_leaf:
                return node.value
            if node.weight is None:
                if x[node.feature] <= node.threshold:
                    idx = node.left
                else:
                    idx = node.right
            else:
                val = x @ node.weight
                if val <= node.threshold:
                    idx = node.left
                else:
                    idx = node.right

    def predict(self, X):
        X = np.asarray(X)
        n_samples = X.shape[0]
        preds = []

        if self.task == "classification":
            for i in range(n_samples):
                votes = []
                for nodes in self.trees_:
                    v = self._predict_tree(nodes, X[i])
                    votes.append(v)
                # majority vote
                values, counts = np.unique(votes, return_counts=True)
                preds.append(values[np.argmax(counts)])
            return np.array(preds)
        else:
            for i in range(n_samples):
                vals = []
                for nodes in self.trees_:
                    vals.append(self._predict_tree(nodes, X[i]))
                preds.append(np.mean(vals))
            return np.array(preds)


# ---------- Wrappers sklearn ----------

class RLTClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        ntrees=100,
        max_depth=None,
        min_samples_leaf=5,
        mtry=None,
        combsplit=3,
        n_linear_splits=10,
        reinforcement=True,
        muting_percent=0.4,
        random_state=None,
    ):
        self.ntrees = ntrees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.mtry = mtry
        self.combsplit = combsplit
        self.n_linear_splits = n_linear_splits
        self.reinforcement = reinforcement
        self.muting_percent = muting_percent
        self.random_state = random_state

    def fit(self, X, y):
        self.ensemble_ = _RLTEnsemble(
            task="classification",
            ntrees=self.ntrees,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            mtry=self.mtry,
            combsplit=self.combsplit,
            n_linear_splits=self.n_linear_splits,
            reinforcement=self.reinforcement,
            muting_percent=self.muting_percent,
            random_state=self.random_state,
        )
        self.ensemble_.fit(X, y)
        self.classes_ = self.ensemble_.classes_
        return self

    def predict(self, X):
        return self.ensemble_.predict(X)


class RLTRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        ntrees=100,
        max_depth=None,
        min_samples_leaf=5,
        mtry=None,
        combsplit=3,
        n_linear_splits=10,
        reinforcement=True,
        muting_percent=0.4,
        random_state=None,
    ):
        self.ntrees = ntrees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.mtry = mtry
        self.combsplit = combsplit
        self.n_linear_splits = n_linear_splits
        self.reinforcement = reinforcement
        self.muting_percent = muting_percent
        self.random_state = random_state

    def fit(self, X, y):
        self.ensemble_ = _RLTEnsemble(
            task="regression",
            ntrees=self.ntrees,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            mtry=self.mtry,
            combsplit=self.combsplit,
            n_linear_splits=self.n_linear_splits,
            reinforcement=self.reinforcement,
            muting_percent=self.muting_percent,
            random_state=self.random_state,
        )
        self.ensemble_.fit(X, y)
        return self

    def predict(self, X):
        return self.ensemble_.predict(X)
