# RLT/Models/rlt.py

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier


class RLTBase(BaseEstimator):
    def __init__(
        self,
        n_estimators: int = 5,
        n_min: int = 5,
        muting: str = "none",
        k: int = 1,
        p_0: int = 10,
        max_depth: int | None = None,
        random_state: int | None = None,
    ):
        self.n_estimators = n_estimators
        self.n_min = n_min
        self.muting = muting
        self.k = k
        self.p_0 = p_0
        self.max_depth = max_depth
        self.random_state = random_state


class RLTRegressor(RLTBase, RegressorMixin):
    """
    Reinforcement Learning Trees - Zhu et al. (2015)
    """

    def _compute_muting_and_protection(self, importances_local, protected_indices, current_split_vars):
        """Calcule muting et protection selon l'article."""
        p_local = len(importances_local)
        
        if self.muting == "none":
            new_protected = protected_indices.union(set(current_split_vars))
            return np.ones(p_local, dtype=bool), new_protected
        
        if self.muting == "moderate":
            muting_rate = 0.5
        elif self.muting == "aggressive":
            muting_rate = 0.8
        else:
            return np.ones(p_local, dtype=bool), protected_indices
        
        # Protected set
        new_protected = protected_indices.union(set(current_split_vars))
        order_desc = np.argsort(-importances_local)
        p_0_eff = min(self.p_0, p_local)
        top_p0_indices = set(order_desc[:p_0_eff])
        new_protected = new_protected.union(top_p0_indices)
        
        # Eligible pour muting
        eligible_for_muting = [i for i in range(p_local) if i not in new_protected]
        
        if len(eligible_for_muting) == 0:
            return np.ones(p_local, dtype=bool), new_protected
        
        # Calculer p_d
        n_eligible = len(eligible_for_muting)
        p_d = int(np.ceil(muting_rate * n_eligible))
        p_d = min(p_d, max(0, n_eligible - 1))
        
        # Sécurité : toujours garder au moins 1 variable
        if p_d >= n_eligible:
            p_d = max(0, n_eligible - 1)
        
        # Muter les moins importantes
        eligible_importances = importances_local[eligible_for_muting]
        order_asc = np.argsort(eligible_importances)
        to_mute_in_eligible = order_asc[:p_d]
        to_mute_global = [eligible_for_muting[i] for i in to_mute_in_eligible]
        
        # Masque final
        keep_mask = np.ones(p_local, dtype=bool)
        keep_mask[to_mute_global] = False
        
        # Sécurité : toujours garder au moins k variables
        if keep_mask.sum() < max(1, self.k):
            keep_mask = np.zeros(p_local, dtype=bool)
            keep_mask[order_desc[:max(self.k, 1)]] = True
        
        return keep_mask, new_protected

    def _build_tree_reg(self, X, y, depth, rng, protected_indices=None):
        """Construction récursive d'un arbre avec muting."""
        n, p_local = X.shape
        
        if protected_indices is None:
            protected_indices = set()

        # Critères d'arrêt
        if n <= self.n_min or (self.max_depth is not None and depth >= self.max_depth):
            return {"is_leaf": True, "prediction": float(np.mean(y))}

        if p_local == 0:
            return {"is_leaf": True, "prediction": float(np.mean(y))}

        if np.allclose(y, y.mean()):
            return {"is_leaf": True, "prediction": float(np.mean(y))}

        # Modèle embarqué
        try:
            et = ExtraTreesRegressor(
                n_estimators=100,
                max_features="sqrt",
                random_state=int(rng.integers(0, 1_000_000)),
                n_jobs=1,
            )
            et.fit(X, y)
            importances_local = et.feature_importances_
        except Exception:
            return {"is_leaf": True, "prediction": float(np.mean(y))}

        if np.all(importances_local <= 0):
            return {"is_leaf": True, "prediction": float(np.mean(y))}

        # Sélection des variables pour le split
        order_desc = np.argsort(-importances_local)
        k_eff = min(self.k, len(order_desc), p_local)
        
        if k_eff == 0:
            return {"is_leaf": True, "prediction": float(np.mean(y))}
        
        selected_idx = order_desc[:k_eff]

        # Appliquer le muting pour les enfants
        keep_mask, new_protected = self._compute_muting_and_protection(
            importances_local, 
            protected_indices, 
            selected_idx
        )

        # Vérifier qu'on garde au moins une variable
        if keep_mask.sum() == 0:
            return {"is_leaf": True, "prediction": float(np.mean(y))}

        # Combinaison linéaire
        w = rng.normal(size=k_eff)
        w = w / (np.linalg.norm(w) + 1e-12)

        X_sel = X[:, selected_idx]
        proj = X_sel @ w

        if np.allclose(proj, proj.mean()):
            return {"is_leaf": True, "prediction": float(np.mean(y))}

        # Chercher le meilleur seuil
        unique_vals = np.unique(proj)
        if unique_vals.size <= 1:
            return {"is_leaf": True, "prediction": float(np.mean(y))}

        qs = np.linspace(0.1, 0.9, 9)
        candidates = np.unique(np.quantile(proj, qs))

        best_threshold = None
        best_mse = np.inf
        best_left_idx = None
        best_right_idx = None

        for thr in candidates:
            left_idx = proj <= thr
            right_idx = ~left_idx
            
            if left_idx.sum() < self.n_min or right_idx.sum() < self.n_min:
                continue
            
            y_left, y_right = y[left_idx], y[right_idx]
            
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            
            mse_left = np.var(y_left) if len(y_left) > 1 else 0
            mse_right = np.var(y_right) if len(y_right) > 1 else 0
            mse = (left_idx.sum() * mse_left + right_idx.sum() * mse_right) / n
            
            if mse < best_mse:
                best_mse = mse
                best_threshold = float(thr)
                best_left_idx = left_idx
                best_right_idx = right_idx

        if best_threshold is None:
            return {"is_leaf": True, "prediction": float(np.mean(y))}

        # Appliquer le muting aux données
        X_kept = X[:, keep_mask]
        
        if X_kept.shape[1] == 0:
            return {"is_leaf": True, "prediction": float(np.mean(y))}

        # Mapper les indices protégés vers le nouvel espace
        original_to_kept = {}
        kept_idx = 0
        for orig_idx in range(p_local):
            if keep_mask[orig_idx]:
                original_to_kept[orig_idx] = kept_idx
                kept_idx += 1
        
        new_protected_mapped = set()
        for orig_idx in new_protected:
            if orig_idx in original_to_kept:
                new_protected_mapped.add(original_to_kept[orig_idx])

        # Construire les enfants
        X_left_child = X_kept[best_left_idx]
        X_right_child = X_kept[best_right_idx]

        try:
            left_tree = self._build_tree_reg(
                X_left_child,
                y[best_left_idx],
                depth + 1,
                rng,
                protected_indices=new_protected_mapped,
            )
        except Exception:
            left_tree = {"is_leaf": True, "prediction": float(np.mean(y[best_left_idx]))}

        try:
            right_tree = self._build_tree_reg(
                X_right_child,
                y[best_right_idx],
                depth + 1,
                rng,
                protected_indices=new_protected_mapped,
            )
        except Exception:
            right_tree = {"is_leaf": True, "prediction": float(np.mean(y[best_right_idx]))}

        return {
            "is_leaf": False,
            "threshold": best_threshold,
            "weights": w,
            "features_local": selected_idx,
            "left": left_tree,
            "right": right_tree,
        }

    def _predict_tree_reg(self, tree, X):
        """Prédiction simplifiée."""
        n, p = X.shape
        preds = np.empty(n, dtype=float)

        def recurse(node, indices):
            if len(indices) == 0:
                return
                
            if node["is_leaf"]:
                preds[indices] = node["prediction"]
                return

            try:
                feats = node["features_local"]
                w = node["weights"]
                thr = node["threshold"]

                X_subset = X[indices]
                
                # Vérifier que les features existent
                if np.any(feats >= X_subset.shape[1]):
                    preds[indices] = node["prediction"] if "prediction" in node else np.mean(preds[preds != 0])
                    return
                
                X_sel = X_subset[:, feats]
                proj = X_sel @ w

                left_mask = proj <= thr
                right_mask = ~left_mask
                
                left_indices = indices[left_mask]
                right_indices = indices[right_mask]

                recurse(node["left"], left_indices)
                recurse(node["right"], right_indices)
            except Exception:
                # En cas d'erreur, utiliser la prédiction du nœud ou la moyenne
                preds[indices] = node.get("prediction", np.nanmean(preds[preds != 0]) if np.any(preds != 0) else 0)

        all_indices = np.arange(n)
        recurse(tree, all_indices)
        return preds

    def fit(self, X, y):
        """Entraînement de l'ensemble d'arbres RLT."""
        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        n, _ = X.shape

        for i in range(self.n_estimators):
            idx = rng.integers(0, n, size=n)
            X_boot = X[idx]
            y_boot = y[idx]
            
            try:
                tree = self._build_tree_reg(
                    X_boot,
                    y_boot,
                    depth=0,
                    rng=rng,
                    protected_indices=set(),
                )
                self.trees_.append(tree)
                
                if (i + 1) % 10 == 0 or i == 0:
                    print(f"[RLT] Arbre {i+1}/{self.n_estimators} construit")
            except Exception as e:
                print(f"[RLT] Arbre {i+1}/{self.n_estimators} échoué: {str(e)[:50]}")
                # Créer un arbre minimal
                self.trees_.append({"is_leaf": True, "prediction": float(np.mean(y_boot))})
        
        return self

    def predict(self, X):
        """Prédiction par moyenne des arbres."""
        all_preds = []
        for tree in self.trees_:
            try:
                pred = self._predict_tree_reg(tree, X)
                all_preds.append(pred)
            except Exception:
                # En cas d'erreur, utiliser la prédiction moyenne
                all_preds.append(np.full(X.shape[0], tree.get("prediction", 0)))
        
        all_preds = np.vstack(all_preds)
        return all_preds.mean(axis=0)


class RLTClassifier(BaseEstimator, ClassifierMixin):
    """
    Random Linear Tree Classifier - Ensemble classification with:
      - Embedded ExtraTrees for variable importances
      - Cumulative muting (none / moderate / aggressive)
      - Splits as linear combinations of k variables
    
    Parameters
    ----------
    n_estimators : int, default=50
        Number of trees in ensemble
    n_min : int, default=5
        Minimum samples to split a node
    muting : {'none', 'moderate', 'aggressive'}, default='moderate'
        Muting strategy for low-importance variables
    k : int, default=2
        Number of variables for linear combination splits
    p_0 : int, default=10
        Number of protected (never muted) variables
    max_depth : int or None, default=None
        Maximum tree depth
    random_state : int, default=42
        Random seed
    """
    
    def __init__(self, n_estimators=50, n_min=5, muting='moderate', 
                 k=2, p_0=10, max_depth=None, random_state=42):
        self.n_estimators = n_estimators
        self.n_min = n_min
        self.muting = muting
        self.k = k
        self.p_0 = p_0
        self.max_depth = max_depth
        self.random_state = random_state
    
    def fit(self, X, y):
        """Train the RLT classifier."""
        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        n, self.n_features_ = X.shape
        
        # Store unique classes
        self.classes_ = np.unique(y)
        
        # Store the most frequent class for fallback predictions
        _, counts = np.unique(y, return_counts=True)
        self.default_class_ = self.classes_[np.argmax(counts)]
        
        # Compute variable importances using ExtraTrees
        et = ExtraTreesClassifier(n_estimators=min(100, n), 
                                   max_depth=self.max_depth,
                                   random_state=self.random_state)
        et.fit(X, y)
        self.importances_ = et.feature_importances_
        
        # Build ensemble
        for i in range(self.n_estimators):
            idx = rng.integers(0, n, size=n)
            X_boot = X[idx]
            y_boot = y[idx]
            
            try:
                tree = self._build_tree_clf(
                    X_boot, y_boot,
                    depth=0, rng=rng,
                    protected_indices=set(),
                )
                self.trees_.append(tree)
                
                # Progress logging
                if (i + 1) % 10 == 0 or i == 0:
                    print(f"[RLT] Tree {i+1}/{self.n_estimators} built")
                    
            except Exception as e:
                print(f"[RLT] Tree {i+1}/{self.n_estimators} failed: {str(e)[:50]}")
                # Fallback: create leaf with majority class
                values, counts = np.unique(y_boot, return_counts=True)
                majority = values[np.argmax(counts)]
                self.trees_.append({"is_leaf": True, "prediction": majority})
        
        return self
    
    def _apply_muting(self, importances, protected_indices, rng):
        """Apply muting to low-importance variables."""
        n_vars = len(importances)
        muted = np.zeros(n_vars, dtype=bool)
        
        if self.muting == 'none':
            return muted
        
        # Sort by importance
        sorted_idx = np.argsort(importances)
        
        if self.muting == 'moderate':
            mute_rate = 0.5
        elif self.muting == 'aggressive':
            mute_rate = 0.8
        else:
            mute_rate = 0.0
        
        n_mute = max(1, int(n_vars * mute_rate))
        
        # Don't mute protected variables
        candidates = [idx for idx in sorted_idx if idx not in protected_indices]
        mute_indices = candidates[:n_mute]
        
        muted[mute_indices] = True
        return muted
    
    def _gini_impurity(self, y):
        """Calculate Gini impurity."""
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        proportions = counts / len(y)
        return 1 - np.sum(proportions ** 2)
    
    def _build_tree_clf(self, X, y, depth=0, rng=None, protected_indices=None):
        """Recursively build decision tree."""
        if protected_indices is None:
            protected_indices = set()
        
        n, p = X.shape
        
        # Stop conditions
        if len(np.unique(y)) == 1 or n < self.n_min or \
           (self.max_depth is not None and depth >= self.max_depth):
            values, counts = np.unique(y, return_counts=True)
            return {"is_leaf": True, "prediction": values[np.argmax(counts)]}
        
        # Apply muting
        muted = self._apply_muting(self.importances_, protected_indices, rng)
        available = ~muted
        
        if not np.any(available):
            values, counts = np.unique(y, return_counts=True)
            return {"is_leaf": True, "prediction": values[np.argmax(counts)]}
        
        # Find best split
        best_gain = -1
        best_split = None
        
        for _ in range(min(p, 10)):  # Try up to 10 random splits
            # Select k random available variables
            avail_idx = np.where(available)[0]
            if len(avail_idx) < self.k:
                k_use = len(avail_idx)
            else:
                k_use = self.k
            
            if k_use == 0:
                continue
            
            feats = rng.choice(avail_idx, size=k_use, replace=False)
            
            # Random weights
            w = rng.normal(0, 1, k_use)
            w = w / (np.linalg.norm(w) + 1e-12)
            
            # Compute projection
            X_proj = X[:, feats] @ w
            
            # Try multiple thresholds
            for threshold in np.percentile(X_proj, [25, 50, 75]):
                left_mask = X_proj <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < 1 or np.sum(right_mask) < 1:
                    continue
                
                y_left = y[left_mask]
                y_right = y[right_mask]
                
                # Information gain
                gini_parent = self._gini_impurity(y)
                gini_left = self._gini_impurity(y_left)
                gini_right = self._gini_impurity(y_right)
                
                gain = gini_parent - (np.sum(left_mask)/n * gini_left + 
                                      np.sum(right_mask)/n * gini_right)
                
                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        'feats': feats,
                        'w': w,
                        'threshold': threshold,
                        'left_mask': left_mask,
                        'right_mask': right_mask,
                    }
        
        # If no good split found, return leaf
        if best_split is None:
            values, counts = np.unique(y, return_counts=True)
            return {"is_leaf": True, "prediction": values[np.argmax(counts)]}
        
        # Recursively build subtrees
        y_left = y[best_split['left_mask']]
        y_right = y[best_split['right_mask']]
        
        try:
            left_tree = self._build_tree_clf(
                X[best_split['left_mask']], y_left,
                depth + 1, rng,
                protected_indices=protected_indices,
            )
        except Exception:
            values, counts = np.unique(y_left, return_counts=True)
            left_tree = {"is_leaf": True, "prediction": values[np.argmax(counts)]}
        
        try:
            right_tree = self._build_tree_clf(
                X[best_split['right_mask']], y_right,
                depth + 1, rng,
                protected_indices=protected_indices,
            )
        except Exception:
            values, counts = np.unique(y_right, return_counts=True)
            right_tree = {"is_leaf": True, "prediction": values[np.argmax(counts)]}
        
        return {
            "is_leaf": False,
            "features_local": best_split['feats'],
            "weights": best_split['w'],
            "threshold": best_split['threshold'],
            "left": left_tree,
            "right": right_tree,
        }
    
    def _predict_tree_clf(self, tree, X):
        """Predict using a single tree - returns valid class labels only."""
        n, _ = X.shape
        # Initialize with default class to ensure all predictions are valid
        preds = np.full(n, self.default_class_, dtype=self.classes_.dtype)
        
        def recurse(node, indices):
            if len(indices) == 0:
                return
            
            if node["is_leaf"]:
                preds[indices] = node["prediction"]
                return
            
            try:
                feats = node["features_local"]
                w = node["weights"]
                thr = node["threshold"]
                
                X_subset = X[indices]
                
                # Check if features are valid
                if np.any(feats >= X_subset.shape[1]):
                    # Use default class for invalid features
                    preds[indices] = self.default_class_
                    return
                
                # Compute projection
                X_sel = X_subset[:, feats]
                proj = X_sel @ w
                
                left_mask = proj <= thr
                right_mask = ~left_mask
                
                left_indices = indices[left_mask]
                right_indices = indices[right_mask]
                
                recurse(node["left"], left_indices)
                recurse(node["right"], right_indices)
            except Exception:
                # Use default class on error
                preds[indices] = self.default_class_
        
        all_indices = np.arange(n)
        recurse(tree, all_indices)
        
        return preds
    
    def predict(self, X):
        """Predict class for X - ensures all predictions are valid class labels."""
        # Get predictions from all trees
        all_preds = np.array([self._predict_tree_clf(tree, X) for tree in self.trees_])
        
        n = X.shape[0]
        final = np.empty(n, dtype=self.classes_.dtype)
        
        for i in range(n):
            predictions = all_preds[:, i]
            
            # All predictions should now be valid, but filter just in case
            valid_mask = np.isin(predictions, self.classes_)
            valid_predictions = predictions[valid_mask]
            
            if len(valid_predictions) > 0:
                vals, counts = np.unique(valid_predictions, return_counts=True)
                final[i] = vals[np.argmax(counts)]
            else:
                # Fallback if all predictions are invalid
                final[i] = self.default_class_
        
        return final
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        all_preds = np.array([self._predict_tree_clf(tree, X) for tree in self.trees_])
        
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        proba = np.zeros((n_samples, n_classes), dtype=float)
        
        # Calculate probabilities from voting
        for i in range(n_samples):
            predictions = all_preds[:, i]
            
            # Filter to valid classes only
            valid_mask = np.isin(predictions, self.classes_)
            valid_predictions = predictions[valid_mask]
            
            n_valid = len(valid_predictions)
            if n_valid > 0:
                for j, cls in enumerate(self.classes_):
                    votes = np.sum(valid_predictions == cls)
                    proba[i, j] = votes / n_valid
            else:
                # Default to uniform if all invalid
                proba[i, :] = 1.0 / n_classes
        
        return proba