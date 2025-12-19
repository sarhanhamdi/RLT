# Models/rlt_classifier_advanced.py
"""
Advanced RLTClassifierR with muting, linear combinations, and reinforcement learning.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

try:
    import rpy2
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects import numpy2ri, pandas2ri
    
    try:
        rlt_r = importr('RLT')
        RLT_AVAILABLE = True
    except ImportError:
        RLT_AVAILABLE = False
        rlt_r = None
        
except ImportError:
    RLT_AVAILABLE = False
    rpy2 = None
    rlt_r = None
    ro = None
    numpy2ri = None
    pandas2ri = None


class RLTClassifierRAdvanced(BaseEstimator, ClassifierMixin):
    """
    Advanced RLTClassifierR with muting, linear combinations, and reinforcement learning.
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees.
    
    mtry : int or None, default=None
        Features to try (None = p/3).
    
    n_min : int, default=1
        Min node size.
    
    replace : bool, default=True
        Bootstrap with replacement.
    
    split_gen : str, default="random"
        Split method ("random" or "best").
    
    combsplit : int, default=1
        Linear combination splits:
        - 1 = binary splits (default)
        - 2+ = linear combinations with that many variables
    
    muting : int, default=-1
        Muting method:
        - -1 = mute by proportion (recommended)
        - >0 = mute by count
    
    muting_percent : float, default=0.5
        Proportion of variables to mute (0-1).
        Only used if muting=-1.
    
    reinforcement : bool, default=False
        Use reinforcement learning for better feature selection.
    
    importance : bool, default=True
        Calculate variable importance.
    
    protect : int or None, default=None
        Number of protected variables (not muted).
        None = log(p).
    
    random_state : int or None, default=None
        Random seed.
    
    verbose : bool, default=False
        Print progress.
    
    Attributes
    ----------
    fit_ : object
        Fitted R RLT model.
    
    classes_ : ndarray
        Unique class labels.
    
    n_features_ : int
        Number of features.
    
    variable_importance_ : ndarray, optional
        Variable importance scores.
    
    Examples
    --------
    >>> model = RLTClassifierRAdvanced(
    ...     n_estimators=100,
    ...     combsplit=3,
    ...     muting=-1,
    ...     muting_percent=0.5,
    ...     reinforcement=True
    ... )
    >>> model.fit(X, y)
    >>> pred = model.predict(X)
    """
    
    def __init__(
        self,
        n_estimators=100,
        mtry=None,
        n_min=1,
        replace=True,
        split_gen="random",
        combsplit=1,
        muting=-1,
        muting_percent=0.5,
        reinforcement=False,
        importance=True,
        protect=None,
        random_state=None,
        verbose=False,
    ):
        self.n_estimators = n_estimators
        self.mtry = mtry
        self.n_min = n_min
        self.replace = replace
        self.split_gen = split_gen
        self.combsplit = combsplit
        self.muting = muting
        self.muting_percent = muting_percent
        self.reinforcement = reinforcement
        self.importance = importance
        self.protect = protect
        self.random_state = random_state
        self.verbose = verbose
    
    def fit(self, X, y):
        """
        Fit the RLT classification model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        
        y : array-like of shape (n_samples,)
            Target class labels.
        
        Returns
        -------
        self : object
            Returns self.
        """
        if not RLT_AVAILABLE:
            raise ImportError(
                "R 'RLT' package not available. "
                "Install: pip install rpy2 && R: install.packages('RLT')"
            )
        
        X, y = check_X_y(X, y, accept_sparse=False)
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        
        # Convert class labels to 0-indexed integers
        y_indexed = np.searchsorted(self.classes_, y).astype(float)
        
        # Set defaults
        if self.mtry is None:
            mtry = max(1, int(self.n_features_ / 3))
        else:
            mtry = self.mtry
        
        if self.protect is None:
            protect = max(1, int(np.log(self.n_features_)))
        else:
            protect = self.protect
        
        # Convert data to R objects
        with localconverter(ro.default_converter + numpy2ri.converter + pandas2ri.converter):
            X_r = ro.conversion.py2rpy(X)
            y_r = ro.conversion.py2rpy(y_indexed)
        
        # Build R parameters
        params = {
            'x': X_r,
            'y': y_r,
            'ntrees': self.n_estimators,
            'mtry': mtry,
            'nmin': self.n_min,
            'replace': self.replace,
            'split.gen': self.split_gen,
            'combsplit': self.combsplit,
            'muting': self.muting,
            'muting.percent': self.muting_percent,
            'reinforcement': self.reinforcement,
            'importance': self.importance,
            'protect': protect,
        }
        
        # Set random seed
        if self.random_state is not None:
            ro.r['set.seed'](self.random_state)
        
        if self.verbose:
            print(f"[RLT] Training with:")
            print(f"  - n_estimators: {self.n_estimators}")
            print(f"  - combsplit: {self.combsplit}")
            print(f"  - muting: {self.muting} (percent: {self.muting_percent})")
            print(f"  - reinforcement: {self.reinforcement}")
            print(f"  - protect: {protect}")
        
        try:
            self.fit_ = rlt_r.RLT(**params)
            
            # Extract variable importance if available
            if self.importance:
                try:
                    vi_r = self.fit_.rx2('VarImp')
                    if vi_r is not None:
                        with localconverter(ro.default_converter + numpy2ri.converter):
                            self.variable_importance_ = np.array(ro.conversion.rpy2py(vi_r))
                    else:
                        self.variable_importance_ = None
                except:
                    self.variable_importance_ = None
        except Exception as e:
            raise RuntimeError(f"Error fitting RLT: {str(e)}")
        
        return self
    
    def predict(self, X):
        """
        Make predictions.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self, 'fit_')
        X = check_array(X, accept_sparse=False)
        
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_}"
            )
        
        # Convert test data
        with localconverter(ro.default_converter + numpy2ri.converter + pandas2ri.converter):
            X_r = ro.conversion.py2rpy(X)
        
        # Make predictions
        try:
            pred_r = ro.r.predict(self.fit_, X_r)
            with localconverter(ro.default_converter + numpy2ri.converter + pandas2ri.converter):
                pred_idx = np.array(ro.conversion.rpy2py(pred_r.rx2('Prediction'))).astype(int)
            predictions = self.classes_[pred_idx]
        except Exception as e:
            raise RuntimeError(f"Error during prediction: {str(e)}")
        
        return predictions
    
    def get_feature_importance(self):
        """
        Get variable importance scores.
        
        Returns
        -------
        importance : ndarray or None
            Variable importance scores, or None if not available.
        """
        check_is_fitted(self, 'fit_')
        if hasattr(self, 'variable_importance_'):
            return self.variable_importance_
        return None
