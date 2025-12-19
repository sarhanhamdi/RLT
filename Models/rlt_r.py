# RLT/Models/rlt_r.py
"""
Wrapper for the official R RLT package using rpy2 (PRODUCTION - FIXED).

This module provides a scikit-learn compatible interface to call the R RLT
(Reinforcement Learning Trees) package from Zhu, Zeng & Kosorok (2015).

Reference: Zhu, R., Zeng, D., & Kosorok, M. R. (2015)
"Reinforcement Learning Trees."
Journal of the American Statistical Association. 110(512), 1770-1784.

Installation Requirements:
    1. Python: pip install rpy2
    2. R: install.packages("RLT")

Usage:
    >>> from Models.rlt_r import RLTRegressorR, RLTClassifierR
    >>> model = RLTRegressorR(n_estimators=100)
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
"""

import warnings
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

try:
    import rpy2
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter
    
    # NEW (rpy2 2.13+): Import converters properly
    from rpy2.robjects import numpy2ri, pandas2ri
    
    # Try to import RLT package
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


class RLTRegressorR(BaseEstimator, RegressorMixin):
    """
    Scikit-learn wrapper for R's RLT (Reinforcement Learning Trees) for regression.
    
    This class calls the RLT::RLT() function from the official R package.
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest (ntrees in R).
    
    mtry : int or None, default=None
        Number of features to consider at each split.
        If None (default), uses mtry = p/3 (following RLT defaults).
    
    n_min : int, default=1
        Minimum node size for splitting.
        Smaller values lead to deeper trees.
    
    replace : bool, default=True
        Whether to use bootstrap sampling (replacement).
    
    split_gen : str, default="random"
        Split generation method ("random" or "best").
    
    random_state : int or None, default=None
        Random seed for reproducibility.
    
    verbose : bool, default=False
        Whether to print progress information.
    
    Attributes
    ----------
    fit_ : object
        The fitted R RLT model object.
    
    n_features_ : int
        Number of features seen during fit.
    
    Examples
    --------
    >>> from Models.rlt_r import RLTRegressorR
    >>> import numpy as np
    >>> X = np.random.randn(100, 20)
    >>> y = X[:, 0] + 2 * X[:, 1] + np.random.randn(100) * 0.1
    >>> model = RLTRegressorR(n_estimators=50, random_state=42)
    >>> model.fit(X, y)
    >>> preds = model.predict(X[:10])
    """
    
    def __init__(
        self,
        n_estimators=100,
        mtry=None,
        n_min=1,
        replace=True,
        split_gen="random",
        random_state=None,
        verbose=False,
    ):
        self.n_estimators = n_estimators
        self.mtry = mtry
        self.n_min = n_min
        self.replace = replace
        self.split_gen = split_gen
        self.random_state = random_state
        self.verbose = verbose
    
    def fit(self, X, y):
        """
        Fit the RLT regression model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        
        y : array-like of shape (n_samples,)
            Target values.
        
        Returns
        -------
        self : object
            Returns self for method chaining.
        """
        if not RLT_AVAILABLE:
            raise ImportError(
                "R 'RLT' package not available. "
                "Install it with: rpy2 (pip install rpy2) and RLT (install.packages('RLT') in R)"
            )
        
        # Validate input
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)
        self.n_features_ = X.shape[1]
        
        # Set default mtry if not provided
        if self.mtry is None:
            mtry = max(1, int(self.n_features_ / 3))
        else:
            mtry = self.mtry
        
        # Convert data to R objects inside context manager
        with localconverter(ro.default_converter + numpy2ri.converter + pandas2ri.converter):
            X_r = ro.conversion.py2rpy(X)
            y_r = ro.conversion.py2rpy(y)
        
        # Build parameter list for R function (outside context to keep R class)
        params = {
            'x': X_r,
            'y': y_r,
            'ntrees': self.n_estimators,
            'mtry': mtry,
            'nmin': self.n_min,
            'replace': self.replace,
            'split.gen': self.split_gen,
        }
        
        # Add random seed if specified
        if self.random_state is not None:
            ro.r['set.seed'](self.random_state)
        
        # Call R's RLT function (outside context - keeps R class!)
        if self.verbose:
            print(f"[RLT] Fitting regression model with {self.n_estimators} trees...")
        
        try:
            self.fit_ = rlt_r.RLT(**params)
        except Exception as e:
            raise RuntimeError(f"Error fitting RLT model in R: {str(e)}")
        
        return self
    
    def predict(self, X):
        """
        Predict using the RLT model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        check_is_fitted(self, 'fit_')
        X = check_array(X, accept_sparse=False)
        
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {X.shape[1]} features but model expects {self.n_features_}"
            )
        
        # Convert test data to R objects inside context manager
        with localconverter(ro.default_converter + numpy2ri.converter + pandas2ri.converter):
            X_r = ro.conversion.py2rpy(X)
        
        # Call predict - uses R's S3 generic predict method (outside context)
        try:
            pred_r = ro.r.predict(self.fit_, X_r)
            # Extract the Prediction component from the result
            # predict.RLT returns a list with $Prediction component
            with localconverter(ro.default_converter + numpy2ri.converter + pandas2ri.converter):
                predictions = np.array(ro.conversion.rpy2py(pred_r.rx2('Prediction')))
        except Exception as e:
            raise RuntimeError(f"Error during prediction: {str(e)}")
        
        return predictions


class RLTClassifierR(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn wrapper for R's RLT (Reinforcement Learning Trees) for classification.
    
    This class calls the RLT::RLT() function from the official R package.
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest (ntrees in R).
    
    mtry : int or None, default=None
        Number of features to consider at each split.
        If None (default), uses mtry = p/3.
    
    n_min : int, default=1
        Minimum node size for splitting.
    
    replace : bool, default=True
        Whether to use bootstrap sampling.
    
    split_gen : str, default="random"
        Split generation method.
    
    random_state : int or None, default=None
        Random seed.
    
    verbose : bool, default=False
        Print progress.
    
    Attributes
    ----------
    fit_ : object
        The fitted R RLT model object.
    
    classes_ : ndarray
        Unique class labels.
    
    n_features_ : int
        Number of features seen during fit.
    
    Examples
    --------
    >>> from Models.rlt_r import RLTClassifierR
    >>> import numpy as np
    >>> X = np.random.randn(100, 20)
    >>> y = (X[:, 0] > 0).astype(int)
    >>> model = RLTClassifierR(n_estimators=50, random_state=42)
    >>> model.fit(X, y)
    >>> preds = model.predict(X[:10])
    """
    
    def __init__(
        self,
        n_estimators=100,
        mtry=None,
        n_min=1,
        replace=True,
        split_gen="random",
        random_state=None,
        verbose=False,
    ):
        self.n_estimators = n_estimators
        self.mtry = mtry
        self.n_min = n_min
        self.replace = replace
        self.split_gen = split_gen
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
        
        # Convert class labels to 0-indexed integers if needed
        y_indexed = np.searchsorted(self.classes_, y).astype(float)
        
        if self.mtry is None:
            mtry = max(1, int(self.n_features_ / 3))
        else:
            mtry = self.mtry
        
        # Convert data to R objects inside context manager
        with localconverter(ro.default_converter + numpy2ri.converter + pandas2ri.converter):
            X_r = ro.conversion.py2rpy(X)
            y_r = ro.conversion.py2rpy(y_indexed)
        
        # Build parameter list (outside context to keep R class)
        params = {
            'x': X_r,
            'y': y_r,
            'ntrees': self.n_estimators,
            'mtry': mtry,
            'nmin': self.n_min,
            'replace': self.replace,
            'split.gen': self.split_gen,
        }
        
        # Add random seed if specified
        if self.random_state is not None:
            ro.r['set.seed'](self.random_state)
        
        if self.verbose:
            print(f"[RLT] Fitting classification model with {self.n_estimators} trees...")
        
        try:
            self.fit_ = rlt_r.RLT(**params)
        except Exception as e:
            raise RuntimeError(f"Error fitting RLT model in R: {str(e)}")
        
        return self
    
    def predict(self, X):
        """
        Predict class labels.
        
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
                f"X has {X.shape[1]} features but model expects {self.n_features_}"
            )
        
        # Convert test data to R objects inside context manager
        with localconverter(ro.default_converter + numpy2ri.converter + pandas2ri.converter):
            X_r = ro.conversion.py2rpy(X)
        
        try:
            pred_r = ro.r.predict(self.fit_, X_r)
            # Extract the Prediction component
            with localconverter(ro.default_converter + numpy2ri.converter + pandas2ri.converter):
                predictions_indexed = np.array(ro.conversion.rpy2py(pred_r.rx2('Prediction'))).astype(int)
            # Convert back to original class labels
            predictions = self.classes_[predictions_indexed]
        except Exception as e:
            raise RuntimeError(f"Error during prediction: {str(e)}")
        
        return predictions


# ============================================================================
# Convenience function to check if RLT is available
# ============================================================================

def check_rlt_installed():
    """
    Check if R and the RLT package are properly installed and accessible.
    
    Returns
    -------
    bool
        True if RLT is available, False otherwise.
    
    Examples
    --------
    >>> from Models.rlt_r import check_rlt_installed
    >>> if check_rlt_installed():
    ...     from Models.rlt_r import RLTRegressorR
    ...     model = RLTRegressorR()
    """
    if not RLT_AVAILABLE:
        print("[WARNING] RLT not available.")
        if rpy2 is None:
            print("  - Install rpy2: pip install rpy2")
        else:
            print("  - rpy2 is installed, but R 'RLT' package is missing")
            print("  - Install RLT in R: install.packages('RLT')")
        return False
    else:
        print("[OK] RLT is available!")
        return True