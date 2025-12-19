import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from utils_cleaning import fix_winequality


def prepare_regression(df, target, dataset_name):
    """
    Main regression preparation function.
    Returns:
        X_skl : numpy array for sklearn models
        y     : numpy array target
        X_rlt : numpy array for RLT python
        y_rlt : numpy array target for RLT python
    """

    # ============================================
    # 1. Fix winequality datasets BEFORE anything
    # ============================================
    if "winequality" in dataset_name.lower():
        csv_path = os.path.join("Data", dataset_name + ".csv")
        df = fix_winequality(csv_path)

    # ============================================
    # 2. Drop rows where target is missing
    # ============================================
    df = df.dropna(subset=[target])

    # ============================================
    # 3. Separate features & target
    # ============================================
    y = df[target].astype(float).values
    X = df.drop(columns=[target])

    # ============================================
    # 4. Convert columns to numeric if possible
    # ============================================
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # ============================================
    # 5. Remove columns that are completely NaN
    # ============================================
    X = X.dropna(axis=1, how="all")

    # ============================================
    # 6. Impute missing values (median)
    # ============================================
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    # ============================================
    # 7. Scale features for sklearn models
    # ============================================
    scaler = StandardScaler()
    X_skl = scaler.fit_transform(X_imputed)

    # ============================================
    # 8. Prepare data for RLT (NO scaling)
    # ============================================
    X_rlt = X_imputed.copy()
    y_rlt = y.copy()

    return X_skl, y, X_rlt, y_rlt
