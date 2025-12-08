# prep_classification.py

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from utils_cleaning import (
    drop_empty_columns,
    replace_invalid_values,
    convert_numeric_features,
    fix_winequality
)


def prepare_classification(df, target, dataset_name=None):

    # ============================================================
    # 1) CLEAN RAW DATA
    # ============================================================
    df = drop_empty_columns(df)
    df = replace_invalid_values(df)

    if dataset_name in ["winequality-red", "winequality-white"]:
        df = fix_winequality(dataset_name + ".csv")

    df = convert_numeric_features(df, target=target)

    # Remove rows where target is missing
    df = df.dropna(subset=[target])

    # Remove rows where ALL features are NaN
    feature_cols = [c for c in df.columns if c != target]
    df = df.dropna(subset=feature_cols, how="all")

    # ============================================================
    # 2) TARGET ENCODING
    # ============================================================
    y = df[target]
    if y.dtype == object:
        y = pd.factorize(y)[0]

    X = df.drop(columns=[target])

    # Remove ID for RLT later
    if "id" in X.columns:
        X = X.drop(columns=["id"])

    # ============================================================
    # 3) SKLEARN PIPELINE (unchanged)
    # ============================================================
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore"))
    ])

    transformer = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols),
    ])

    X_sklearn = transformer.fit_transform(X)

    # ============================================================
    # 4) RLT PREPARATION
    # ============================================================
    X_rlt = X.copy()

    # Force numeric
    for col in X_rlt.columns:
        X_rlt[col] = pd.to_numeric(X_rlt[col], errors="coerce")

    # Fill NaN with column median, if median is NaN → replace by 0
    for col in X_rlt.columns:
        med = X_rlt[col].median()
        if pd.isna(med):
            med = 0
        X_rlt[col] = X_rlt[col].fillna(med)

    # Standardize for RLT (VERY IMPORTANT)
    scaler = StandardScaler()
    X_rlt = pd.DataFrame(scaler.fit_transform(X_rlt), columns=X_rlt.columns)

    return X_sklearn, y, X_rlt, y
