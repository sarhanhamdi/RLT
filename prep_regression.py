# prep_regression.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from utils_cleaning import (
    drop_empty_columns,
    replace_invalid_values,
    convert_numeric_features,
    fix_winequality,
)

def prepare_regression(df, target, dataset_name=None):

    # CLEAN RAW
    df = drop_empty_columns(df)
    df = replace_invalid_values(df)

    # WINE QUALITY FIX
    if dataset_name in ["winequality-red", "winequality-white"]:
        df = fix_winequality(dataset_name + ".csv")

    # Remove missing target
    df = df.dropna(subset=[target])

    # Convert numeric
    df = convert_numeric_features(df, target)

    # Remove useless columns
    cols_to_remove = ["car name", "name", "CarName", "model", "id"]
    for col in cols_to_remove:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Separate X / y
    y = df[target]
    X = df.drop(columns=[target])

    # ==========================
    # SKLEARN PIPELINE
    # ==========================
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore"))
    ])

    transformer = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])

    X_sklearn = transformer.fit_transform(X)

    # ==========================
    # RLT VERSION
    # ==========================
    X_rlt = X.copy()

    for col in X_rlt.columns:
        X_rlt[col] = pd.to_numeric(X_rlt[col], errors="coerce")

    for col in X_rlt.columns:
        med = X_rlt[col].median()
        if pd.isna(med):
            med = 0
        X_rlt[col] = X_rlt[col].fillna(med)

    scaler = StandardScaler()
    X_rlt = pd.DataFrame(scaler.fit_transform(X_rlt), columns=X_rlt.columns)

    return X_sklearn, y, X_rlt, y
