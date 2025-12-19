import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)
from sklearn.linear_model import Lasso

from rlt_python import RLTRegressor   # Ton modèle RLT Python !


def modeling_regression(X_skl, y, X_rlt, y_rlt, dataset_name):
    """
    Train regression models : RF, ET, Boosting, Lasso, and Python RLT.
    """

    results = []

    # -------------------------
    #  Train/test split
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_skl, y, test_size=0.2, random_state=42
    )

    n_features = X_skl.shape[1]

    # -------------------------
    # Define sklearn models
    # -------------------------
    models = {
        "RF": RandomForestRegressor(),
        "RF_sqrtp": RandomForestRegressor(max_features="sqrt"),
        "RF_logp": RandomForestRegressor(max_features=max(1, int(np.log(n_features)))),
        "ET": ExtraTreesRegressor(),
        "Boosting": GradientBoostingRegressor(),
        "Lasso": Lasso(alpha=0.01)
    }

    # -------------------------
    # Train classical models
    # -------------------------
    for name, model in models.items():
        print(f"Running {name}...")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        rmse = mean_squared_error(y_test, pred) ** 0.5
        r2 = r2_score(y_test, pred)

        results.append({
            "dataset": dataset_name,
            "model": name,
            "rmse": rmse,
            "r2": r2
        })

    # -------------------------
    #  Run Python RLT
    # -------------------------
    print("Running RLT (Python)...")

    try:
        rlt = RLTRegressor(
            ntrees=100,
            min_samples_leaf=5,
            combsplit=3,
            reinforcement=True,
            muting_percent=0.4,
            random_state=42
        )

        rlt.fit(X_train, y_train)

        pred_rlt = rlt.predict(X_test)

        rmse_rlt = mean_squared_error(y_test, pred_rlt) ** 0.5
        r2_rlt = r2_score(y_test, pred_rlt)

        results.append({
            "dataset": dataset_name,
            "model": "RLT",
            "rmse": rmse_rlt,
            "r2": r2_rlt
        })

    except Exception as e:
        print(f"❌ RLT FAILED: {e}")
        results.append({
            "dataset": dataset_name,
            "model": "RLT",
            "rmse": np.nan,
            "r2": np.nan
        })

    return pd.DataFrame(results)
