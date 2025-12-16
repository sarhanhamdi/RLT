# RLT/Pipelines/modeling.py

import os
import csv
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

from Models.rlt import RLTRegressor, RLTClassifier

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def run_experiments(X_train, X_test, y_train, y_test, config):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    task_type = config.get("task_type", "regression")
    random_state = config.get("random_state", 42)

    results = []

    if task_type == "regression":
        # Baseline : RandomForestRegressor
        rf = RandomForestRegressor(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        mse_rf = mean_squared_error(y_test, y_pred_rf)
        results.append(("RandomForestRegressor", mse_rf))

        # RLT régression
        rlt = RLTRegressor(
            n_estimators=10,
            random_state=random_state
        )
        rlt.fit(X_train, y_train)
        y_pred_rlt = rlt.predict(X_test)
        mse_rlt = mean_squared_error(y_test, y_pred_rlt)
        results.append(("RLTRegressor", mse_rlt))

        out_file = os.path.join(RESULTS_DIR, "metrics_regression.csv")
        write_results(out_file, results, config, metric_name="MSE")

    else:  # classification
        # Baseline : RandomForestClassifier
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        acc_rf = accuracy_score(y_test, y_pred_rf)
        results.append(("RandomForestClassifier", acc_rf))

        # RLT classification
        rlt = RLTClassifier(
            n_estimators=10,
            random_state=random_state
        )
        rlt.fit(X_train, y_train)
        y_pred_rlt = rlt.predict(X_test)
        acc_rlt = accuracy_score(y_test, y_pred_rlt)
        results.append(("RLTClassifier", acc_rlt))

        out_file = os.path.join(RESULTS_DIR, "metrics_classification.csv")
        write_results(out_file, results, config, metric_name="ACC")


def write_results(path, results, config, metric_name):
    header = ["dataset", "task_type", "model", "metric", "value", "scenario"]
    file_exists = os.path.exists(path)

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        for model_name, metric_value in results:
            writer.writerow([
                config.get("dataset_name", "unknown"),
                config.get("task_type", "regression"),
                model_name,
                metric_name,
                metric_value,
                config.get("scenario", "default")
            ])
