# RLT/Pipelines/modeling.py

import os
import csv
from sklearn.metrics import mean_squared_error, accuracy_score

from Models.registry import get_benchmark_models

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def run_experiments(X_train, X_test, y_train, y_test, config: dict):
    """
    Entraîne tous les modèles du benchmark sur (X_train, y_train),
    évalue sur le test, et écrit les métriques dans un CSV.

    - Pour la régression : MSE.
    - Pour la classification : accuracy.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    task_type = config.get("task_type", "regression")
    random_state = config.get("random_state", 42)

    models = get_benchmark_models(task_type, random_state=random_state)

    if task_type == "regression":
        metric_name = "MSE"
        out_file = os.path.join(RESULTS_DIR, "metrics_regression.csv")
    else:
        metric_name = "ACC"
        out_file = os.path.join(RESULTS_DIR, "metrics_classification.csv")

    for name, model in models.items():
        print(f"\n=== Entraînement modèle: {name} ===")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        if task_type == "regression":
            metric_value = mean_squared_error(y_test, y_pred)
        else:
            metric_value = accuracy_score(y_test, y_pred)

        write_results(
            out_file,
            [(name, metric_value)],
            config,
            metric_name=metric_name
        )


def write_results(path: str, results, config: dict, metric_name: str):
    """
    Ajoute les résultats dans le fichier CSV (créé s'il n'existe pas).
    """
    header = ["dataset", "task_type", "model", "metric", "value", "scenario", "source"]
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
                config.get("scenario", "default"),
                config.get("source", "real"),
            ])
