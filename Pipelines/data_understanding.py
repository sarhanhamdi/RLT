# RLT/Pipelines/data_understanding.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # si tu ne l'as pas, enlève seaborn et la heatmap

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

REAL_DATASETS = [
    "HousingData",
    "parkinsons",
    "sonar",
    "winequality-red",
    "winequality-white",
    "ReplicatedAcousticFeatures-ParkinsonDatabase",
    "ozone",
    "concrete_data",
    "BreastCanDT",
    "auto-mpg",
]

TARGET_COLS = {
    "HousingData": "MEDV",
    "parkinsons": "status",
    "sonar": "R",
    "winequality-white": "quality",
    "winequality-red": "quality",
    "ReplicatedAcousticFeatures-ParkinsonDatabase": "Status",
    "ozone": "pluie",
    "concrete_data": "concrete_compressive_strength",
    "BreastCanDT": "diagnosis",
    "auto-mpg": "mpg",
}


def infer_task_type(y: np.ndarray) -> str:
    unique_vals = np.unique(y)
    if not np.issubdtype(y.dtype, np.number):
        return "classification"
    if len(unique_vals) <= 10:
        return "classification"
    return "regression"


def summarize_single_dataset(dataset_name: str, target_col: str | None = None) -> dict | None:
    path = os.path.join(DATA_DIR, f"{dataset_name}.csv")
    if not os.path.exists(path):
        print(f"[WARN] Fichier introuvable pour {dataset_name}: {path}")
        return None

    df = pd.read_csv(path)

    if target_col is None:
        target_col = df.columns[-1]
        print(
            f"[INFO] target_col non spécifié pour {dataset_name}, "
            f"on utilise la dernière colonne: {target_col}"
        )

    if target_col not in df.columns:
        print(f"[ERROR] target_col='{target_col}' n'existe pas dans {dataset_name}")
        print(f"Colonnes disponibles: {list(df.columns)}")
        return None

    feature_cols = [c for c in df.columns if c != target_col]
    n_rows, n_cols = df.shape

    X = df[feature_cols].values
    y = df[target_col].values

    task_type = infer_task_type(y)

    print("\n=== Data Understanding ===")
    print(f"Dataset: {dataset_name}")
    print(f"Path: {path}")
    print(f"Target column: {target_col}")
    print(f"Features ({len(feature_cols)}): {feature_cols[:10]}{' ...' if len(feature_cols) > 10 else ''}")
    print(f"Task type (inféré): {task_type}")
    print(f"Nb lignes: {n_rows}, nb colonnes: {n_cols}")
    print(f"y unique (max 10): {np.unique(y)[:10]}")

    target_mean = None
    target_std = None
    if np.issubdtype(y.dtype, np.number):
        target_mean = float(np.mean(y))
        target_std = float(np.std(y))
        print(f"Target mean: {target_mean:.3f}, std: {target_std:.3f}")
    else:
        values, counts = np.unique(y, return_counts=True)
        print("Distribution des classes :")
        for val, cnt in zip(values, counts):
            print(f"  {val}: {cnt} ({cnt / len(y):.2%})")

    dtypes_str = "; ".join([f"{col}:{str(dt)}" for col, dt in df.dtypes.items()])

    # Dossier pour les plots de ce dataset
    plots_dir = os.path.join(RESULTS_DIR, "plots", dataset_name)
    os.makedirs(plots_dir, exist_ok=True)

    # Plot 1 : distribution de la cible
    plt.figure(figsize=(6, 4))
    if np.issubdtype(y.dtype, np.number):
        plt.hist(y, bins=20, edgecolor="black")
        plt.title(f"Distribution de la cible - {dataset_name}")
        plt.xlabel(target_col)
        plt.ylabel("Effectif")
    else:
        values, counts = np.unique(y, return_counts=True)
        plt.bar(values, counts)
        plt.title(f"Distribution des classes - {dataset_name}")
        plt.xlabel(target_col)
        plt.ylabel("Effectif")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "target_distribution.png"))
    plt.close()

    # Corrélations numériques pour plots
    corr_strength = None
    top_corr_features = ""
    numeric_df = df.select_dtypes(include=[np.number])

    if len(feature_cols) <= 30 and np.issubdtype(y.dtype, np.number) and target_col in numeric_df.columns:
        corr = numeric_df.corr()
        cor_target = corr[target_col].drop(target_col)
        if len(cor_target) > 0:
            corr_strength = float(cor_target.abs().mean())
            top5 = cor_target.abs().sort_values(ascending=False).head(4)
            top_corr_features = "; ".join(
                [f"{feat}:{cor_target[feat]:.3f}" for feat in top5.index]
            )
            print("\nTop corrélation (abs) target vs features :")
            for feat in top5.index:
                print(f"  {feat}: corr = {cor_target[feat]:.3f}")

            # Plot 2 : distributions des top features
            df_top = df[list(top5.index)]
            df_top = df_top.iloc[:, :4]  # max 4 features
            n_top = df_top.shape[1]
            plt.figure(figsize=(4 * n_top, 3))
            for i, col in enumerate(df_top.columns):
                plt.subplot(1, n_top, i + 1)
                plt.hist(df_top[col], bins=20, edgecolor="black")
                plt.title(col)
            plt.suptitle(f"Top features corrélées à {target_col} - {dataset_name}")
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(os.path.join(plots_dir, "top_features_hist.png"))
            plt.close()

            # Plot 3 : heatmap de corrélation
            plt.figure(figsize=(max(6, 0.5 * corr.shape[1]), 6))
            sns.heatmap(corr, cmap="coolwarm", center=0)
            plt.title(f"Corrélation numérique - {dataset_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "corr_heatmap.png"))
            plt.close()
    elif len(feature_cols) > 30:
        print("Trop de colonnes pour afficher les corrélations / heatmap.")

    summary = {
        "dataset": dataset_name,
        "task_type_inferred": task_type,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_features": len(feature_cols),
        "target_col": target_col,
        "target_mean": target_mean,
        "target_std": target_std,
        "n_unique_y": int(len(np.unique(y))),
        "dtypes": dtypes_str,
        "mean_abs_corr_target": corr_strength,
        "top_corr_features": top_corr_features,
    }
    return summary


def summarize_all_real_datasets() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    summaries: list[dict] = []
    for name in REAL_DATASETS:
        target_col = TARGET_COLS.get(name, None)
        s = summarize_single_dataset(name, target_col=target_col)
        if s is not None:
            summaries.append(s)

    if summaries:
        out_path = os.path.join(RESULTS_DIR, "data_summary.csv")
        pd.DataFrame(summaries).to_csv(out_path, index=False)
        print(f"\nRésumé global enregistré dans {out_path}")
