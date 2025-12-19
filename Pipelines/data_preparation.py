# RLT/Pipelines/data_preparation.py

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from Pipelines.data_understanding import TARGET_COLS, infer_task_type

# Dossier des données réelles
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Data")


# =========================
#   DONNÉES RÉELLES
# =========================

def load_raw_dataset(dataset_name: str, target_col: str | None = None):
    """
    Charge un dataset brut (Data/<dataset_name>.csv), renvoie (df, target_col).
    """
    path = os.path.join(DATA_DIR, f"{dataset_name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable: {path}")

    df = pd.read_csv(path)

    # Déterminer la colonne cible
    if target_col is None:
        target_col = TARGET_COLS.get(dataset_name, None)
    if target_col is None:
        target_col = df.columns[-1]
        print(
            f"[INFO] target_col non spécifié pour {dataset_name}, "
            f"on utilise la dernière colonne: {target_col}"
        )
    if target_col not in df.columns:
        raise ValueError(
            f"target_col='{target_col}' n'existe pas dans {dataset_name}. "
            f"Colonnes disponibles: {list(df.columns)}"
        )

    return df, target_col


def prepare_features_and_target(df: pd.DataFrame, target_col: str):
    """
    Sépare X/y et prépare les features pour les données réelles :
      - imputation des NaN (numérique: moyenne, catégoriel: modalité la plus fréquente)
      - standardisation des colonnes numériques
      - one-hot encoding des colonnes catégorielles.

    Renvoie:
      X : np.array (features prêtes pour les modèles)
      y : np.array (cible brute)
      meta : infos sur la préparation (noms de colonnes, scalers, etc.).
    """
    # 1) Séparer cible et features
    y = df[target_col].values
    X_df = df.drop(columns=[target_col])

    # 2) Séparer numériques / catégorielles
    num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_df.columns if c not in num_cols]

    X_num = X_df[num_cols].values if num_cols else None
    X_cat = X_df[cat_cols].astype("category") if cat_cols else None

    # 3) Imputation + standardisation des numériques
    num_imputer = None
    scaler = None
    if X_num is not None:
        num_imputer = SimpleImputer(strategy="mean")
        X_num_imputed = num_imputer.fit_transform(X_num)

        scaler = StandardScaler()
        X_num_scaled = scaler.fit_transform(X_num_imputed)
    else:
        X_num_scaled = None

    # 4) Imputation + encodage des catégorielles
    cat_imputer = None
    ohe = None
    X_cat_encoded = None
    if X_cat is not None and X_cat.shape[1] > 0:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X_cat_imputed = cat_imputer.fit_transform(X_cat)

        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_cat_encoded = ohe.fit_transform(X_cat_imputed)

    # 5) Concaténation des features
    if X_num_scaled is not None and X_cat_encoded is not None:
        X = np.hstack([X_num_scaled, X_cat_encoded])
    elif X_num_scaled is not None:
        X = X_num_scaled
    elif X_cat_encoded is not None:
        X = X_cat_encoded
    else:
        raise ValueError("Aucune feature utilisable (numérique ou catégorielle).")

    # 6) Noms de features après encodage
    feature_names = []
    if num_cols:
        feature_names.extend(num_cols)
    if cat_cols and ohe is not None:
        cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
        feature_names.extend(cat_feature_names)

    meta = {
        "source": "real",
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "feature_names": feature_names,
        "num_imputer": num_imputer,
        "cat_imputer": cat_imputer,
        "scaler": scaler,
        "encoder": ohe,
    }

    return X, y, meta


# =========================
#   DONNÉES SIMULÉES (scénarios de l'article)
# =========================

def generate_simulated_data(config: dict):
    """
    Génère des données simulées conformes aux 4 scénarios de Zhu (2015).

    Paramètres dans config["simulation_params"] :
      - p : nombre de variables (200, 500, 1000, etc.)
      - n (optionnel) : si non fourni, on utilise les N de l'article :
          scenario1: 100, scenario2: 100, scenario3: 300, scenario4: 200.

    config["scenario"] doit être : "scenario1", "scenario2", "scenario3", "scenario4".
    """
    sim_params = config.get("simulation_params", {})
    p = sim_params.get("p", 200)

    scenario = config.get("scenario", "scenario1")
    random_state = config.get("random_state", 42)
    rng = np.random.default_rng(random_state)

    # Choix de N si non donné
    if "n" in sim_params:
        n = sim_params["n"]
    else:
        if scenario == "scenario1":
            n = 100
        elif scenario == "scenario2":
            n = 100
        elif scenario == "scenario3":
            n = 300
        elif scenario == "scenario4":
            n = 200
        else:
            raise ValueError(f"Scénario simulé inconnu: {scenario}")

    # -------- Scenario 1: classification, X ~ Unif[0,1]^p ----------
    if scenario == "scenario1":
        X = rng.uniform(0.0, 1.0, size=(n, p))

        # mu_i = Phi( 10*(X1 - 1) + 20*|X2 - 0.5| ), Phi cdf normale
        from math import sqrt, erf

        def phi_cdf(z):
            return 0.5 * (1.0 + erf(z / sqrt(2.0)))

        lin = 10.0 * (X[:, 0] - 1.0) + 20.0 * np.abs(X[:, 1] - 0.5)
        mu = np.array([phi_cdf(z) for z in lin])
        y = (rng.random(n) < mu).astype(int)
        task_type = "classification"

    # -------- Scenario 2: non-linear regression, X ~ Unif[0,1]^p ----------
    elif scenario == "scenario2":
        X = rng.uniform(0.0, 1.0, size=(n, p))
        part2 = np.maximum(X[:, 1] - 0.25, 0.0)
        f = 100.0 * (X[:, 0] - 0.5) ** 2 * part2
        eps = rng.normal(loc=0.0, scale=1.0, size=n)
        y = f + eps
        task_type = "regression"

    # -------- Scenario 3: checkerboard-like regression with strong correlation ----------
    elif scenario == "scenario3":
        if p < 200:
            raise ValueError("Scenario3 nécessite p >= 200")
        idx = np.arange(p)
        Sigma = 0.9 ** np.abs(idx[:, None] - idx[None, :])
        X = rng.multivariate_normal(mean=np.zeros(p), cov=Sigma, size=n)
        eps = rng.normal(loc=0.0, scale=1.0, size=n)
        y = (
            2.0 * X[:, 49] * X[:, 99]
            + 2.0 * X[:, 149] * X[:, 199]
            + eps
        )
        task_type = "regression"

    # -------- Scenario 4: linear regression, correlated X ----------
    elif scenario == "scenario4":
        if p < 150:
            raise ValueError("Scenario4 nécessite p >= 150")
        idx = np.arange(p)
        base = 0.5 ** np.abs(idx[:, None] - idx[None, :])
        Sigma = base + 0.2 * (1 - np.eye(p))
        X = rng.multivariate_normal(mean=np.zeros(p), cov=Sigma, size=n)
        eps = rng.normal(loc=0.0, scale=1.0, size=n)
        y = 2.0 * X[:, 49] + 2.0 * X[:, 99] + 4.0 * X[:, 149] + eps
        task_type = "regression"

    else:
        raise ValueError(f"Scénario simulé inconnu: {scenario}")

    feature_names = [f"X{i+1}" for i in range(p)]
    meta = {
        "source": "simulated",
        "scenario": scenario,
        "task_type": task_type,
        "feature_names": feature_names,
        "n": n,
        "p": p,
    }

    return X, y, meta


# =========================
#   DISPATCH GÉNÉRAL
# =========================

def load_and_prepare_data(config: dict):
    """
    Pipeline générale de préparation :
      - source = "real"      : charge CSV, prépare X/y, split train/test.
      - source = "simulated" : génère un scénario, split train/test.

    Renvoie X_train, X_test, y_train, y_test, meta.
    """
    source = config.get("source", "real")
    random_state = config.get("random_state", 42)

    if source == "real":
        dataset_name = config.get("dataset_name")
        if dataset_name is None:
            raise ValueError("config['dataset_name'] doit être spécifié pour source='real'.")

        df, target_col = load_raw_dataset(dataset_name)
        X, y, meta = prepare_features_and_target(df, target_col)

        task_type = config.get("task_type")
        if task_type is None:
            task_type = infer_task_type(y)
        meta["task_type"] = task_type
        meta["target_col"] = target_col

    elif source == "simulated":
        X, y, meta = generate_simulated_data(config)
        task_type = meta["task_type"]

    else:
        raise ValueError(f"source inconnue: {source} (attendu 'real' ou 'simulated').")

    # Split train/test
    if task_type == "classification":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=random_state, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=random_state
        )

    return X_train, X_test, y_train, y_test, meta
