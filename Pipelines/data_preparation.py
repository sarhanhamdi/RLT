# RLT/Pipelines/data_preparation.py

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from Pipelines.data_understanding import TARGET_COLS, infer_task_type

# Dossier des données
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Data")


# =========================
#   DONNÉES RÉELLES
# =========================

def load_raw_dataset(dataset_name: str, target_col: str | None = None):
    """
    Charge un dataset brut (Data/<dataset_name>.csv), renvoie df, target_col.
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
    Sépare X/y et gère num/cat :
      - imputation NaN
      - X_num standardisé (StandardScaler)
      - X_cat one-hot encodé
    Renvoie X (np.array), y (np.array), meta dict.
    """
    # Séparer cible
    y = df[target_col].values
    X_df = df.drop(columns=[target_col])

    # Séparer numériques / catégorielles
    num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_df.columns if c not in num_cols]

    X_num = X_df[num_cols].values if num_cols else None
    X_cat = X_df[cat_cols].astype("category") if cat_cols else None

    # 1) Imputation + standardisation des numériques
    num_imputer = None
    scaler = None
    if X_num is not None:
        num_imputer = SimpleImputer(strategy="mean")
        X_num_imputed = num_imputer.fit_transform(X_num)

        scaler = StandardScaler()
        X_num_scaled = scaler.fit_transform(X_num_imputed)
    else:
        X_num_scaled = None

    # 2) Imputation + encodage des catégorielles
    cat_imputer = None
    ohe = None
    X_cat_encoded = None
    if X_cat is not None and X_cat.shape[1] > 0:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X_cat_imputed = cat_imputer.fit_transform(X_cat)

        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_cat_encoded = ohe.fit_transform(X_cat_imputed)

    # 3) Concaténation
    if X_num_scaled is not None and X_cat_encoded is not None:
        X = np.hstack([X_num_scaled, X_cat_encoded])
    elif X_num_scaled is not None:
        X = X_num_scaled
    elif X_cat_encoded is not None:
        X = X_cat_encoded
    else:
        raise ValueError("Aucune feature utilisable (numérique ou catégorielle).")

    # 4) Noms de features après encodage
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
#   DONNÉES SIMULÉES
# =========================

def generate_simulated_data(config: dict):
    """
    Génère des données simulées pour différents scénarios.
    Paramètres dans config["simulation_params"] :
      - n: nombre d'observations (par défaut 500)
      - p: nombre de variables (par défaut 200, comme dans l'article)
    config["scenario"] peut être : "simu1", "simu2", "simu3", "simu4"
    """
    sim_params = config.get("simulation_params", {})
    n = sim_params.get("n", 500)
    p = sim_params.get("p", 200)
    random_state = config.get("random_state", 42)
    rng = np.random.default_rng(random_state)

    scenario = config.get("scenario", "simu1")

    # X ~ N(0,1) indépendantes
    X = rng.normal(size=(n, p))

    # Scénario 1 : régression linéaire sparse
    if scenario == "simu1":
        beta = np.zeros(p)
        beta[:5] = [3, -2, 1.5, 0.5, -1]
        noise = rng.normal(scale=1.0, size=n)
        y = X @ beta + noise
        task_type = "regression"

    # Scénario 2 : régression non-linéaire (polynomiale + interaction)
    elif scenario == "simu2":
        f = (
            2.0 * np.sin(X[:, 0])
            + 1.5 * (X[:, 1] ** 2)
            - 3.0 * X[:, 2] * X[:, 3]
        )
        noise = rng.normal(scale=1.0, size=n)
        y = f + noise
        task_type = "regression"

    # Scénario 3 : classification logistique sparse
    elif scenario == "simu3":
        beta = np.zeros(p)
        beta[:5] = [1.2, -1.0, 0.8, 0.5, -0.7]
        lin_pred = X @ beta
        prob = 1 / (1 + np.exp(-lin_pred))
        y = (rng.random(n) < prob).astype(int)
        task_type = "classification"

    # Scénario 4 : checkerboard (interaction forte sur X1, X2)
    elif scenario == "simu4":
        X = np.hstack([
            rng.uniform(0, 1, size=(n, 2)),
            rng.normal(size=(n, p - 2))
        ])
        y = (
            ((X[:, 0] > 0.5) & (X[:, 1] > 0.5)) |
            ((X[:, 0] <= 0.5) & (X[:, 1] <= 0.5))
        ).astype(int)
        task_type = "classification"

    else:
        raise ValueError(f"Scénario simulé inconnu: {scenario}")

    feature_names = [f"X{i+1}" for i in range(p)]
    meta = {
        "source": "simulated",
        "scenario": scenario,
        "task_type": task_type,
        "feature_names": feature_names,
    }

    return X, y, meta


# =========================
#   DISPATCH GÉNÉRAL
# =========================

def load_and_prepare_data(config: dict):
    """
    Pipeline générale data_preparation :
      - source = "real"      -> données réelles (CSV, standardisation, encodage)
      - source = "simulated" -> scénarios de simulation
    Renvoie X_train, X_test, y_train, y_test, meta
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
