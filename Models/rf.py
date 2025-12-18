# RLT/Models/rf.py

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


def make_rf_regressor(
    random_state: int = 42,
    n_estimators: int = 500,
    max_features = "sqrt",
):
    """
    Construit un RandomForestRegressor avec des hyperparamètres par défaut.

    random_state : graine pour la reproductibilité.
    n_estimators : nombre d'arbres.
    max_features : nombre de features candidates à chaque split.
    """
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_features=max_features,
        random_state=random_state,
        n_jobs=-1,
    )


def make_rf_classifier(
    random_state: int = 42,
    n_estimators: int = 200,
    max_features = "sqrt",
):
    """
    Construit un RandomForestClassifier avec des hyperparamètres par défaut.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        random_state=random_state,
        n_jobs=-1,
    )
