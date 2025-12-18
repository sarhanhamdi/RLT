# RLT/Models/et.py

from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier


def make_et_regressor(
    random_state: int = 42,
    n_estimators: int = 500,
    max_features = "sqrt",
):
    """
    ExtraTrees pour la régression.
    """
    return ExtraTreesRegressor(
        n_estimators=n_estimators,
        max_features=max_features,
        random_state=random_state,
        n_jobs=-1,
    )


def make_et_classifier(
    random_state: int = 42,
    n_estimators: int = 200,
    max_features = "sqrt",
):
    """
    ExtraTrees pour la classification.
    """
    return ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        random_state=random_state,
        n_jobs=-1,
    )
