# RLT/Models/linear_models.py

from sklearn.linear_model import Lasso, ElasticNet, LogisticRegression


def make_lasso(random_state: int = 42, alpha: float = 0.01):
    """
    Lasso pour la régression (données déjà standardisées par data_preparation).
    """
    return Lasso(alpha=alpha, random_state=random_state)


def make_elasticnet(
    random_state: int = 42,
    alpha: float = 0.01,
    l1_ratio: float = 0.5,
):
    """
    ElasticNet pour la régression.
    """
    return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)


def make_logreg_l2(
    random_state: int = 42,
    C: float = 1.0,
):
    """
    Régression logistique L2 pour classification binaire/multiclasse.
    """
    return LogisticRegression(
        penalty="l2",
        C=C,
        solver="liblinear",
        random_state=random_state,
    )
