# RLT/Models/gb.py

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier


def make_gbr(random_state: int = 42):
    """
    Gradient Boosting pour la régression.
    """
    return GradientBoostingRegressor(random_state=random_state)


def make_gbc(random_state: int = 42):
    """
    Gradient Boosting pour la classification.
    """
    return GradientBoostingClassifier(random_state=random_state)
