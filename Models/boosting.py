# boosting.py
# Gradient Boosting Regressor

from sklearn.ensemble import GradientBoostingRegressor


def get_boosting_model(**kwargs):
    """
    Retourne un modèle Gradient Boosting Regressor.
    
    Args:
        **kwargs: Paramètres additionnels pour GradientBoostingRegressor
        
    Returns:
        GradientBoostingRegressor: Modèle configuré
    """
    return GradientBoostingRegressor(**kwargs)

