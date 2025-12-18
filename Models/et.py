# et.py
# Extra Trees Regressor

from sklearn.ensemble import ExtraTreesRegressor


def get_et_model(**kwargs):
    """
    Retourne un modèle Extra Trees Regressor.
    
    Args:
        **kwargs: Paramètres additionnels pour ExtraTreesRegressor
        
    Returns:
        ExtraTreesRegressor: Modèle configuré
    """
    return ExtraTreesRegressor(**kwargs)

