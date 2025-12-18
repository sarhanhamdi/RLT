# rf.py
# Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor


def get_rf_model(**kwargs):
    """
    Retourne un modèle Random Forest Regressor.
    
    Args:
        **kwargs: Paramètres additionnels pour RandomForestRegressor
        
    Returns:
        RandomForestRegressor: Modèle configuré
    """
    return RandomForestRegressor(**kwargs)

