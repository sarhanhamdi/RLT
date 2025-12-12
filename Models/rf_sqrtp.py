# rf_sqrtp.py
# Random Forest Regressor avec max_features="sqrt"

from sklearn.ensemble import RandomForestRegressor


def get_rf_sqrtp_model(**kwargs):
    """
    Retourne un modèle Random Forest Regressor avec max_features="sqrt".
    
    Args:
        **kwargs: Paramètres additionnels pour RandomForestRegressor
        
    Returns:
        RandomForestRegressor: Modèle configuré avec max_features="sqrt"
    """
    return RandomForestRegressor(max_features="sqrt", **kwargs)

