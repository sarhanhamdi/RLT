# rf_logp.py
# Random Forest Regressor avec max_features=log(n_features)

import numpy as np
from sklearn.ensemble import RandomForestRegressor


def get_rf_logp_model(n_features=None, **kwargs):
    """
    Retourne un modèle Random Forest Regressor avec max_features=log(n_features).
    
    Args:
        n_features: Nombre de features (utilisé pour calculer max_features)
        **kwargs: Paramètres additionnels pour RandomForestRegressor
        
    Returns:
        RandomForestRegressor: Modèle configuré avec max_features=log(n_features)
    """
    if n_features is None:
        # Valeur par défaut si n_features n'est pas fourni
        max_features = None
    else:
        max_features = max(1, int(np.log(n_features)))
    
    return RandomForestRegressor(max_features=max_features, **kwargs)

