# lasso.py
# Lasso Regressor

from sklearn.linear_model import Lasso


def get_lasso_model(alpha=0.01, **kwargs):
    """
    Retourne un modèle Lasso Regressor.
    
    Args:
        alpha: Paramètre de régularisation (défaut: 0.01)
        **kwargs: Paramètres additionnels pour Lasso
        
    Returns:
        Lasso: Modèle configuré
    """
    return Lasso(alpha=alpha, **kwargs)

