# Models package
# Contient tous les modèles de régression

from .rf import get_rf_model
from .rf_sqrtp import get_rf_sqrtp_model
from .rf_logp import get_rf_logp_model
from .et import get_et_model
from .boosting import get_boosting_model
from .lasso import get_lasso_model
from .rlt import get_rlt_model

__all__ = [
    'get_rf_model',
    'get_rf_sqrtp_model',
    'get_rf_logp_model',
    'get_et_model',
    'get_boosting_model',
    'get_lasso_model',
    'get_rlt_model',
]

# Mapping des noms de modèles vers leurs fonctions
MODEL_REGISTRY = {
    'RF': get_rf_model,
    'RF_sqrtp': get_rf_sqrtp_model,
    'RF_logp': get_rf_logp_model,
    'ET': get_et_model,
    'Boosting': get_boosting_model,
    'Lasso': get_lasso_model,
    'RLT': get_rlt_model,
}

