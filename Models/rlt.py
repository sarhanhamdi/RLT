# rlt.py
# Reinforcement Learning Trees Regressor

import sys
import os

# rlt_python.py est dans le même dossier Models/
# Import direct depuis le même dossier
from .rlt_python import RLTRegressor


def get_rlt_model(dataset_name=None, n_samples=None, **kwargs):
    """
    Retourne un modèle RLT Regressor avec paramètres adaptés selon le dataset.
    
    Args:
        dataset_name: Nom du dataset (pour ajuster les paramètres)
        n_samples: Nombre d'échantillons (pour ajuster les paramètres)
        **kwargs: Paramètres additionnels pour RLTRegressor
        
    Returns:
        RLTRegressor: Modèle configuré avec paramètres adaptés
    """
    # Paramètres par défaut
    default_params = {
        'min_samples_leaf': 5,
        'combsplit': 3,
        'reinforcement': True,
        'muting_percent': 0.4,
        'random_state': 42
    }
    
    # Ajuster le nombre d'arbres selon le dataset
    if dataset_name and "winequality-white" in dataset_name.lower():
        n_trees = 5  # Very few trees for very large white wine dataset
    elif dataset_name and "winequality-red" in dataset_name.lower():
        n_trees = 10  # Fewer trees for large red wine dataset
    elif n_samples and n_samples > 400:
        n_trees = 20  # Medium trees for medium datasets
    else:
        n_trees = 50  # Standard for smaller datasets
    
    default_params['ntrees'] = n_trees
    
    # Fusionner avec les kwargs fournis (kwargs ont priorité)
    params = {**default_params, **kwargs}
    
    return RLTRegressor(**params)

