import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Importer les modèles depuis le dossier Models
from Models import MODEL_REGISTRY


def modeling_regression(X_skl, y, X_rlt, y_rlt, dataset_name):
    """
    Train regression models : RF, ET, Boosting, Lasso, and Python RLT.
    Utilise les modèles du dossier Models.
    """

    results = []

    # -------------------------
    #  Train/test split
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_skl, y, test_size=0.2, random_state=42
    )

    n_samples = X_train.shape[0]

    # -------------------------
    # Liste des modèles à entraîner avec leurs paramètres
    # -------------------------
    models_to_train = [
        {
            'name': 'RF',
            'get_model': MODEL_REGISTRY['RF'],
            'kwargs': {}
        },
        {
            'name': 'RF_sqrtp',
            'get_model': MODEL_REGISTRY['RF_sqrtp'],
            'kwargs': {}
        },
        {
            'name': 'RF_logp',
            'get_model': MODEL_REGISTRY['RF_logp'],
            'kwargs': {'n_features': X_skl.shape[1]}
        },
        {
            'name': 'ET',
            'get_model': MODEL_REGISTRY['ET'],
            'kwargs': {}
        },
        {
            'name': 'Boosting',
            'get_model': MODEL_REGISTRY['Boosting'],
            'kwargs': {}
        },
        {
            'name': 'Lasso',
            'get_model': MODEL_REGISTRY['Lasso'],
            'kwargs': {}
        },
        {
            'name': 'RLT',
            'get_model': MODEL_REGISTRY['RLT'],
            'kwargs': {'dataset_name': dataset_name, 'n_samples': n_samples}
        }
    ]

    # -------------------------
    # Entraîner tous les modèles
    # -------------------------
    for model_config in models_to_train:
        model_name = model_config['name']
        get_model_func = model_config['get_model']
        model_kwargs = model_config['kwargs']
        
        print(f"Running {model_name}...")
        
        try:
            # Obtenir le modèle depuis le dossier Models
            model = get_model_func(**model_kwargs)
            
            # Entraîner le modèle
            model.fit(X_train, y_train)
            
            # Prédire
            pred = model.predict(X_test)
            
            # Calculer les métriques
            rmse = mean_squared_error(y_test, pred) ** 0.5
            r2 = r2_score(y_test, pred)
            
            # Afficher des informations supplémentaires pour RLT
            if model_name == 'RLT':
                if hasattr(model, 'ntrees'):
                    n_trees = model.ntrees
                    if 'winequality-white' in dataset_name.lower():
                        print(f"  Using very reduced parameters for {dataset_name} (ntrees={n_trees}, n_samples={n_samples})")
                    elif 'winequality-red' in dataset_name.lower():
                        print(f"  Using reduced parameters for {dataset_name} (ntrees={n_trees}, n_samples={n_samples})")
                    elif n_samples > 400:
                        print(f"  Using medium parameters for {dataset_name} (ntrees={n_trees}, n_samples={n_samples})")
                    else:
                        print(f"  Using standard parameters for {dataset_name} (ntrees={n_trees}, n_samples={n_samples})")
            
            results.append({
                "dataset": dataset_name,
                "model": model_name,
                "rmse": rmse,
                "r2": r2
            })
            
        except Exception as e:
            print(f"❌ {model_name} FAILED: {e}")
            results.append({
                "dataset": dataset_name,
                "model": model_name,
                "rmse": np.nan,
                "r2": np.nan
            })

    return pd.DataFrame(results)
