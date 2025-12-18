#!/usr/bin/env python
# coding: utf-8

"""
pipeline_model_regression.py
Pipeline de modeling pour la régression utilisant les datasets préparés.
Fusionne run_all_regression, modeling_regression et run_single_regression.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Models"))

# Importer les modèles
from Models import MODEL_REGISTRY

# ============================================
# CONFIGURATION DES CHEMINS
# ============================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PREPARED_DIR = os.path.join(BASE_DIR, "data_prepared")
OUTPUT_DIR = BASE_DIR


# ============================================
# LISTE DES DATASETS DE RÉGRESSION
# ============================================

REG_DATASETS = [
    "auto-mpg",
    "concrete_data",
    "HousingData",
    "ozone",
    "winequality-red",
    "winequality-white",
    "scenario1_p1000",
    "scenario1_p200",
    "scenario1_p500",
    "scenario2_p1000",
    "scenario2_p200",
    "scenario2_p500",
    "scenario3_p1000",
    "scenario3_p200",
    "scenario3_p500",
    "scenario4_p1000",
    "scenario4_p200",
    "scenario4_p500",
]


# ============================================
# FONCTION DE MODELING POUR UN DATASET
# ============================================

def modeling_regression_dataset(dataset_name):
    """
    Entraîne tous les modèles de régression sur un dataset préparé.
    
    Args:
        dataset_name: Nom du dataset (ex: 'auto-mpg')
        
    Returns:
        DataFrame avec les résultats pour ce dataset
    """
    # Charger le dataset préparé
    prepared_file = os.path.join(DATA_PREPARED_DIR, f"{dataset_name}_prepared.csv")
    
    if not os.path.exists(prepared_file):
        print(f"❌ Fichier préparé non trouvé: {prepared_file}")
        return None
    
    print(f"\n{'='*60}")
    print(f"  TRAITEMENT: {dataset_name}")
    print(f"{'='*60}")
    
    try:
        # Charger le dataset
        df = pd.read_csv(prepared_file)
        
        # Identifier la colonne target (dernière colonne)
        target_col = df.columns[-1]
        
        # Séparer X et y
        X = df.drop(columns=[target_col]).values
        y = df[target_col].values
        
        print(f"   Dataset shape: {df.shape}")
        print(f"   Target: {target_col}")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        n_samples = X_train.shape[0]
        n_features = X_train.shape[1]
        
        results = []
        
        # ============================================
        # LISTE DES MODÈLES À ENTRÂINER
        # ============================================
        models_to_train = [
            {
                'name': 'RF',
                'get_model': MODEL_REGISTRY['RF'],
                'kwargs': {'random_state': 42},
                'use_all_features': True
            },
            {
                'name': 'RF_sqrtp',
                'get_model': MODEL_REGISTRY['RF_sqrtp'],
                'kwargs': {'random_state': 42},
                'use_all_features': True
            },
            {
                'name': 'RF_logp',
                'get_model': MODEL_REGISTRY['RF_logp'],
                'kwargs': {'n_features': n_features, 'random_state': 42},
                'use_all_features': False  # Utilise log(n_features) features
            },
            {
                'name': 'ET',
                'get_model': MODEL_REGISTRY['ET'],
                'kwargs': {'random_state': 42},
                'use_all_features': True
            },
            {
                'name': 'Boosting',
                'get_model': MODEL_REGISTRY['Boosting'],
                'kwargs': {'random_state': 42},
                'use_all_features': True
            },
            {
                'name': 'Lasso',
                'get_model': MODEL_REGISTRY['Lasso'],
                'kwargs': {'alpha': 0.01, 'random_state': 42},
                'use_all_features': False  # Lasso fait de la sélection de features
            },
            {
                'name': 'RLT',
                'get_model': MODEL_REGISTRY['RLT'],
                'kwargs': {'dataset_name': dataset_name, 'n_samples': n_samples},
                'use_all_features': False  # RLT peut faire de la sélection
            }
        ]
        
        # ============================================
        # ENTRÂINER TOUS LES MODÈLES
        # ============================================
        for model_config in models_to_train:
            model_name = model_config['name']
            get_model_func = model_config['get_model']
            model_kwargs = model_config['kwargs']
            use_all = model_config['use_all_features']
            
            print(f"   Running {model_name}...", end=" ")
            
            try:
                # Obtenir le modèle
                model = get_model_func(**model_kwargs)
                
                # Entraîner le modèle
                model.fit(X_train, y_train)
                
                # Prédire
                y_pred = model.predict(X_test)
                
                # Calculer les métriques
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                results.append({
                    "dataset": dataset_name,
                    "model": model_name,
                    "rmse": round(rmse, 6),
                    "r2": round(r2, 6),
                    "n_features_used": n_features if use_all else "subset"
                })
                
                print(f"✓ (RMSE: {rmse:.4f}, R²: {r2:.4f})")
                
            except Exception as e:
                print(f"❌ FAILED: {e}")
                results.append({
                    "dataset": dataset_name,
                    "model": model_name,
                    "rmse": np.nan,
                    "r2": np.nan,
                    "n_features_used": "error"
                })
        
        return pd.DataFrame(results)
        
    except Exception as e:
        print(f"❌ ERREUR lors du traitement de {dataset_name}: {e}")
        return None


# ============================================
# FONCTION PRINCIPALE
# ============================================

def run_all_regression():
    """
    Exécute tous les modèles de régression sur tous les datasets préparés.
    Sauvegarde les résultats dans un fichier CSV.
    """
    print("\n" + "="*60)
    print("  PIPELINE DE MODELING - RÉGRESSION")
    print("  Tous les modèles sur tous les datasets préparés")
    print("="*60)
    
    all_results = []
    successful = []
    failed = []
    
    # Parcourir tous les datasets
    for i, dataset_name in enumerate(REG_DATASETS, 1):
        print(f"\n[{i}/{len(REG_DATASETS)}] {dataset_name}...")
        
        results_df = modeling_regression_dataset(dataset_name)
        
        if results_df is not None and not results_df.empty:
            all_results.append(results_df)
            successful.append(dataset_name)
        else:
            failed.append(dataset_name)
    
    # ============================================
    # SAUVEGARDE DES RÉSULTATS COMBINÉS
    # ============================================
    
    print("\n" + "="*60)
    print("  RÉSUMÉ")
    print("="*60)
    print(f"  ✓ Datasets réussis: {len(successful)}/{len(REG_DATASETS)}")
    print(f"  ❌ Datasets échoués: {len(failed)}/{len(REG_DATASETS)}")
    
    if successful:
        print(f"\n  Datasets réussis: {', '.join(successful)}")
    if failed:
        print(f"\n  Datasets échoués: {', '.join(failed)}")
    
    if len(all_results) > 0:
        # Combiner tous les résultats
        df_final = pd.concat(all_results, ignore_index=True)
        
        # Sauvegarder le fichier final
        output_file = os.path.join(OUTPUT_DIR, "regression_results.csv")
        df_final.to_csv(output_file, index=False)
        
        print(f"\n{'='*60}")
        print(f"  ✓ RÉSULTATS SAUVEGARDÉS")
        print(f"  → {output_file}")
        print(f"{'='*60}")
        print(f"\n  Total: {len(df_final)} résultats")
        print(f"  Datasets: {df_final['dataset'].nunique()}")
        print(f"  Modèles: {df_final['model'].nunique()}")
        
        # Afficher un aperçu
        print(f"\n  Aperçu des résultats (premières lignes):")
        print(df_final.head(20).to_string(index=False))
        
        # Statistiques par dataset
        print(f"\n  Meilleur R² par dataset:")
        for dataset in sorted(df_final['dataset'].unique()):
            dataset_results = df_final[df_final['dataset'] == dataset]
            best = dataset_results.loc[dataset_results['r2'].idxmax()]
            print(f"    {dataset:25s} → {best['model']:12s} (R² = {best['r2']:.4f})")
        
        return df_final
    else:
        print("\n  ❌ Aucun résultat n'a pu être généré.")
        return None


# ============================================
# POINT D'ENTRÉE PRINCIPAL
# ============================================

if __name__ == "__main__":
    run_all_regression()




